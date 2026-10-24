"""Validate the original DENSEGW Fast GPU Newton projector without edits."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import torch

from validate_drot_kkt import frank_wolfe_gap, sparse_kkt
from validate_drot_large import make_candidate
from validation_runtime import preflight


def load_densegw_model(root: Path):
    model_path = root / "model.py"
    spec = importlib.util.spec_from_file_location("densegw_gpu_newton_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_small_candidate(size: int, seed: int, kind: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_part = torch.randn((size, size), generator=generator, dtype=torch.float64)
    identity = torch.eye(size, dtype=torch.float64)
    if kind == "random":
        return random_part
    if kind == "diagonal":
        return identity + 0.05 * random_part
    if kind == "off_diagonal":
        return torch.roll(identity, shifts=1, dims=1) + 0.05 * random_part
    if kind == "gw":
        z1 = torch.randn((size, 16), generator=generator, dtype=torch.float64)
        z2 = torch.randn((size, 16), generator=generator, dtype=torch.float64)
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        relation_1 = 1.0 - z1 @ z1.T
        relation_2 = 1.0 - z2 @ z2.T
        initial = identity / size
        gradient = -2.0 * (
            relation_1 @ initial @ relation_2.T
            + relation_1.T @ initial @ relation_2
        )
        return initial - 0.05 * gradient
    raise ValueError(f"unknown candidate kind {kind}")


def cvx_projection(candidate: torch.Tensor) -> np.ndarray:
    values = candidate.numpy()
    size = values.shape[0]
    plan = cp.Variable((size, size), nonneg=True)
    marginal = np.full(size, 1.0 / size, dtype=np.float64)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(plan - values)),
        [cp.sum(plan, axis=1) == marginal, cp.sum(plan, axis=0) == marginal],
    )
    problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-9,
        tol_feas=1e-9,
        tol_gap_rel=1e-9,
        max_iter=1000,
    )
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or plan.value is None:
        raise RuntimeError(f"CVXPY failed: {problem.status}")
    return np.asarray(plan.value, dtype=np.float64)


def newton_project(projector, candidate: torch.Tensor, max_iter: int = 50):
    marginal = 1.0 / candidate.shape[0]
    return projector._gpu_newton_bistochastic_projection(
        candidate,
        marginal,
        max_iter=max_iter,
        tol=1e-10,
        relative_tol=None,
        precision="float64",
        fallback_to_float64=False,
        no_accept_damping=True,
    )


@torch.no_grad()
def small_validation(model_module, sizes: list[int]) -> list[dict[str, object]]:
    projector = model_module.OursModel.__new__(model_module.OursModel)
    torch.nn.Module.__init__(projector)
    records = []
    for size in sizes:
        for offset, kind in enumerate(("random", "diagonal", "off_diagonal", "gw")):
            candidate_cpu = make_small_candidate(size, 20261020 + size * 10 + offset, kind)
            reference = cvx_projection(candidate_cpu)
            candidate = candidate_cpu.cuda()
            started = time.perf_counter()
            plan = newton_project(projector, candidate)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            plan_np = plan.detach().cpu().numpy().astype(np.float64)
            size_float = float(size)
            row_rel = float(np.max(np.abs(plan_np.sum(axis=1) - 1.0 / size))) * size_float
            col_rel = float(np.max(np.abs(plan_np.sum(axis=0) - 1.0 / size))) * size_float
            records.append({
                "size": size,
                "kind": kind,
                "status": "success",
                "elapsed_ms": elapsed_ms,
                "max_plan_abs_error_vs_cvx": float(np.max(np.abs(plan_np - reference))),
                "pseudo_label_max_abs_error_vs_cvx": float(np.max(np.abs(size_float * plan_np - size_float * reference))),
                "objective_gap": float(
                    0.5 * np.square(plan_np - candidate_cpu.numpy()).sum()
                    - 0.5 * np.square(reference - candidate_cpu.numpy()).sum()
                ),
                "row_relative_residual": row_rel,
                "col_relative_residual": col_rel,
            })
            print(json.dumps(records[-1], sort_keys=True), flush=True)
    return records


@torch.no_grad()
def large_validation(model_module, sizes: list[int], kinds: list[str], repeats: int) -> list[dict[str, object]]:
    projector = model_module.OursModel.__new__(model_module.OursModel)
    torch.nn.Module.__init__(projector)
    records = []
    for size in sizes:
        for kind_index, kind in enumerate(kinds):
            candidate = make_candidate(size, kind, 20261020 + size * 100 + kind_index).to(dtype=torch.float64)
            marginal = 1.0 / size
            outputs = []
            times = []
            for _ in range(repeats):
                started = time.perf_counter()
                plan = newton_project(projector, candidate)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - started) * 1000.0)
                outputs.append(plan)
            plan = outputs[-1]
            row_rel = float(torch.abs(plan.sum(1) - marginal).max().item()) * size
            col_rel = float(torch.abs(plan.sum(0) - marginal).max().item()) * size
            min_value = float(plan.min().item())
            stats = getattr(projector, "_dense_projection_stats", {})
            records.append({
                "size": size,
                "kind": kind,
                "repeats": repeats,
                "elapsed_ms": times,
                "row_relative_residual": row_rel,
                "col_relative_residual": col_rel,
                "min_plan_value": min_value,
                "strict_feasible": row_rel <= 1e-3 and col_rel <= 1e-3 and min_value >= -1e-7,
                "frank_wolfe_gap": frank_wolfe_gap(plan, candidate),
                "kkt": sparse_kkt(plan, candidate),
                "newton_calls": stats.get("calls"),
                "newton_iterations_total": stats.get("newton_iterations"),
                "newton_success_total": stats.get("newton_success"),
                "projection_rejections_total": stats.get("projection_rejections"),
            })
            print(json.dumps(records[-1], sort_keys=True), flush=True)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--densegw-root", type=Path, default=Path("/home/dingyn/code/DENSEGW_GPU_NEWTON"))
    parser.add_argument("--small-sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--large-sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--large-kinds", nargs="+", default=["random", "gw"])
    parser.add_argument("--large-repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    environment = preflight()
    model_module = load_densegw_model(args.densegw_root)
    small = small_validation(model_module, args.small_sizes)
    large = large_validation(model_module, args.large_sizes, args.large_kinds, args.large_repeats)
    args.output.write_text(
        json.dumps({"environment": environment, "small": small, "large": large}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"small_records": len(small), "large_records": len(large)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
