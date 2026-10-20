"""Large-batch DROT feasibility and KKT diagnostics.

This is not a performance benchmark.  It records numerical evidence for the
same Euclidean projection adapter at the batch sizes used by STRIDE, without
using CVXPY for million-variable problems.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from solver.drot_adapter import DrotEuclideanProjector
from validation_runtime import preflight


def make_candidate(size: int, kind: str, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    random_part = torch.randn((size, size), device="cuda", dtype=torch.float32, generator=generator)
    identity = torch.eye(size, device="cuda", dtype=torch.float32)
    if kind == "random":
        return random_part
    if kind == "diagonal":
        return identity + 0.05 * random_part
    if kind == "off_diagonal":
        return torch.roll(identity, shifts=1, dims=1) + 0.05 * random_part
    if kind == "gw":
        z1 = torch.randn((size, 16), device="cuda", dtype=torch.float32, generator=generator)
        z2 = torch.randn((size, 16), device="cuda", dtype=torch.float32, generator=generator)
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        d1 = 1.0 - z1 @ z1.T
        d2 = 1.0 - z2 @ z2.T
        t0 = identity / size
        gradient = -2.0 * ((d1 @ t0 @ d2.T) + (d1.T @ t0 @ d2))
        return t0 - 0.05 * gradient
    raise ValueError(f"unknown candidate kind: {kind}")


@torch.no_grad()
def kkt_diagnostics(plan: torch.Tensor, candidate: torch.Tensor, iterations: int) -> dict[str, float | int]:
    """Estimate the KKT stationarity certificate using active-set dual fitting."""
    active = plan > 1e-7
    gradient = plan - candidate
    u = torch.zeros(plan.shape[0], device=plan.device, dtype=plan.dtype)
    v = torch.zeros(plan.shape[1], device=plan.device, dtype=plan.dtype)
    for _ in range(iterations):
        row_count = active.sum(dim=1).clamp_min(1).to(plan.dtype)
        u = -(gradient + v[None, :]).masked_fill(~active, 0.0).sum(dim=1) / row_count
        col_count = active.sum(dim=0).clamp_min(1).to(plan.dtype)
        v = -(gradient + u[:, None]).masked_fill(~active, 0.0).sum(dim=0) / col_count
        shift = u.mean()
        u = u - shift
        v = v + shift
    stationarity = gradient + u[:, None] + v[None, :]
    active_values = torch.abs(stationarity[active])
    inactive_values = torch.relu(-stationarity[~active])
    return {
        "active_entries": int(active.sum().item()),
        "active_stationarity_inf": float(active_values.max().item()) if active_values.numel() else float("inf"),
        "inactive_dual_violation_inf": float(inactive_values.max().item()) if inactive_values.numel() else 0.0,
        "kkt_residual_inf": max(
            float(active_values.max().item()) if active_values.numel() else float("inf"),
            float(inactive_values.max().item()) if inactive_values.numel() else 0.0,
        ),
        "dual_fit_iterations": iterations,
    }


@torch.no_grad()
def run_case(size: int, kind: str, rho: float, solver_eps: float, max_iterations: int, seed: int) -> dict[str, object]:
    candidate = make_candidate(size, kind, seed)
    marginal = torch.full((size,), 1.0 / size, device="cuda", dtype=torch.float32)
    projector = DrotEuclideanProjector(
        rho=rho,
        max_iterations=max_iterations,
        solver_eps=solver_eps,
        relative_tolerance=2e-3,
        absolute_tolerance=1e-8,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        result = projector.project(candidate, marginal, marginal)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        plan = result.plan
        finite = bool(torch.isfinite(plan).all().item())
        min_value = float(plan.min().item())
        row_abs = float(torch.abs(plan.sum(dim=1) - marginal).max().item())
        col_abs = float(torch.abs(plan.sum(dim=0) - marginal).max().item())
        row_relative = row_abs / (1.0 / size)
        col_relative = col_abs / (1.0 / size)
        negative_violation = max(0.0, -min_value)
        objective = float((0.5 * torch.square(plan - candidate).sum()).item())
        kkt = kkt_diagnostics(plan, candidate, iterations=30) if finite else {}
        strict_success = finite and negative_violation <= 1e-7 and row_relative <= 1e-3 and col_relative <= 1e-3
        record: dict[str, object] = {
            "size": size,
            "kind": kind,
            "rho": rho,
            "solver_eps": solver_eps,
            "max_iterations": max_iterations,
            "backend_status": result.diagnostics.status,
            "strict_success": strict_success,
            "elapsed_ms": elapsed_ms,
            "objective": objective,
            "min_value": min_value,
            "negative_violation": negative_violation,
            "row_abs_residual": row_abs,
            "col_abs_residual": col_abs,
            "row_relative_residual": row_relative,
            "col_relative_residual": col_relative,
            "solver_iteration_count": "not exposed by upstream API",
            **kkt,
        }
    except Exception as error:  # noqa: BLE001 - preserve per-case evidence
        torch.cuda.synchronize()
        record = {
            "size": size,
            "kind": kind,
            "rho": rho,
            "solver_eps": solver_eps,
            "max_iterations": max_iterations,
            "strict_success": False,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    del candidate
    torch.cuda.empty_cache()
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--kinds", nargs="+", default=["random", "diagonal", "off_diagonal", "gw"])
    parser.add_argument("--rhos", nargs="+", type=float, default=[2.0, 4.0, 10.0])
    parser.add_argument("--solver-eps", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    preflight()
    records = []
    for size in args.sizes:
        for kind_index, kind in enumerate(args.kinds):
            for rho_index, rho in enumerate(args.rhos):
                records.append(
                    run_case(
                        size,
                        kind,
                        rho,
                        args.solver_eps,
                        args.max_iterations,
                        args.seed + size * 100 + kind_index * 10 + rho_index,
                    )
                )
    payload = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "strict_acceptance": {
            "negative_violation": "<= 1e-7",
            "row_relative_residual": "<= 1e-3",
            "col_relative_residual": "<= 1e-3",
            "kkt_residual": "reported diagnostically, not used as a hidden pass condition",
        },
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    passed = sum(bool(record.get("strict_success")) for record in records)
    print(f"strict_success={passed}/{len(records)}", flush=True)
    return 0 if passed == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
