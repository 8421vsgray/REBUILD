"""Record Euclidean projection objectives without the expensive KKT fit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from solver.drot_adapter import DrotEuclideanProjector
from validate_drot_kkt import frank_wolfe_gap
from validate_drot_large import make_candidate
from validation_runtime import preflight


@torch.no_grad()
def run_case(size: int, kind: str, rho: float, eps: float, max_iterations: int, seed: int) -> dict[str, object]:
    candidate = make_candidate(size, kind, seed)
    marginal = torch.full((size,), 1.0 / size, device="cuda", dtype=torch.float32)
    projector = DrotEuclideanProjector(
        rho=rho,
        max_iterations=max_iterations,
        solver_eps=eps,
        relative_tolerance=2e-3,
        absolute_tolerance=1e-8,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = projector.project(candidate, marginal, marginal)
    torch.cuda.synchronize()
    plan = result.plan
    size_float = float(size)
    min_value = float(plan.min().item())
    row_relative = float(torch.abs(plan.sum(1) - marginal).max().item()) * size_float
    col_relative = float(torch.abs(plan.sum(0) - marginal).max().item()) * size_float
    record = {
        "size": size,
        "kind": kind,
        "rho": rho,
        "solver_eps": eps,
        "max_iterations": max_iterations,
        "backend_status": result.diagnostics.status,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "euclidean_projection_objective": float((0.5 * torch.square(plan - candidate).sum()).item()),
        "min_plan_value": min_value,
        "negative_violation": max(0.0, -min_value),
        "row_relative_residual": row_relative,
        "col_relative_residual": col_relative,
        "frank_wolfe_gap": frank_wolfe_gap(plan, candidate),
        "frank_wolfe_is_upper_bound": (
            row_relative <= 1e-6 and col_relative <= 1e-6 and min_value >= -1e-7
        ),
        "strict_feasible": (
            row_relative <= 1e-3 and col_relative <= 1e-3 and min_value >= -1e-7
        ),
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    del candidate, plan
    torch.cuda.empty_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 1024])
    parser.add_argument("--kinds", nargs="+", default=["random", "gw"])
    parser.add_argument("--rhos", nargs="+", type=float, default=[2.0, 4.0, 10.0])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1e-6, 1e-7, 1e-8])
    parser.add_argument("--max-iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    preflight()
    records = []
    for solver_eps in args.epsilons:
        for size in args.sizes:
            for kind_index, kind in enumerate(args.kinds):
                for rho_index, rho in enumerate(args.rhos):
                    records.append(
                        run_case(
                            size,
                            kind,
                            rho,
                            solver_eps,
                            args.max_iterations,
                            args.seed + size * 100 + kind_index * 10 + rho_index,
                        )
                    )
    payload = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"records={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
