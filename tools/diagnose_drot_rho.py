"""Measure DROT sensitivity to rho without changing the third-party kernel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from solver.drot_adapter import prepare_drot_inputs
from validate_drot_large import make_candidate


@torch.no_grad()
def run_case(size: int, kind: str, rho: float, max_iterations: int, seed: int) -> dict[str, object]:
    candidate = make_candidate(size, kind, seed)
    marginal = torch.full((size,), 1.0 / size, device="cuda", dtype=torch.float32)
    inputs = prepare_drot_inputs(candidate, marginal, marginal)
    import reg_drot

    torch.cuda.synchronize()
    started = time.perf_counter()
    plan = reg_drot.quadratic_drot(
        inputs.cost,
        inputs.col_marginal,
        inputs.row_marginal,
        rho,
        inputs.regularization_weight,
        max_iterations,
        1e-6,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    row_relative = float(torch.abs(plan.sum(1) - marginal).max().item()) * size
    col_relative = float(torch.abs(plan.sum(0) - marginal).max().item()) * size
    record = {
        "size": size,
        "kind": kind,
        "rho": rho,
        "max_iterations": max_iterations,
        "elapsed_ms": elapsed_ms,
        "row_relative_residual": row_relative,
        "col_relative_residual": col_relative,
        "min_plan_value": float(plan.min().item()),
        "objective": float((0.5 * torch.square(plan - candidate).sum()).item()),
        "iteration_count": "not exposed by upstream API",
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    del candidate, plan
    torch.cuda.empty_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--kinds", nargs="+", default=["random", "gw"])
    parser.add_argument("--rhos", nargs="+", type=float, default=[2.0, 4.0, 10.0])
    parser.add_argument("--max-iterations", nargs="+", type=int, default=[1000, 5000, 20000])
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for kind_index, kind in enumerate(args.kinds):
        for rho in args.rhos:
            for max_iterations in args.max_iterations:
                records.append(
                    run_case(
                        args.size,
                        kind,
                        rho,
                        max_iterations,
                        args.seed + args.size * 100 + kind_index,
                    )
                )
    args.output.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    print(f"records={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
