"""Small-sample timing diagnosis before any large benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from solver.drot_adapter import DrotEuclideanProjector, prepare_drot_inputs
from validate_drot_large import make_candidate
from validation_runtime import preflight


@torch.no_grad()
def run_case(size: int, kind: str, repeats: int, seed: int) -> dict[str, object]:
    candidate = make_candidate(size, kind, seed)
    marginal = torch.full((size,), 1.0 / size, device="cuda", dtype=torch.float32)
    projector = DrotEuclideanProjector(
        rho=4.0,
        max_iterations=20_000,
        solver_eps=1e-6,
        relative_tolerance=1e-3,
        absolute_tolerance=1e-8,
    )
    inputs = prepare_drot_inputs(candidate, marginal, marginal)
    backend = projector._backend
    if backend is None:
        raise RuntimeError("DROT backend unavailable after preflight")

    # One warmup per path, excluded from the reported samples.
    projector.project(candidate, marginal, marginal)
    backend.quadratic_drot(
        inputs.cost,
        inputs.col_marginal,
        inputs.row_marginal,
        4.0,
        inputs.regularization_weight,
        20_000,
        1e-6,
    )
    torch.cuda.synchronize()

    adapter_ms = []
    native_ms = []
    adapter_status = []
    row_relative = []
    col_relative = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = projector.project(candidate, marginal, marginal)
        torch.cuda.synchronize()
        adapter_ms.append((time.perf_counter() - started) * 1000.0)
        adapter_status.append(result.diagnostics.status)
        plan = result.plan
        row_relative.append(float(torch.abs(plan.sum(1) - marginal).max().item()) * size)
        col_relative.append(float(torch.abs(plan.sum(0) - marginal).max().item()) * size)

        started = time.perf_counter()
        backend.quadratic_drot(
            inputs.cost,
            inputs.col_marginal,
            inputs.row_marginal,
            4.0,
            inputs.regularization_weight,
            20_000,
            1e-6,
        )
        torch.cuda.synchronize()
        native_ms.append((time.perf_counter() - started) * 1000.0)

    record = {
        "size": size,
        "kind": kind,
        "repeats": repeats,
        "rho": 4.0,
        "solver_eps": 1e-6,
        "max_iterations": 20_000,
        "adapter_ms": adapter_ms,
        "native_ms": native_ms,
        "adapter_status": adapter_status,
        "max_row_relative_residual": max(row_relative),
        "max_col_relative_residual": max(col_relative),
        "iteration_count": "not exposed by upstream API",
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    del candidate
    torch.cuda.empty_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--kinds", nargs="+", default=["random", "gw"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    environment = preflight()
    records = []
    for size in args.sizes:
        for kind_index, kind in enumerate(args.kinds):
            records.append(run_case(size, kind, args.repeats, args.seed + size * 100 + kind_index))
    payload = {
        "environment": environment,
        "gpu_snapshot": subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"records={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
