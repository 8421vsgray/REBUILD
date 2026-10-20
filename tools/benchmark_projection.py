"""Benchmark the complete DROT Euclidean projection adapter.

The benchmark is intentionally separate from training.  It records both the
native CUDA call and the end-to-end adapter call, while checking feasibility of
every returned plan.  A fast but infeasible result is not counted as valid.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from solver.drot_adapter import DrotEuclideanProjector, prepare_drot_inputs
from solver.projection_interface import validate_transport_plan
from validate_drot_large import make_candidate
from validation_runtime import preflight


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values),
        "min_ms": min(values),
    }


def _strict_metrics(plan: torch.Tensor, marginal: torch.Tensor) -> dict[str, float | bool]:
    size = plan.shape[0]
    finite = bool(torch.isfinite(plan).all().item())
    minimum = float(plan.min().item())
    negative_violation = max(0.0, -minimum)
    row_relative = float(torch.abs(plan.sum(1) - marginal).max().item()) * size
    col_relative = float(torch.abs(plan.sum(0) - marginal).max().item()) * size
    strict = (
        finite
        and negative_violation <= 1e-7
        and row_relative <= 1e-3
        and col_relative <= 1e-3
    )
    return {
        "finite": finite,
        "min_plan_value": minimum,
        "negative_violation": negative_violation,
        "row_relative_residual": row_relative,
        "col_relative_residual": col_relative,
        "strict_feasible": strict,
    }


@torch.no_grad()
def run_case(
    size: int,
    kind: str,
    rho: float,
    solver_eps: float,
    repeats: int,
    warmup: int,
    seed: int,
) -> dict[str, object]:
    candidate = make_candidate(size, kind, seed)
    marginal = torch.full((size,), 1.0 / size, device="cuda", dtype=torch.float32)
    projector = DrotEuclideanProjector(
        rho=rho,
        max_iterations=20_000,
        solver_eps=solver_eps,
        relative_tolerance=1e-3,
        absolute_tolerance=1e-8,
    )
    inputs = prepare_drot_inputs(candidate, marginal, marginal)
    backend = projector._backend
    if backend is None:
        raise RuntimeError("DROT backend is unavailable")

    for _ in range(warmup):
        projector.project(candidate, marginal, marginal)
        backend.quadratic_drot(
            inputs.cost,
            inputs.col_marginal,
            inputs.row_marginal,
            rho,
            inputs.regularization_weight,
            20_000,
            solver_eps,
        )
    torch.cuda.synchronize()

    adapter_times: list[float] = []
    native_times: list[float] = []
    adapter_metrics: list[dict[str, float | bool]] = []
    native_metrics: list[dict[str, float | bool]] = []
    for _ in range(repeats):
        started = time.perf_counter()
        adapter_result = projector.project(candidate, marginal, marginal)
        torch.cuda.synchronize()
        adapter_times.append((time.perf_counter() - started) * 1000.0)
        adapter_metrics.append(_strict_metrics(adapter_result.plan, marginal))

        started = time.perf_counter()
        native_plan = backend.quadratic_drot(
            inputs.cost,
            inputs.col_marginal,
            inputs.row_marginal,
            rho,
            inputs.regularization_weight,
            20_000,
            solver_eps,
        )
        torch.cuda.synchronize()
        native_times.append((time.perf_counter() - started) * 1000.0)
        native_metrics.append(_strict_metrics(native_plan, marginal))

    def aggregate(metrics: list[dict[str, float | bool]]) -> dict[str, object]:
        return {
            "strict_feasible_count": sum(bool(item["strict_feasible"]) for item in metrics),
            "strict_feasible_rate": statistics.fmean(bool(item["strict_feasible"]) for item in metrics),
            "max_row_relative_residual": max(float(item["row_relative_residual"]) for item in metrics),
            "max_col_relative_residual": max(float(item["col_relative_residual"]) for item in metrics),
            "max_negative_violation": max(float(item["negative_violation"]) for item in metrics),
        }

    record = {
        "size": size,
        "kind": kind,
        "rho": rho,
        "solver_eps": solver_eps,
        "repeats": repeats,
        "warmup": warmup,
        "adapter": {"timing": _summary(adapter_times), **aggregate(adapter_metrics)},
        "native_drot": {"timing": _summary(native_times), **aggregate(native_metrics)},
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    del candidate
    torch.cuda.empty_cache()
    return record


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# DROT Euclidean Projection Benchmark",
        "",
        f"- Device: `{payload['device']}`",
        f"- PyTorch: `{payload['torch_version']}`; CUDA: `{payload['torch_cuda']}`",
        f"- Repeats: `{payload['repeats']}`; warmup: `{payload['warmup']}`",
        "- Strict feasibility: negative violation <= 1e-7 and row/column relative residual <= 1e-3.",
        "- The native DROT timing excludes adapter validation; adapter timing includes the complete projection call and validation.",
        "",
        "| B | Input | Adapter mean / median / P95 (ms) | Native mean / median / P95 (ms) | Adapter valid | Native valid |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for record in payload["records"]:
        adapter = record["adapter"]
        native = record["native_drot"]
        lines.append(
            f"| {record['size']} | {record['kind']} | "
            f"{adapter['timing']['mean_ms']:.3f} / {adapter['timing']['median_ms']:.3f} / {adapter['timing']['p95_ms']:.3f} | "
            f"{native['timing']['mean_ms']:.3f} / {native['timing']['median_ms']:.3f} / {native['timing']['p95_ms']:.3f} | "
            f"{adapter['strict_feasible_count']}/{record['repeats']} | {native['strict_feasible_count']}/{record['repeats']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--kinds", nargs="+", default=["random", "diagonal", "off_diagonal", "gw"])
    parser.add_argument("--rho", type=float, default=4.0)
    parser.add_argument("--solver-eps", type=float, default=1e-6)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    preflight()
    records = []
    for size in args.sizes:
        for kind_index, kind in enumerate(args.kinds):
            records.append(
                run_case(
                    size,
                    kind,
                    args.rho,
                    args.solver_eps,
                    args.repeats,
                    args.warmup,
                    args.seed + size * 100 + kind_index,
                )
            )
    payload = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(args.output.with_suffix(".md"), payload)
    print(f"records={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
