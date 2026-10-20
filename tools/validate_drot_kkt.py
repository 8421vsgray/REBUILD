"""Independent sparse KKT certificate for large DROT plans."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from scipy.optimize import linear_sum_assignment
import torch

from solver.drot_adapter import DrotEuclideanProjector
from validate_drot_large import make_candidate
from validation_runtime import preflight


def _active_graph_components(size: int, rows: np.ndarray, cols: np.ndarray) -> int:
    parent = np.arange(2 * size, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for row, col in zip(rows.tolist(), cols.tolist()):
        left = find(int(row))
        right = find(size + int(col))
        if left != right:
            parent[right] = left
    return len({find(index) for index in range(2 * size)})


def sparse_kkt(plan: torch.Tensor, candidate: torch.Tensor, threshold: float = 1e-7) -> dict[str, object]:
    plan_np = plan.detach().cpu().numpy().astype(np.float64)
    candidate_np = candidate.detach().cpu().numpy().astype(np.float64)
    size = plan_np.shape[0]
    active = plan_np > threshold
    rows, cols = np.nonzero(active)
    # Fix v[0] = 0 to remove the usual additive gauge ambiguity.
    edge_indices = np.arange(rows.shape[0])
    variable_cols = size + cols - 1
    has_nonzero_v = cols != 0
    design_rows = np.concatenate((edge_indices, edge_indices[has_nonzero_v]))
    design_cols = np.concatenate((rows, variable_cols[has_nonzero_v]))
    design = sparse.coo_matrix(
        (
            np.ones(design_rows.shape[0], dtype=np.float64),
            (design_rows, design_cols),
        ),
        shape=(rows.shape[0], 2 * size - 1),
    ).tocsr()
    rhs = -(plan_np - candidate_np)[rows, cols]
    fit = sparse_linalg.lsqr(design, rhs, atol=1e-10, btol=1e-10, iter_lim=10000)
    dual = fit[0]
    u = dual[:size]
    v = np.concatenate(([0.0], dual[size:]))
    stationarity = plan_np - candidate_np + u[:, None] + v[None, :]
    active_residual = np.abs(stationarity[active])
    inactive_violation = np.maximum(0.0, -stationarity[~active])
    return {
        "active_threshold": threshold,
        "active_entries": int(active.sum()),
        "active_graph_components": _active_graph_components(size, rows, cols),
        "active_stationarity_inf": float(active_residual.max()) if active_residual.size else float("inf"),
        "inactive_dual_violation_inf": float(inactive_violation.max()) if inactive_violation.size else 0.0,
        "kkt_residual_inf": max(
            float(active_residual.max()) if active_residual.size else float("inf"),
            float(inactive_violation.max()) if inactive_violation.size else 0.0,
        ),
        "lsqr_istop": int(fit[1]),
        "lsqr_iterations": int(fit[2]),
        "lsqr_residual_norm": float(fit[3]),
    }


def frank_wolfe_gap(plan: torch.Tensor, candidate: torch.Tensor) -> float:
    plan_np = plan.detach().cpu().numpy().astype(np.float64)
    candidate_np = candidate.detach().cpu().numpy().astype(np.float64)
    gradient = plan_np - candidate_np
    row_indices, col_indices = linear_sum_assignment(gradient)
    linear_minimum = float(gradient[row_indices, col_indices].sum() / plan_np.shape[0])
    return float(np.sum(gradient * plan_np) - linear_minimum)


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
    row_abs = float(torch.abs(plan.sum(1) - marginal).max().item())
    col_abs = float(torch.abs(plan.sum(0) - marginal).max().item())
    row_relative = row_abs * size
    col_relative = col_abs * size
    min_value = float(plan.min().item())
    negative_violation = max(0.0, -min_value)
    objective = float((0.5 * torch.square(plan - candidate).sum()).item())
    record: dict[str, object] = {
        "size": size,
        "kind": kind,
        "rho": rho,
        "solver_eps": eps,
        "backend_status": result.diagnostics.status,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "row_relative_residual": row_relative,
        "col_relative_residual": col_relative,
        "min_plan_value": min_value,
        "negative_violation": negative_violation,
        "euclidean_projection_objective": objective,
        "strict_feasible": row_relative <= 1e-3 and col_relative <= 1e-3 and min_value >= -1e-7,
        "frank_wolfe_gap": frank_wolfe_gap(plan, candidate),
        "frank_wolfe_is_upper_bound": row_relative <= 1e-6 and col_relative <= 1e-6 and min_value >= -1e-7,
    }
    record.update(sparse_kkt(plan, candidate))
    print(json.dumps(record, sort_keys=True), flush=True)
    del candidate, plan
    torch.cuda.empty_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--kinds", nargs="+", default=["random", "gw"])
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
    args.output.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
