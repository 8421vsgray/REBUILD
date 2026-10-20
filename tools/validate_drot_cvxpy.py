"""Independent small-problem validation for the DROT Euclidean adapter.

This script is intentionally separate from the training code.  It compares
the adapter output with an independently formulated CVXPY convex problem and
fails loudly when the CUDA extension or a numerical check is unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cvxpy as cp
import numpy as np
import torch

from validation_runtime import preflight


def _candidate(size: int, seed: int, kind: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_part = torch.randn((size, size), generator=generator, dtype=torch.float32)
    if kind == "random":
        return random_part
    identity = torch.eye(size, dtype=torch.float32)
    if kind == "diagonal":
        return identity + 0.05 * random_part
    if kind == "off_diagonal":
        shifted = torch.roll(identity, shifts=1, dims=1)
        return shifted + 0.05 * random_part
    if kind == "gw":
        z1 = torch.randn((size, 16), generator=generator, dtype=torch.float32)
        z2 = torch.randn((size, 16), generator=generator, dtype=torch.float32)
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
    raise ValueError(f"unknown candidate kind: {kind}")


def _cvx_projection(candidate: torch.Tensor) -> np.ndarray:
    values = candidate.detach().cpu().numpy().astype(np.float64)
    size = values.shape[0]
    plan = cp.Variable((size, size), nonneg=True)
    marginal = np.full(size, 1.0 / size, dtype=np.float64)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(plan - values)),
        [cp.sum(plan, axis=1) == marginal, cp.sum(plan, axis=0) == marginal],
    )
    result = problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-9,
        tol_feas=1e-9,
        tol_gap_rel=1e-9,
        max_iter=1000,
    )
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or plan.value is None:
        raise RuntimeError(f"CVXPY failed: status={problem.status}, value={result}")
    return np.asarray(plan.value, dtype=np.float64)


def _exact_projection_2x2(candidate: torch.Tensor) -> np.ndarray:
    values = candidate.detach().cpu().numpy().astype(np.float64)
    diagonal_mass = (
        1.0
        + values[0, 0]
        + values[1, 1]
        - values[0, 1]
        - values[1, 0]
    ) / 4.0
    diagonal_mass = float(np.clip(diagonal_mass, 0.0, 0.5))
    return np.array(
        [[diagonal_mass, 0.5 - diagonal_mass],
         [0.5 - diagonal_mass, diagonal_mass]],
        dtype=np.float64,
    )


def _record(
    candidate: torch.Tensor,
    cvx_plan: np.ndarray,
    drot_plan: torch.Tensor,
    status: str,
) -> dict[str, object]:
    candidate_np = candidate.cpu().numpy().astype(np.float64)
    drot_np = drot_plan.detach().cpu().numpy().astype(np.float64)
    size = candidate.shape[0]
    marginal = 1.0 / size

    def objective(plan: np.ndarray) -> float:
        return float(0.5 * np.square(plan - candidate_np).sum())

    row_abs = float(np.max(np.abs(drot_np.sum(axis=1) - marginal)))
    col_abs = float(np.max(np.abs(drot_np.sum(axis=0) - marginal)))
    cvx_objective = objective(cvx_plan)
    drot_objective = objective(drot_np)
    return {
        "size": size,
        "status": status,
        "min_value": float(drot_np.min()),
        "max_plan_abs_error_vs_cvx": float(np.max(np.abs(drot_np - cvx_plan))),
        "plan_frobenius_error_vs_cvx": float(np.linalg.norm(drot_np - cvx_plan)),
        "drot_objective": drot_objective,
        "cvx_objective": cvx_objective,
        "objective_gap": drot_objective - cvx_objective,
        "row_abs_residual": row_abs,
        "col_abs_residual": col_abs,
        "row_relative_residual": row_abs / marginal,
        "col_relative_residual": col_abs / marginal,
        "pseudo_label_max_abs_error_vs_cvx": float(
            np.max(np.abs(size * drot_np - size * cvx_plan))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--solver-eps", type=float, default=1e-6)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DROT validation")
    preflight()

    from solver.drot_adapter import DrotEuclideanProjector

    projector = DrotEuclideanProjector(
        rho=args.rho,
        max_iterations=args.max_iterations,
        solver_eps=args.solver_eps,
        relative_tolerance=args.relative_tolerance,
    )
    if not projector.available:
        raise RuntimeError(f"DROT extension unavailable: {projector.backend_error}")

    records: list[dict[str, object]] = []
    for size in args.sizes:
        for offset, kind in enumerate(("random", "diagonal", "off_diagonal", "gw")):
            candidate = _candidate(size, 20261020 + size * 10 + offset, kind)
            cvx_plan = _cvx_projection(candidate)
            candidate_cuda = candidate.cuda()
            marginal = torch.full((size,), 1.0 / size, device="cuda", dtype=torch.float32)
            result = projector.project(candidate_cuda, marginal, marginal)
            record = _record(candidate, cvx_plan, result.plan, result.diagnostics.status)
            record["kind"] = kind
            record["diagnostics"] = result.diagnostics.__dict__
            if size == 2:
                analytic_plan = _exact_projection_2x2(candidate)
                record["analytic_max_error_vs_cvx"] = float(
                    np.max(np.abs(analytic_plan - cvx_plan))
                )
                record["analytic_max_error_vs_drot"] = float(
                    np.max(np.abs(analytic_plan - result.plan.detach().cpu().numpy()))
                )
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if result.diagnostics.status != "success":
                raise RuntimeError(f"DROT feasibility check failed: {record}")
            if record["objective_gap"] > 5e-4:
                raise RuntimeError(f"DROT objective gap too large: {record}")
            if record["pseudo_label_max_abs_error_vs_cvx"] > 1e-3:
                raise RuntimeError(f"DROT pseudo-label error too large: {record}")

    output = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "records": records,
    }
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
