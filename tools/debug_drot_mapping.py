"""Temporary diagnostic for the DROT quadratic coefficient mapping."""

from __future__ import annotations

import json

import cvxpy as cp
import numpy as np
import torch

from solver.drot_adapter import prepare_drot_inputs


def main() -> None:
    import reg_drot

    size = 2
    generator = torch.Generator(device="cpu").manual_seed(20261040)
    candidate = torch.randn((size, size), generator=generator, dtype=torch.float32)
    values = candidate.numpy().astype(np.float64)
    plan = cp.Variable((size, size), nonneg=True)
    marginal_np = np.full(size, 1.0 / size)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(plan - values)),
        [cp.sum(plan, axis=1) == marginal_np, cp.sum(plan, axis=0) == marginal_np],
    )
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    cvx = np.asarray(plan.value)
    candidate_cuda = candidate.cuda()
    marginal = torch.full((size,), 1.0 / size, device="cuda", dtype=torch.float32)
    inputs = prepare_drot_inputs(candidate_cuda, marginal, marginal)
    shifted_cost = inputs.cost + 1e-3
    outputs = []
    for rho in (0.1, 0.5, 1.0, 2.0, 4.0, 10.0):
        result = reg_drot.quadratic_drot(
            shifted_cost,
            marginal,
            marginal,
            rho,
            0.0,
            100000,
            1e-6,
        )
        result_np = result.detach().cpu().numpy().astype(np.float64)
        outputs.append(
            {
                "rho": rho,
                "plan": result_np.tolist(),
                "objective": float(0.5 * np.square(result_np - values).sum()),
                "max_error": float(np.max(np.abs(result_np - cvx))),
            }
        )
    print(json.dumps({"candidate": values.tolist(), "cvx": cvx.tolist(), "base_r_weight": inputs.regularization_weight, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
