"""Check the independent KKT certificate against CVXPY solutions."""

from __future__ import annotations

import json

import cvxpy as cp
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from validate_drot_cvxpy import _candidate, _cvx_projection
from validate_drot_kkt import sparse_kkt


def frank_wolfe_gap(plan: np.ndarray, candidate: np.ndarray) -> float:
    gradient = plan - candidate
    row_indices, col_indices = linear_sum_assignment(gradient)
    linear_minimum = float(gradient[row_indices, col_indices].sum() / plan.shape[0])
    return float(np.sum(gradient * plan) - linear_minimum)


def main() -> int:
    records = []
    for size in (2, 3, 4, 8, 16):
        for offset, kind in enumerate(("random", "diagonal", "off_diagonal")):
            candidate = _candidate(size, 20261020 + size * 10 + offset, kind)
            cvx_plan = _cvx_projection(candidate)
            cvx_tensor = torch.from_numpy(cvx_plan)
            kkt = sparse_kkt(cvx_tensor, candidate.double())
            gap = frank_wolfe_gap(cvx_plan, candidate.numpy().astype(np.float64))
            record = {
                "size": size,
                "kind": kind,
                "cvx_status": "optimal_reference",
                "frank_wolfe_gap": gap,
                **kkt,
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if record["active_stationarity_inf"] > 1e-5 or abs(gap) > 1e-5:
                raise RuntimeError(f"certificate failed on known CVXPY solution: {record}")
    print(json.dumps({"records": records}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
