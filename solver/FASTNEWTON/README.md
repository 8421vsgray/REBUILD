# FASTNEWTON (included solver package)

FASTNEWTON projects a square matrix onto the uniform transport polytope by solving

```text
min_T 0.5 * ||T - X||_F^2
s.t. T >= 0, T 1 = 1/B, T.T 1 = 1/B.
```

The package is included under `solver/FASTNEWTON` and installed from the REBUILD root with:

```bash
python -m pip install -e ./solver/FASTNEWTON
```

Minimal use:

```python
import torch
from fastnewton import project

batch_size = 32
X = 0.1 * torch.eye(batch_size, device="cuda")
T, info = project(X, relative_tol=1e-3, max_iter=200, return_info=True)
if info.get("status") != "verified":
    raise RuntimeError(f"Projection was not verified: {info}")
```

FASTNEWTON reports `verified`, `feasible_only`, or `failed`. Callers that require an optimality certificate should require `status == "verified"`; feasible marginals alone are not an optimality certificate. The public API raises on failure unless diagnostic return is requested.

Install the REBUILD requirements before running tests or training. The optional CVXPY reference tests are declared in `requirements-test.txt`.

Source provenance is recorded in `vendor/BASELINE_PROVENANCE.md`. No DROT code or runtime dependency is required by the current REBUILD solver path.
