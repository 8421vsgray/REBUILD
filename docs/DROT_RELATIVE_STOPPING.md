# REBUILD DROT Relative Stopping

REBUILD uses uniform transport marginals and exposes a separate
`quadratic_drot_relative` CUDA entry point. Its `eps` is interpreted as the
relative marginal tolerance

`max(B_row * ||T 1 - p||_2, B_col * ||T^T 1 - q||_2)`.

The L2 residual is a conservative upper bound on the maximum marginal error
after scaling by the inverse uniform marginal. The Python adapter still
recomputes the returned plan's row and column residuals and rejects any plan
whose maximum relative error exceeds `1e-3`.

The upstream `quadratic_drot` entry point retains its original absolute
residual semantics. The relative entry point is intended only for REBUILD's
uniform-marginal projection path.

The current Scene15 configuration uses `solver.eps: 1.0e-3`. Large-scale
optimality remains separately marked unverified; this change only defines and
checks feasibility stopping.
