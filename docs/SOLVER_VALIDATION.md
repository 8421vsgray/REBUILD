# Solver validation record

## Current status

The REBUILD transport interface, CPU reference Euclidean projector and
Regularized-DROT adapter are implemented. The DROT source snapshot is vendored
under `solver/Regularized-DROT/` and remains unmodified.

The CUDA extension has been built and the upstream DROT test has passed in the
independent `rebuild-cuda128` environment. This is separate from the stronger
REBUILD Euclidean-projection validation below: the upstream test checks
feasibility residuals, not whether the returned plan minimizes the Euclidean
projection objective.

## Environment and build evidence

- GPU: NVIDIA GeForce RTX 4090.
- PyTorch: `2.8.0+cu128`.
- PyTorch CUDA runtime: `12.8`.
- Compiler: CUDA Toolkit `12.8.93`, from
  `/home/dingyn/.conda/envs/rebuild-cuda128/bin/nvcc`.
- The system CUDA 13.0 installation was not modified.
- Extension: `reg_drot.cpython-311-x86_64-linux-gnu.so`.
- Build command: `python setup.py build_ext --inplace` in the vendored
  `pytorch` directory, with `CUDA_HOME` set to the independent environment.

The official `pytorch/test.py` completed all seven quadratic and seven
group-lasso cases and printed `PASS`; the reported upstream residuals were at
or below `0.000100`. Generated build files and the extension are ignored by
Git and are not treated as source evidence.

## REBUILD Euclidean projection mapping

For a candidate matrix `X`, REBUILD uses

```text
s = max(X) - min(X)
C = (max(X) - X) / s
r_weight = 1 / ((m + n) * s)
```

with uniform marginals for the square STRIDE batch. Constant candidates use
the analytic uniform coupling and do not call DROT. The adapter does not
silently fall back to Newton, Sinkhorn or another backend.

The validated adapter defaults are `rho=4.0` and `solver_eps=1e-6`. The
upstream example's `rho=2.0`, `eps=1e-4` settings are suitable for its
feasibility demonstration but produced a visibly suboptimal active-set
solution in the first 2x2 Euclidean check, so they are not the REBUILD default.
Both values remain configurable for controlled audits.

## Independent numerical evidence

The standalone script `tools/validate_drot_cvxpy.py` formulates the same
Euclidean projection in CVXPY/CLARABEL and compares the DROT plan, objective,
feasibility residuals and matrix distance. For `B=2`, it also compares the
closed-form projection

```text
a = clip((1 + X00 + X11 - X01 - X10) / 4, 0, 1/2)
T = [[a, 1/2-a], [1/2-a, a]]
```

The first diagnostic run with `rho=2`, `solver_eps=1e-4` had:

| quantity | value |
| --- | ---: |
| batch size | 2 |
| CVXPY objective | 2.4699158489 |
| DROT objective | 2.4872966439 |
| absolute objective gap | 0.0173807950 |
| relative objective gap | 0.704% |
| plan Frobenius distance | 0.1407356443 |

That run was rejected. With `rho=4`, `solver_eps=1e-6` and the same input,
the DROT plan agreed with the analytic/CVXPY boundary solution within
`1.8e-7` Frobenius error. The expanded CVXPY comparison over random,
diagonal-dominant and off-diagonal candidates at `B=2,3,4,8` completed with
all feasibility checks passing; the largest recorded plan Frobenius error was
`2.2884e-4` (the `B=3` random case), and the largest absolute objective gap
was below `1.5e-6`. The raw JSON evidence is kept under `.artifacts/` outside
Git.

These results validate the mapping and chosen numerical settings for the
tested small cases. They do not by themselves certify 512x512 or 1024x1024
plans.

## Large-batch feasibility

`tools/validate_drot_large.py` tested random, diagonal, off-diagonal and
GW-like candidates at `B=128,256,512,1024`, with three `rho` values. All 48
calls returned finite plans satisfying the current feasibility gate:

- negative violation <= `1e-7`;
- row and column relative residual <= `1e-3`.

This is a feasibility result only. The upstream extension does not expose its
internal iteration count. Its C++ wrapper allocates a local `n_iters` value but
discards it before returning the plan; REBUILD therefore records iteration
count as unavailable rather than inventing one. `max_iterations=20000` is
recorded as the configured cap.

## Independent optimality diagnostics

The first large KKT diagnostic had an implementation error in its sparse
design matrix and is discarded. The corrected certificate was checked against
CVXPY solutions at `B=2,3,4,8,16`: active-set stationarity was approximately
`1e-6` or lower and the Frank--Wolfe gaps were approximately `1e-8` or lower.
This validates the diagnostic implementation on independent small solutions.

For large outputs, the corrected sweep covered
`B=128,1024` x `random,GW` x `rho=2,4,10` x
`eps=1e-6,1e-7,1e-8`. The important `B=1024` observations were:

| input | rho | row/column relative residual | active stationarity | FW gap | strict FW upper bound |
| --- | ---: | ---: | ---: | ---: | --- |
| random | 2 | `6.45e-4 / 6.81e-4` | `7.18e-5` | `2.68e-6` | no |
| random | 4 | `3.33e-4 / 3.34e-4` | `2.50e-6` | `-1.39e-6` | no |
| random | 10 | `8.18e-5 / 6.62e-5` | `9.40e-8` | `-1.37e-6` | no |
| GW | 2 | `9.38e-5 / 1.14e-4` | `7.88e-10` | `3.07e-8` | no |
| GW | 4 | `6.13e-5 / 6.21e-5` | `9.21e-10` | `-7.88e-6` | no |
| GW | 10 | `1.82e-4 / 1.83e-4` | `1.06e-9` | `-2.48e-5` | no |

The values were materially unchanged between `eps=1e-6`, `1e-7` and `1e-8`
for these FP32 cases. This means that tightening the upstream primal
residual tolerance alone did not establish Euclidean optimality. The negative
or non-positive FW values are not accepted as upper bounds because the plans
are not feasible to the stricter `1e-6` relative criterion. The fixed-gauge
inactive dual residual is also not used as a pass/fail certificate when the
active graph is disconnected; the CVXPY self-check demonstrated why that
quantity can be gauge-nonunique.

The full objective-value sweep is run separately without the expensive sparse
KKT fit and will be recorded alongside this document. Until its output and
the performance benchmark are reviewed, DROT is classified as **large-scale
feasible, but not yet fully certified as an exact STRIDE Euclidean projector**.

## Validation runtime contract

All CUDA validation jobs now use `tools/run_rebuild_validation.sh`. It fixes
the REBUILD root, the CUDA 12.8 environment and the project/extension import
paths, then runs `tools/validation_runtime.py` before the requested job. The
preflight requires CUDA, imports the compiled `reg_drot` extension, and runs a
minimal 2x2 GPU call. A failed preflight returns non-zero before any batch
records are written. The earlier 36-record `not_verified` objective output is
kept only as an invalid-run artifact and is not used as evidence.

## Remaining validation

Before using DROT as a formal training backend, REBUILD still needs:

1. objective values and convergence comparison for the large-batch cases;
2. the independent projection performance benchmark;
3. real STRIDE GW candidate inputs with the same acceptance checks;
4. explicit failure-rate and checkpoint records for those sizes.

No formal training result may treat the feasibility gate as an optimality
proof, and no silent Newton/Sinkhorn fallback is allowed.
