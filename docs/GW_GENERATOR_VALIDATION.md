# Stage 3A — GW+DROT pseudo-label generator validation

Baseline before this stage: `7e96e36f26219c09f698b3b7206d74b69ed41b9c`.

This stage implements the standalone pseudo-label generator only. It does not
modify `model.forward`, `engine_train.py`, the optimizer, EMA updates, or the
training entrypoint.

## Implemented mathematics

For each view, the generator forms the symmetric cosine-distance relation

```text
D = symmetrize(1 - normalize(F) normalize(F)^T)
```

and starts with `T0 = I / B`. Each finite GW step computes

```text
G(T) = -2 * (D1 T D2^T + D1^T T D2)
X    = T - step_size * G(T)
T'   = DROT-EuclideanProject(X)
P    = B * T'
```

The output `P` is detached, nonnegative, and row-stochastic. The projector
diagnostics are retained; projection feasibility is rechecked by REBUILD and
failure raises `GWProjectionFailure` rather than using a fallback plan.

The two directions are independent:

```text
P12 = GW(p[0], z_t[1])
P21 = GW(p[1], z_t[0])
```

The implementation never derives `P21` from `P12.T`.

## DROT configuration mapping

`projector_from_solver_config` requires all four fields and passes them without
hard-coded replacements:

| YAML field | DROT adapter field |
|---|---|
| `solver.backend` | backend selector (`drot` only) |
| `solver.rho` | `rho` |
| `solver.eps` | `solver_eps` |
| `solver.max_iter` | `max_iterations` |

`steps` and `step_size` remain GW parameters and are not mixed with `rho`.
The adapter loads the pinned extension from
`solver/Regularized-DROT/pytorch/` when it is not installed in the active
environment. It does not modify the third-party CUDA source or system driver
links.

## Tests executed on the RTX 3090

Environment:

```text
Python: /root/.venvs/rebuild-cu124/bin/python
PyTorch: 2.6.0+cu124
CUDA runtime: 12.4
GPU: RTX 3090
CVXPY: 1.9.3
```

Command:

```bash
cd /root/code/REBUILD
PYTHONPATH=/root/code/REBUILD/solver/Regularized-DROT/pytorch:$PYTHONPATH \
  /root/.venvs/rebuild-cu124/bin/python \
  -m unittest -v tests.test_gw_generator
```

Result: **11/11 PASS**.

The tests cover relation/GW gradient agreement with autograd, diagonal
initialization, independent direction pairing, explicit solver parameters,
failure propagation, batch sizes 7/32/512, invalid inputs, CVXPY reference
availability at B=2/4/8/16/32, and DROT-vs-CVXPY projection error at those
small sizes. Across the 20 DROT/CVXPY cases, the largest observed
transport-plan element error was `1.74e-5`; the largest row-stochastic
pseudo-label element error was `5.56e-4`, below the `1e-3` small-scale
acceptance threshold.

The full repository regression suite also passed:

```text
45 tests, OK
```

## Real Scene15 B=512 diagnostic

Command:

```bash
cd /root/code/REBUILD
PYTHONPATH=. /root/.venvs/rebuild-cu124/bin/python \
  -m tools.validate_gw_scene15
```

Configuration: Scene15, views `[1, 0]`, `MR=0.8`, `steps=3`,
`step_size=0.05`, `rho=4`, `eps=1e-6`, `max_iter=20000`.

Observed result:

| quantity | P12 | P21 |
|---|---:|---:|
| batch size | 512 | 512 |
| total three-step projection time | 3.447 s | 3.364 s |
| non-diagonal mass after step 3 | 10.853% | 16.834% |
| maximum row/column absolute residual | 5.98e-8 | 5.96e-8 |
| maximum relative row/column residual | 3.06e-5 | 3.06e-5 |

Both directions returned `status=success`, and the complete bidirectional
call took **6.929 s**. The generated matrices were `[512, 512]`, detached,
finite, nonnegative, and row-stochastic within the reported FP32 residual.

## Acceptance status

- GW relation and finite-step gradient: **PASS**.
- DROT small-scale projection comparison: **PASS** for B=2/4/8/16/32.
- Independent bidirectional target semantics: **PASS**.
- Real Scene15 B=512 execution and feasibility: **PASS**.
- Large-scale DROT global optimality certificate at B=512: **UNVERIFIED**.
- Training-loop integration: **NOT STARTED** by design.

The real-batch diagnostic therefore validates execution and feasibility, not a
formal large-scale optimality proof. Stage 3B must decide how to consume these
targets in training without changing the already-frozen external-target loss
interface.
