# Solver validation record

## Current status

The REBUILD transport interface and a CPU reference Euclidean projector are
implemented and independently tested. The Regularized-DROT source snapshot is
vendored under `solver/Regularized-DROT/`.

The DROT CUDA extension is not yet marked verified. The supplied ZIP does not
contain a Git directory, so its upstream commit cannot be recovered from the
archive. The archive identity is recorded in
`solver/REGULARIZED_DROT_PROVENANCE.md`.

## Evidence completed

- The reference projection tests cover constant, diagonal, random and
  infeasible-plan cases.
- The full REBUILD unit-test suite passed 18/18 after adding the reference
  projector.
- The reference backend checks non-negativity, row/column marginal residuals,
  finite values and explicit status codes.

## DROT verification still pending

An official build attempt was made with:

```text
cd solver/Regularized-DROT/pytorch
python setup.py build_ext --inplace
```

It failed before compilation because the environment exposes CUDA 13.0 while
the installed PyTorch is `2.8.0+cu128` (CUDA 12.8). No extension was produced.
The environment also has no `reg_drot` module installed. This is recorded as
an environment compatibility failure, not as a source-validation success.

The following must be run after a matching CUDA toolchain is made available
and before DROT can be used as a formal training backend:

1. Build `solver/Regularized-DROT/pytorch` against the target PyTorch/CUDA
   environment.
2. Run the upstream `pytorch/test.py` without modifying the third-party
   implementation.
3. Add an REBUILD adapter that maps the Euclidean projection candidate to the
   DROT cost and quadratic regularization parameters.
4. Recompute feasibility diagnostics in REBUILD and compare small problems
   against an independent convex reference.
5. Record the actual compiler, CUDA, PyTorch, GPU and extension details.

Until these checks pass, a DROT output must not be reported as a verified
Euclidean projection and no silent Newton/Sinkhorn fallback is allowed.
