# Regularized-DROT CUDA build record

Date: 2026-10-20

## Source provenance

The vendored source is the user-supplied `Regularized-DROT-main.zip` snapshot.
The archive SHA-256 is recorded in
`solver/REGULARIZED_DROT_PROVENANCE.md`; the archive did not contain Git
metadata, so no upstream commit SHA is fabricated.

## Independent build environment

- Environment: `/home/dingyn/.conda/envs/rebuild-cuda128`
- PyTorch: `2.8.0+cu128`
- `torch.version.cuda`: `12.8`
- `nvcc`: `/home/dingyn/.conda/envs/rebuild-cuda128/bin/nvcc`
- CUDA compiler: `12.8.93`
- GPU: NVIDIA GeForce RTX 4090
- System CUDA 13.0: left unchanged

The environment was created independently from the existing training
environment and populated with the CUDA 12.8.90/12.8.93 compiler, runtime,
CRT and NVVM packages. The user-provided PyPI `nvcc` wheel was inspected but
was not treated as a complete compiler because it only supplied `ptxas`.

## Build

The upstream `solver/Regularized-DROT/pytorch/setup.py build_ext --inplace`
command completed successfully with `CUDA_HOME` pointing to the independent
12.8 environment. Both CUDA source files compiled and linked to produce
`reg_drot.cpython-311-x86_64-linux-gnu.so`.

Generated build directories and the extension are ignored by Git; source and
build evidence are kept separate.

## Upstream functional test

The unmodified upstream `pytorch/test.py` was run with the new environment.
All seven quadratic-DROT and seven group-lasso-DROT cases completed, with the
reported residuals at or below `0.000100`, and the process printed `PASS`.

This establishes:

1. CUDA 12.8 can compile the supplied extension against PyTorch cu128.
2. The extension can load and execute on the target RTX 4090.
3. The upstream feasibility examples run successfully.

It does not by itself establish that DROT solves REBUILD's Euclidean
projection objective; that is recorded separately in
`docs/SOLVER_VALIDATION.md` and the CVXPY comparison artifact.
