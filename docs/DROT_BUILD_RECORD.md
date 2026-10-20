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

## New RTX 3090 validation environment

Date: 2026-10-20

- Host: newly provisioned RTX 3090 server.
- Environment: `/root/.venvs/rebuild-cu124`.
- PyTorch: `2.6.0+cu124`; `torch.version.cuda=12.4`.
- Compiler: system `/usr/local/cuda/bin/nvcc`, CUDA 12.0.
- Extension: `reg_drot.cpython-312-x86_64-linux-gnu.so`.
- Build target: `TORCH_CUDA_ARCH_LIST=8.6`.

The extension was rebuilt without changing the third-party CUDA sources. The
unmodified upstream `pytorch/test.py` completed all seven quadratic and all
seven group-lasso cases and printed `PASS`.

The host exposed a driver/userspace mismatch after the toolkit installation:
the kernel driver was 550.76 while the container restored 580 userspace
symlinks on a new session. The successful build/test/timing commands selected
the matching 550.76 libraries within the same validation session. This caveat
must remain attached to any result from this environment.

The empty-3090 timing diagnostic recorded full-adapter median times of
approximately 168/286/429/2682 ms for random B=128/256/512/1024 and
29/24/24/2670 ms for the corresponding GW-like inputs. B=1024 is therefore
retained as a solver stress test only; formal REBUILD training/evaluation is
capped at B=512.

It does not by itself establish that DROT solves REBUILD's Euclidean
projection objective; that is recorded separately in
`docs/SOLVER_VALIDATION.md` and the CVXPY comparison artifact.
