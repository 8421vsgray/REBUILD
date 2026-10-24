# REBUILD

**REBUILD — Robust Multi-View Clustering under Noisy Correspondences** is a research codebase for two-view clustering when the sample order differs between views.

This repository contains a CorreGen-derived training pipeline with the STRIDE-style GW supervision path and the standalone FASTNEWTON Euclidean transport projector. The project reports the implementation that is present here; it does not claim to be an official release of a published STRIDE method.

## Method overview

The implemented `stride_gw` training path uses online encoders, EMA target encoders, and cross-view decoders. It starts with diagonal cross-view supervision, then constructs within-view cosine-distance relations from the current training features. A finite number of structural GW updates starts from the uniform diagonal coupling `I / B`; after each update FASTNEWTON projects the candidate onto the uniform transport polytope. The returned coupling is scaled to row-stochastic pseudo-labels and used by the existing cross-view maximum-likelihood loss. The two directions are solved independently.

A configurable prior lock is available in selected experiment configurations. It can retain diagonal supervision after low measured diffusion or a configured projection failure. See the YAML itself for the active settings. The lock is not part of every dataset configuration.

The evaluation path extracts target-encoder features, performs the existing cross-view matching and feature concatenation, and reports ACC, NMI, ARI, F-score, and CAR.

## Repository layout

```text
config/                 Five canonical MR=0.8 experiment configurations
config/archive/          Older configurations retained for reference
solver/FASTNEWTON/       Standalone Euclidean transport projection package
solver/                  REBUILD projector adapter and reference interface
pseudolabel/gw.py         GW relations, updates, and bidirectional targets
main_train.py             YAML-driven training entry point
dataset_loader.py         Dataset selection and mismatch construction
engine_train.py           Training and evaluation orchestration
evaluation.py             Clustering and alignment metrics
tests/                    Unit and lightweight integration tests
archive/drot_comparison/  Historical comparison records; not a runtime solver
```

`docs/` is retained in the local project directory and is not part of the public repository tree. Dataset files, checkpoints, and run outputs are not included.

## Requirements

- Python 3.11 (the validated environment used Python 3.11)
- PyTorch 2.8.0
- NumPy, SciPy, scikit-learn, PyYAML, and Munkres
- A CUDA-capable PyTorch installation and NVIDIA GPU for the standard training configurations

FASTNEWTON uses PyTorch and does not require compiling a separate CUDA extension.

## Installation

From the repository root on Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e ./solver/FASTNEWTON
```

FASTNEWTON is installed from the checked-in package after the runtime dependencies are available. The no-build-isolation option uses the installed setuptools and avoids a separate build-environment download.

For the optional CVXPY-based reference tests:

```bash
python -m pip install -r requirements-test.txt
```

## Dataset preparation

Place the dataset files at the paths below, or update `data_path` in the selected YAML:

| Dataset | Expected file | YAML view order and input dimensions |
|---|---|---|
| Scene15 | `dataset/Scene_15.mat` | `[1, 0]`, 59-D then 20-D |
| Handwritten | `dataset/Handwritten.mat` or `dataset/HandWritten.mat` | `[0, 1]`, 240-D then 76-D |
| CCV20 | `dataset/CCV20.mat` or `dataset/CCV.mat` | `[1, 2]`, 20-D then 20-D |
| Food101 | `data/umpc_food101.mat` | `[0, 1]`, 768-D then 768-D |
| Reuters | `dataset/Reuters_dim10.mat` | `[0, 1]`, 10-D then 10-D |

The Handwritten order is based on the two-view matrices in the `Handwritten.mat` file used for these experiments: pixel averages (240 dimensions) followed by Fourier coefficients (76 dimensions). The loader checks configured input dimensions against loaded features. No dataset is distributed with this repository.

## Canonical configurations

Each root-level YAML is a recorded MR=0.8 configuration and runs five seeds starting at seed 1. The five formal configurations use `solver.backend: fastnewton` and `solver.max_iter: 200`.

| Config | Batch / eval batch | GW steps, step size | Other selected settings | Recorded final-epoch mean ACC, seeds 1–5 |
|---|---:|---:|---|---:|
| `config/scene15.yaml` | 512 / 512 | 3, 0.05 | momentum 0.98; diagonal prior lock at 0.001 | 43.30% |
| `config/handwritten.yaml` | 2000 / 2000 | 8, 0.01 | dropout 0.4; EMA dropout 0.0 | 87.82% |
| `config/ccv20.yaml` | 512 / 512 | 5, 0.02 | momentum 0.8; prior lock mass 0.1 | 22.31% |
| `config/food101.yaml` | 1024 / 1024 | 8, 0.01 | momentum 0.995; no prior lock for MR=0.8 | 45.01% |
| `config/reuters.yaml` | 1024 / 1024 | 4, 0.005 | dropout 0.5; momentum 0.995; prior lock mass 0.15 | 55.59% |

These figures summarize completed five-seed final-epoch records available during repository cleanup. They are configuration-specific observations, not an exhaustive hyperparameter optimum or a guarantee of identical results on other hardware and software stacks. Full run logs and checkpoints are not included.

The configurations set `m_ratio: 0.8`, `seed: 1`, and `num_runs: 5`; the training entry point runs seeds 1 through 5. The archived YAML files retain older experiment settings and are not the canonical defaults.

## Configuration and training

Validate a configuration without starting training:

```bash
python main_train.py --config_file config/scene15.yaml --validate_config
```

Run the selected five-seed experiment:

```bash
python main_train.py --config_file config/scene15.yaml
```

Replace `scene15.yaml` with `handwritten.yaml`, `ccv20.yaml`, `food101.yaml`, or `reuters.yaml` for the other configurations. `--config_file` accepts a repository-relative path or a path under `config/`.

The YAML is the source of dataset, view order, model dimensions, batch sizes, training schedule, GW parameters, prior-lock settings, and FASTNEWTON tolerances. `ema_dropout` controls only the EMA target encoder; if omitted, it inherits `drop_rate`. An explicit `ema_dropout: 0.0` disables dropout in that branch. The older `target_drop_rate` name remains a deprecated compatibility alias and conflicting values are rejected.

Each invocation creates a timestamped run ID and separate seed directories under the configured output directory. Existing per-run directories are not overwritten. Effective configuration snapshots and training logs are written with the run.

## Evaluation

Evaluation runs as part of training at the intervals configured by the training entry point. It uses the YAML evaluation batch size, which is explicit in all canonical configurations. The loader uses the same mismatch-corrupted views for training and evaluation. The metrics are reported by the existing evaluation implementation; CAR is retained as that implementation defines it.

## Reproducibility

- Five-seed runs use seeds 1–5; individual runs keep their seed-specific output directories.
- The effective YAML is saved for each run.
- FASTNEWTON is installed from the version included in this repository; its source provenance is recorded in `solver/FASTNEWTON/vendor/BASELINE_PROVENANCE.md`.
- Dataset files, checkpoints, and output directories are ignored by Git.
- The canonical configs select observed experiment settings; they do not imply that every setting was independently optimized.

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

## Acknowledgements and third-party attribution

The training and data-loading code descends from [CorreGen](https://github.com/XLearning-SCU/2026-ICLR-CorreGen). The GW training design was reconstructed with reference to [STRIDE](https://github.com/8421vsgray/STRIDE). The Euclidean transport solver is [FASTNEWTON](https://github.com/8421vsgray/FASTNEWTON), with its source provenance recorded in the checked-in solver package.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and licensing status. This repository does not include the datasets.
