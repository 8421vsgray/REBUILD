# REBUILD changelog

## 2026-10-20 — formal batch boundary and multi-location checkpoint

Commit: recorded in Git history after this document is added.

Completed:

- changed all maintained dataset YAML files to explicit
  `batch_size: 512` and `eval_batch_size: 512`;
- added a training/evaluation guard rejecting formal batch sizes above 512;
- kept B=1024 DROT measurements as solver stress diagnostics only;
- recorded the new RTX 3090 build caveat and timing evidence separately from
  the historical RTX 4090 CUDA 12.8 evidence;
- established the intended 3090-primary, local-mirror and drift-backup Git
  synchronization policy.

Validation:

- configuration and documentation edits pass `git diff --check`;
- the batch protocol regression tests are pending until this checkpoint is
  committed;
- no model math or third-party DROT source was changed by this protocol edit.

## 2026-10-20 — Stage 0: initialize from CorreGen

Commit: recorded in Git history after this document is added.

Completed:

- created `/home/dingyn/code/REBUILD` from the committed CorreGen upstream tree;
- preserved the upstream Git history;
- added annotated tag `corregen-baseline-20261020`;
- created branch `rebuild/develop`;
- recorded the upstream URL and baseline SHA in `docs/REBUILD_SOURCE_PROVENANCE.md`.

Validation:

- target path was absent before creation;
- baseline SHA: `c16f9d37fc7e7d3e82ff34d2b6b118877a953924`;
- PyTorch environment is available on `drift` (`2.8.0+cu128`, CUDA available);
- no training or model implementation changes were made in Stage 0.

Not yet verified:

- CorreGen end-to-end baseline training;
- STRIDE implementation audit;
- configurable evaluation batch behavior;
- Regularized-DROT build and numerical correctness;
- GW, training, and formal experiment results.

## 2026-10-20 — Stage 1: audit and execution scope

Commits:

- `a8ccd62` — `docs: audit stride implementation`;
- `f9966d6` — `docs: record rebuild execution prompts`;
- `2c350d3` — `docs: add engineering constraints and freeze scope`.

Completed:

- recorded the CorreGen/STRIDE implementation audit;
- recorded the full REBUILD task, naming-refactor task, legacy-algorithm-removal task, and engineering constraints;
- froze the scope against unrequested losses, gates, matching strategies, and result-driven changes;
- required a final evidence-based `docs/REBUILD_VERIFICATION_REPORT.md`.

Validation:

- prompt documents are stored under `docs/`;
- `git diff --check` passed before the commits;
- no model, optimizer, data, or evaluation implementation was changed by these documentation commits.

## 2026-10-20 — Stage 2: evaluation protocol and data contracts

Commits:

- `3622bdd` — `fix: make evaluation batch size configurable`;
- `9ac9cd4` — `chore: ignore generated experiment artifacts`;
- `5d79eae` — `test: establish data protocol contracts`.

Completed:

- added `eval_batch_size`, defaulting to the configured training batch size;
- removed the evaluation DataLoader's hard-coded `512` batch size;
- kept independent RandomSamplers and `drop_last=False` for train/evaluation;
- added explicit train/evaluation batch logging;
- documented the current protocol in `docs/EXPERIMENT_PROTOCOL.md`;
- added evaluation coverage and data/noise protocol tests;
- added repository ignores for generated caches and training artifacts.

Validation:

- `python -m py_compile main_train.py tests/test_eval_protocol.py` passed;
- `python -m unittest discover -s tests -p "test_*.py" -v` passed: 7 tests;
- Scene15 fixture checks passed for repeatable MR construction, sample identity indexing, and CR train/evaluation separation;
- DROT, GW, two-stage training, and end-to-end training remain unverified.

## 2026-10-20 — Stage 2b: NC-only data scope

Commits:

- `fc0e813` — `docs: define nc-only data scope`;
- `73db0ed` — `refactor(data): remove corruption ratio augmentation`;
- `86c4930` — `test(data): verify mismatch-only protocol`;
- `82b420f` — `docs: document NC-only evaluation protocol`.

Completed:

- removed `c_ratio` from the training CLI and output-directory naming;
- rejected `c_ratio` in YAML instead of silently ignoring it;
- removed CR Gaussian feature-noise generation and the noisy training-data branch;
- kept MR permutation, sample identities, labels for offline checks, and standardization;
- made training and evaluation use the same MR-processed data version;
- updated the protocol document to define NC-only scope.

Validation:

- `python -m py_compile` passed for the changed loader, entrypoint, and data tests;
- the complete test suite passed: 9 tests;
- repeated MR construction, zero-MR identity, Anchor preservation, train/evaluation identity, and removed-config rejection passed;
- remaining `c_ratio` references are only the explicit rejection check and its test fixture;
- CorreGen OT/GMM removal, DROT, GW, and end-to-end training remain unverified.

## 2026-10-20 — Stage 3: naming refactor

Commits:

- `3eebe62` — `docs: define rebuild naming conventions`;
- `d38b1e3` — `refactor(naming): clarify model and function names`;
- `61b8205` — `refactor(naming): clarify training and solver variables`;
- `7325be6` — `refactor(naming): standardize module names` (`eval.py` → `evaluation.py`);
- `9ba71bf` — `refactor(naming): standardize module names` (import update);
- `5b2e12c` — `refactor(config): standardize configuration names`;
- `d76431e` — `refactor(logging): standardize experiment diagnostics`.

Completed:

- renamed the active baseline class to `CorreGenBaselineModel` while retaining an explicit compatibility alias for old serialized objects;
- clarified training state/phase and evaluation sample-index variable names;
- renamed the abbreviated evaluation module and updated imports;
- mapped legacy `train_time` YAML to `num_runs` with conflict detection;
- made experiment directory and log names explicit without changing metric definitions.

Validation:

- state-dict key lists and deterministic CPU feature outputs match through the class-name compatibility path;
- all imports and module compilation passed;
- the complete test suite passed: 14 tests;
- no naming-stage commit changed the model math, optimizer, data protocol, or metric definitions.

Not yet verified:

- CorreGen baseline end-to-end training;
- DROT compilation and numerical equivalence to Euclidean projection;
- GW gradient and pseudo-label tests;
- removal of Sinkhorn/Partial OT/GMM and two-stage training integration.
