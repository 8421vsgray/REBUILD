# REBUILD changelog

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
