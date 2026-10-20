# REBUILD source provenance

## CorreGen upstream baseline

- Upstream repository: `https://github.com/XLearning-SCU/2026-ICLR-CorreGen.git`
- Baseline commit: `c16f9d37fc7e7d3e82ff34d2b6b118877a953924`
- Baseline tag: `corregen-baseline-20261020`
- Development branch: `rebuild/develop`
- Target project: `/home/dingyn/code/REBUILD`

The baseline commit was verified in an existing local clone on `drift` whose
`origin` points to the official CorreGen repository. The existing clone had
uncommitted experiment changes, so REBUILD was created with
`git clone --no-hardlinks` from its committed tree. Uncommitted files and
working-tree changes were not copied.

The REBUILD `origin` remote was then reset to the official upstream URL. The
drift host could not establish a GitHub HTTPS connection during initialization,
so the already-present official clone was used as the transport source. This
fact is recorded rather than hidden.

## Initial verification

The following were checked after initialization:

- the target path did not exist before cloning;
- the target has a Git work tree and preserves the upstream commit history;
- `HEAD` equals the baseline commit above;
- the baseline tag points at `HEAD`;
- the active branch is `rebuild/develop`;
- the target origin is the official CorreGen URL.

No STRIDE code, DROT code, dataset, checkpoint, or training artifact has been
added at this stage.
