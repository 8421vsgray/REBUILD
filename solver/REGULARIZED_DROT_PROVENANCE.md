# Regularized-DROT provenance

This directory is a vendored source snapshot of the upstream Regularized-DROT
project. It is kept separate from REBUILD adapter code and is not modified by
the REBUILD naming or algorithm refactors.

- Upstream repository: https://github.com/WangZesen/Regularized-DROT
- Source archive: `Regularized-DROT-main.zip`
- Archive SHA-256: `C0D63BC7A948F274D89003FB853EA2A85F17491612D5B25C1DC499567EE5C8D0`
- Snapshot date: 2026-10-20
- Upstream Git commit: unavailable from the supplied archive; no commit is
  fabricated.
- Local source modification: none intended; the archive contents are placed
  under `solver/Regularized-DROT/` for auditable integration.

The PyTorch extension is defined by `pytorch/setup.py` and exposes
`reg_drot.quadratic_drot`. Building and running the official test remains a
separate verification step. A copied source tree is not evidence that the
CUDA extension compiled or that its numerical output satisfies REBUILD's
projection acceptance criteria.
