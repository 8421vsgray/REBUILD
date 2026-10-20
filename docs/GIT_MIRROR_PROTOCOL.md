# REBUILD Git mirror protocol

## Purpose

REBUILD is kept in multiple locations so that a server failure or an
uncommitted experiment cannot destroy the only copy of the source history.
Git commits are the source of truth; CUDA extensions, checkpoints, logs and
datasets are not copied into the source mirror.

## Current locations

- Primary development worktree: `/root/code/REBUILD` on the RTX 3090 host,
  branch `rebuild/develop`.
- Local bare mirror: `REBUILD-mirror/REBUILD.git` on the Windows development
  machine.
- Drift copy: `/home/dingyn/code/REBUILD` on branch `rebuild/develop`.
  While its worktree has unrelated generated-file modifications, new commits
  are first stored there as immutable `backup/<3090-commit>` refs rather than
  overwriting the worktree.

The public CorreGen repository remains the upstream `origin`; it is not used
as a destination for REBUILD commits.

## Checkpoint procedure

After each independently tested change:

1. Run the relevant tests and `git diff --check` on the 3090 worktree.
2. Commit on `rebuild/develop` with a semantic message.
3. Create a Git bundle from the commit range on the 3090 host.
4. Import that bundle into the local bare mirror and verify the commit SHA.
5. Import the same commit into `drift` under an immutable backup ref and verify
   the SHA.
6. Only fast-forward the active drift worktree after its existing local
   modifications have been explicitly reconciled.

No force-push, broad reset, or `git clean` is part of this procedure. If a
worktree is dirty, the dirty state is preserved and the new commit is stored
under a separate ref until it is safe to advance the active branch.

## Required evidence

Every checkpoint records:

- commit SHA;
- source worktree status;
- tests and validation commands;
- actual YAML/configuration;
- environment and GPU where relevant.

The same SHA must be visible in the 3090 branch, the local bare mirror and a
drift backup ref before the checkpoint is considered multi-location saved.
