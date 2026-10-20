# REBUILD YAML configuration contract

## Standard entry point

The standard configuration entry point is:

```bash
python main_train.py --config_file config/Scene15_MR08.yaml
```

The loader accepts either a repository-relative path such as
`config/Scene15_MR08.yaml` or a basename resolved below `config/`.
`yaml.safe_load` is used, and unknown top-level fields or unknown fields inside
`solver` are rejected.

## Required standard fields

The standard schema records the dataset, view order, normalization, input
dimensions, training schedule, mismatch ratio, and GW/DROT settings. The
`view_indices` order is authoritative: index 0 is the anchor view and index 1
is the view whose correspondence is perturbed by MR.

Formal YAML experiments must declare `num_runs` explicitly. The maintained
Scene15 smoke configuration uses `seed: 1` and `num_runs: 1`; this prevents a
single-seed request from silently expanding into the legacy five-run loop.

`diagonal_epochs` is kept separate from `warmup_epochs`. The former controls
the supervision phase; the latter controls the learning-rate schedule.

The formal training/evaluation batch-size ceiling is 512, and maintained
dataset YAML files explicitly use `batch_size: 512` and
`eval_batch_size: 512`. B=1024 is a solver stress-diagnostic size only.

## Solver separation

The nested `solver` mapping separates DROT parameters from GW parameters:

- `steps` and `step_size` belong to GW diffusion;
- `solver.rho`, `solver.eps`, and `solver.max_iter` belong to DROT;
- `init: diagonal` selects the `I/B` initialization.

No parameter is silently accepted and ignored. The current REBUILD branch
still contains the CorreGen baseline training backend; therefore a config with
`algorithm: stride_gw` is validated and then stopped explicitly until the
two-stage GW training backend is integrated. This is intentional: it prevents
the standard STRIDE fields from giving a false impression that they already
control the baseline model.

Use `--validate_config` to validate and print a standard configuration without
starting training.

When training is eventually enabled, each invocation receives a unique run ID
and writes per-seed artifacts below:

```text
<output>/<dataset>/<experiment>/run_<timestamp>_<id>/seed_<seed>/
```

An existing run directory is never silently reused.
