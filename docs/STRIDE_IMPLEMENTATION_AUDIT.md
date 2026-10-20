# STRIDE implementation audit

Date: 2026-10-20

This document is an audit of the code currently available on `drift`; it is
not a claim that every historical experiment used the same source. The
CorreGen baseline is pinned in REBUILD at commit
`c16f9d37fc7e7d3e82ff34d2b6b118877a953924`. The STRIDE reference checkout is
`/home/dingyn/code/NEWSTRIDE`, remote `git@github.com:8421vsgray/STRIDE.git`,
commit `c27937b68cb1656cde90e6fb1f724d92d55015d0`; that checkout has
uncommitted changes, so line references below are marked as working-tree
references where relevant.

## 1. CorreGen baseline behavior

The REBUILD starting point is the official CorreGen implementation.

- `model.py:20-44` creates online and EMA target encoders. The target branch
  is initialized from the online branch and updated by
  `target = m * target + (1-m) * online`.
- `model.py:65-103` estimates intra-view relations with entropic Wasserstein
  OT and cross-view relations with Partial OT. The intra-view plan is mixed
  with the identity matrix using `beta`; the cross-view plan is rescaled by
  the retained mass.
- `model.py:120-134` uses identity supervision during the rectify phase and
  otherwise combines intra- and cross-view maximum-likelihood losses.
- `model.py:186-198` normalizes features, flattens the complete `B x B`
  similarity matrix, applies one global softmax, rescales it by `B`, and
  multiplies by a detached target. This is not row-wise InfoNCE.
- `main_train.py:90-111` uses `RandomSampler` for both training and
  evaluation, but the evaluation DataLoader hard-codes `batch_size=512`.
- `eval.py:68-89` writes features back by the sample indices, concatenates
  the complete dataset, runs cosine K-means, and computes CAR from the
  selected view-2 labels. `eval.py:94-110` fixes K-means randomness at zero.

## 2. Data protocol and order-sensitive behavior

In the pinned CorreGen source:

- `dataset_loader.py:64-80` uses `RandomState(1234)` to select and permute
  the affected rows of later views; view-2 labels and identity IDs are
  permuted with the features.
- `dataset_loader.py:82-97` constructs optional corruption noise after the
  mismatch operation.
- `dataset_loader.py:99-113` normalizes the resulting arrays. Therefore the
  historical CorreGen order is effectively corrupt/permute, then normalize.
- `dataset_loader.py:118-125` uses noisy data for training when CR is enabled
  and clean data for evaluation; this is a transductive protocol, not a
  conventional disjoint train/test split.

The audited STRIDE reference loader (`NEWSTRIDE/data_loader.py:165-203`)
normalizes first and then applies the fixed partial shuffle. It also returns
the same correspondence-corrupted data object for train and test. These are
material protocol differences and must be represented explicitly in REBUILD;
they must not be inferred from a dataset name.

## 3. STRIDE mechanisms found in the reference checkout

The following observations refer to the `NEWSTRIDE` working tree and must be
rechecked against a clean pinned STRIDE commit before implementation claims
are made.

- `model.py:23-101`: online/target encoders, EMA target updates, and optional
  temporal evaluation fusion.
- `model.py:206-218`: diagonal warm-up uses an identity matrix as the
  cross-view teacher before `warmup_epochs` completes.
- `model.py:220-267`: after warm-up, normalized decoder/target features form
  view relations; the two cross-view directions call the transport solver.
- `model.py:272-281`: intra-view supervision remains diagonal while the
  detached transport plans supervise the two cross-view decoder directions.
- `transport_solver.py:555-630`: the initial transport is either
  `I * (1/B)` or a uniform feasible plan; the solver stores mass convention
  and later returns `B*T` as a row-stochastic teacher.
- `transport_solver.py:632-657`: the finite update is
  `X = T - eta * G(T)`, where the structural gradient is implemented as
  `-2 * (D1 @ T @ D2.T + D1.T @ T @ D2)`.
- `transport_solver.py:166-220`: the current reference projector is a
  self-authored semismooth-Newton Euclidean projection onto the uniform
  transport polytope, not Regularized-DROT.
- `transport_solver.py:778-899`: optional stabilization, energy guards,
  update mixing, prior gates, warm starts, and strict repairs can alter the
  trajectory. They must become explicit switches, not hidden defaults.
- `model.py:322-347` preserves the global flattened-softmax likelihood
  convention and detaches the pseudo-label target.

## 4. Evaluation differences that must be fixed in REBUILD

The reference STRIDE entry (`NEWSTRIDE/main_train.py:110-118`) currently
creates a shuffled training loader but a sequential evaluation loader with
`batch_size=min(512, len(test_set))`. CorreGen uses an independent
`RandomSampler` for evaluation but also fixes the batch size to 512.

REBUILD must therefore:

1. expose `eval_batch_size` in YAML with default `batch_size`;
2. expose the evaluation sampler/protocol explicitly;
3. record effective train/eval batch sizes in the runtime snapshot;
4. verify full index coverage and no duplicate/missing evaluation indices;
5. keep the anchor-view labels separate from the disturbed-view labels used
   for CAR;
6. keep K-means seed and metric definitions explicit.

Changing the evaluation batch size is an evaluation-protocol change. It must
not be silently used to select the most favorable checkpoint or dataset.

## 5. What is not yet established

The following are intentionally unverified at this stage:

- whether the historical STRIDE experiments used exactly the current
  `NEWSTRIDE` working tree;
- whether the displayed GW gradient is the complete gradient of the intended
  objective under all relation symmetry assumptions;
- whether the existing Newton projection is numerically equivalent to the
  intended Euclidean projection at real batch sizes;
- whether Regularized-DROT's quadratic objective and scaling match that
  projection without a formal reduction and independent reference solver;
- whether prior gates, energy guards, adaptive ridge, or update mixing belong
  in the minimal STRIDE algorithm or only in historical variants;
- whether any existing result used a different normalization/shuffle order,
  target-decoder EMA policy, or evaluation sampler.

No DROT result, STRIDE performance result, or formal reproduction claim is
made by this audit.

## 6. Migration boundary

The REBUILD implementation should preserve CorreGen's verified training and
loss behavior first. The relation builder, transport projector, and
pseudo-label generator must be separate interfaces. The existing Newton
projector may be retained only as a labeled reference backend; it must not be
silently replaced by DROT or reported as DROT. DROT integration is blocked on
its own CUDA build and small-matrix numerical validation.
