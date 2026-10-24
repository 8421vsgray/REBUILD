"""Independent finite GW pseudo-label generation for REBUILD.

This module is intentionally outside the training model. It consumes source
and target embeddings, performs finite structural diffusion with a pluggable
Euclidean transport projector, and returns detached row-stochastic labels.
The two cross-view directions are solved independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from solver.projection_interface import TransportProjector, validate_transport_plan


class GWProjectionFailure(RuntimeError):
    """Raised when one finite GW projection is not feasible."""


@dataclass(frozen=True)
class GWStepDiagnostics:
    step: int
    projection_status: str
    projection_seconds: float
    gradient_norm: float
    row_abs_residual: float
    col_abs_residual: float
    row_relative_residual: float
    col_relative_residual: float
    minimum_value: float
    off_diagonal_mass: float
    optimality_status: str = "unverified"


def _require_feature_pair(source_features: torch.Tensor, target_features: torch.Tensor) -> int:
    if not torch.is_tensor(source_features) or not torch.is_tensor(target_features):
        raise TypeError("source_features and target_features must be tensors")
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("source_features and target_features must be matrices")
    if source_features.shape[0] != target_features.shape[0]:
        raise ValueError("source and target batch sizes must match")
    if source_features.shape[0] <= 0:
        raise ValueError("batch size must be positive")
    if source_features.device != target_features.device:
        raise ValueError("source and target features must share a device")
    if source_features.dtype != target_features.dtype:
        raise ValueError("source and target features must share a dtype")
    if not source_features.is_floating_point():
        raise TypeError("features must use a floating-point dtype")
    if not torch.isfinite(source_features).all() or not torch.isfinite(target_features).all():
        raise ValueError("features must be finite")
    return source_features.shape[0]


@torch.no_grad()
def relation_matrix(features: torch.Tensor) -> torch.Tensor:
    """Return the symmetrized cosine-distance relation matrix."""
    if features.ndim != 2 or features.shape[0] <= 0:
        raise ValueError("features must be a non-empty matrix")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("features must be finite floating-point values")
    normalized = F.normalize(features.detach(), dim=1)
    relation = (1.0 - normalized @ normalized.t()).clamp(0.0, 2.0)
    relation = 0.5 * (relation + relation.t())
    if not torch.isfinite(relation).all():
        raise ValueError("relation matrix contains non-finite values")
    return relation


@torch.no_grad()
def pointwise_gradient(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    """Return the fixed cross-view cosine-distance gradient.

    This is the pointwise ablation counterpart to gw_gradient. The returned
    matrix is 1 - cosine_similarity(source_i, target_j) and is therefore the
    gradient of the linear pointwise matching objective with respect to the
    transport plan. Features are normalized exactly as in relation_matrix;
    no transport or model gradients are propagated.
    """
    batch_size = _require_feature_pair(source_features, target_features)
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError(
            "source and target feature dimensions must match for pointwise similarity"
        )
    source_normalized = F.normalize(source_features.detach(), dim=1)
    target_normalized = F.normalize(target_features.detach(), dim=1)
    gradient = (1.0 - source_normalized @ target_normalized.t()).clamp(0.0, 2.0)
    if gradient.shape != (batch_size, batch_size) or not torch.isfinite(gradient).all():
        raise ValueError("pointwise gradient contains invalid values")
    return gradient


def gw_gradient(
    source_relation: torch.Tensor,
    target_relation: torch.Tensor,
    transport: torch.Tensor,
) -> torch.Tensor:
    """Return the finite STRIDE structural GW gradient."""
    if source_relation.ndim != 2 or target_relation.ndim != 2 or transport.ndim != 2:
        raise ValueError("GW relations and transport must be matrices")
    if source_relation.shape[0] != source_relation.shape[1]:
        raise ValueError("source relation must be square")
    if target_relation.shape[0] != target_relation.shape[1]:
        raise ValueError("target relation must be square")
    if transport.shape != (source_relation.shape[0], target_relation.shape[0]):
        raise ValueError("transport shape is incompatible with relations")
    return -2.0 * (
        source_relation @ transport @ target_relation.t()
        + source_relation.t() @ transport @ target_relation
    )


def _projector_parameters(projector: TransportProjector) -> dict[str, Any]:
    return {
        "backend": projector.__class__.__name__,
        "rho": getattr(projector, "rho", None),
        "eps": getattr(projector, "solver_eps", None),
        "relative_tol": getattr(projector, "relative_tolerance", None),
        "max_iter": getattr(projector, "max_iterations", None),
        "warm_start": getattr(projector, "warm_start", None),
    }


def _validate_gw_config(config: Mapping[str, Any]) -> tuple[str, int, float, str]:
    init = str(config.get("init", "diagonal")).lower()
    if init != "diagonal":
        raise ValueError("GW generator currently supports init=diagonal only")
    steps = config.get("steps")
    step_size = config.get("step_size")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if not isinstance(step_size, (float, int)) or isinstance(step_size, bool):
        raise ValueError("step_size must be numeric")
    step_size = float(step_size)
    if step_size < 0.0:
        raise ValueError("step_size must be non-negative")
    transport_gradient = str(
        config.get("transport_gradient", "structural")
    ).lower()
    if transport_gradient not in {"structural", "pointwise"}:
        raise ValueError(
            "transport_gradient must be structural or pointwise"
        )
    return init, steps, step_size, transport_gradient


@torch.no_grad()
def generate_gw_targets(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    gw_config: Mapping[str, Any],
    projector: TransportProjector,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Generate one detached row-stochastic GW pseudo-label matrix."""
    batch_size = _require_feature_pair(source_features, target_features)
    _, steps, step_size, transport_gradient = _validate_gw_config(gw_config)
    if not isinstance(projector, TransportProjector):
        raise TypeError("projector must implement TransportProjector")

    source = source_features.detach()
    target = target_features.detach()
    if transport_gradient == "structural":
        source_relation = relation_matrix(source)
        target_relation = relation_matrix(target)
        fixed_gradient = None
    else:
        source_relation = None
        target_relation = None
        fixed_gradient = pointwise_gradient(source, target)
    prior_gate_precheck = bool(gw_config.get("prior_gate_precheck", False))
    raw_offdiag_tol = float(gw_config.get("raw_offdiag_tol", 1.0e-3))
    marginal = torch.full(
        (batch_size,),
        1.0 / float(batch_size),
        dtype=source.dtype,
        device=source.device,
    )
    transport = torch.eye(
        batch_size, dtype=source.dtype, device=source.device
    ) * marginal[0]

    step_diagnostics: list[GWStepDiagnostics] = []
    total_projection_seconds = 0.0
    for step in range(steps):
        if transport_gradient == "structural":
            gradient = gw_gradient(source_relation, target_relation, transport)
        else:
            gradient = fixed_gradient
        candidate = transport - step_size * gradient
        if step == 0 and prior_gate_precheck:
            positive_candidate = candidate.clamp_min(0.0)
            positive_total = float(positive_candidate.sum().item())
            if positive_total > 0.0:
                raw_offdiag_mass = 1.0 - float(
                    positive_candidate.diagonal().sum().item() / positive_total
                )
            else:
                raw_offdiag_mass = 0.0
            if raw_offdiag_mass <= raw_offdiag_tol:
                pseudo_labels = (transport * float(batch_size)).detach()
                return pseudo_labels, {
                    "status": "prior_gate",
                    "prior_gate": True,
                    "prior_gate_reason": "low_raw_diffusion",
                    "batch_size": batch_size,
                    "steps": steps,
                    "step_size": step_size,
                    "transport_gradient": transport_gradient,
                    "solver_parameters": _projector_parameters(projector),
                    "row_abs_residual": 0.0,
                    "col_abs_residual": 0.0,
                    "row_relative_residual": 0.0,
                    "col_relative_residual": 0.0,
                    "non_diagonal_mass": 0.0,
                    "raw_offdiag_mass": raw_offdiag_mass,
                    "projection_seconds": 0.0,
                    "step_diagnostics": [],
                }
        projection_start = time.perf_counter()
        try:
            result = projector.project(candidate, marginal, marginal)
        except Exception as error:
            raise GWProjectionFailure(
                "GW projection raised at step {}: {}".format(step, error)
            ) from error
        projection_seconds = time.perf_counter() - projection_start
        diagnostics = result.diagnostics
        if diagnostics.status not in {"success", "feasible_only"}:
            raise GWProjectionFailure(
                "GW projection status={} at step {}".format(
                    diagnostics.status, step
                )
            )
        transport = result.plan.detach().to(
            device=source.device, dtype=source.dtype
        )
        if (transport < -1e-6).any():
            raise GWProjectionFailure(
                "GW projection returned materially negative mass at step {}".format(
                    step
                )
            )
        # Backends are allowed a small floating-point feasibility tolerance;
        # expose mathematically valid nonnegative pseudo-labels downstream.
        transport = transport.clamp_min(0.0)
        feasibility = validate_transport_plan(
            transport,
            marginal,
            marginal,
            atol=1e-6,
            rtol=1e-3,
        )
        if feasibility.status != "success":
            raise GWProjectionFailure(
                "GW projection failed post-validation at step {}: {}".format(
                    step, feasibility.status
                )
            )
        pseudo_labels = transport * float(batch_size)
        off_diagonal_mass = 1.0 - float(pseudo_labels.diagonal().mean().item())
        step_diagnostics.append(
            GWStepDiagnostics(
                step=step,
                projection_status=diagnostics.status,
                projection_seconds=projection_seconds,
                gradient_norm=float(gradient.norm().item()),
                row_abs_residual=feasibility.row_abs_residual,
                col_abs_residual=feasibility.col_abs_residual,
                row_relative_residual=feasibility.row_relative_residual,
                col_relative_residual=feasibility.col_relative_residual,
                minimum_value=float(transport.min().item()),
                off_diagonal_mass=off_diagonal_mass,
                optimality_status=(
                    "verified" if diagnostics.optimality_verified else "unverified"
                ),
            )
        )
        total_projection_seconds += projection_seconds

    pseudo_labels = (transport * float(batch_size)).detach()
    ones = torch.ones(batch_size, dtype=source.dtype, device=source.device)
    if not torch.isfinite(pseudo_labels).all() or (pseudo_labels < 0).any():
        raise GWProjectionFailure("generated pseudo-labels are invalid")
    if not torch.allclose(pseudo_labels.sum(dim=1), ones, atol=1e-5, rtol=1e-3):
        raise GWProjectionFailure("generated pseudo-label rows are not normalized")

    diagonal_mass = float(pseudo_labels.diagonal().mean().item())
    diagnostics = {
        "status": "success",
        "prior_gate": False,
        "prior_gate_reason": None,
        "raw_offdiag_mass": None,
        "optimality": (
            "verified"
            if step_diagnostics
            and all(item.optimality_status == "verified" for item in step_diagnostics)
            else "unverified"
        ),
        "batch_size": batch_size,
        "steps": steps,
        "step_size": step_size,
        "transport_gradient": transport_gradient,
        "solver_parameters": _projector_parameters(projector),
        "row_abs_residual": max(
            (item.row_abs_residual for item in step_diagnostics), default=0.0
        ),
        "col_abs_residual": max(
            (item.col_abs_residual for item in step_diagnostics), default=0.0
        ),
        "row_relative_residual": max(
            (item.row_relative_residual for item in step_diagnostics), default=0.0
        ),
        "col_relative_residual": max(
            (item.col_relative_residual for item in step_diagnostics), default=0.0
        ),
        "non_diagonal_mass": 1.0 - diagonal_mass,
        "projection_seconds": total_projection_seconds,
        "step_diagnostics": step_diagnostics,
    }
    return pseudo_labels, diagnostics


@torch.no_grad()
def generate_bidirectional_gw_targets(
    predictions: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
    target_embeddings: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
    gw_config: Mapping[str, Any],
    projector: TransportProjector,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Generate independent P12 and P21 targets; never force a transpose.

    ``predictions`` contains the two online cross-view predictions ``p[0]``
    and ``p[1]``. ``target_embeddings`` contains the two EMA embeddings
    ``z_t[0]`` and ``z_t[1]``. The directions intentionally use different
    source/target pairs:

    ``P12 = GW(p[0], z_t[1])`` and ``P21 = GW(p[1], z_t[0])``.
    """
    if len(predictions) != 2 or len(target_embeddings) != 2:
        raise ValueError("predictions and target_embeddings must each contain two views")
    reset_state = getattr(projector, "reset_state", None)
    if reset_state is not None:
        reset_state()
    p12, diagnostics12 = generate_gw_targets(
        predictions[0], target_embeddings[1], gw_config, projector
    )
    if reset_state is not None:
        reset_state()
    p21, diagnostics21 = generate_gw_targets(
        predictions[1], target_embeddings[0], gw_config, projector
    )
    return {
        "P12": p12,
        "P21": p21,
    }, {
        "P12": diagnostics12,
        "P21": diagnostics21,
    }


def projector_from_solver_config(solver_config: Mapping[str, Any]) -> TransportProjector:
    """Construct FASTNEWTON with its own explicit solver configuration."""
    backend = str(solver_config.get("backend", "")).lower()
    if backend == "drot":
        raise ValueError(
            "solver.backend 'drot' has been removed; use 'fastnewton'"
        )
    if backend != "fastnewton":
        raise ValueError("unsupported GW projector backend: {}".format(backend))

    allowed = {
        "backend",
        "relative_tol",
        "max_iter",
        "absolute_tol",
        "certificate_tol",
        "warm_start",
    }
    unknown = sorted(set(solver_config).difference(allowed))
    if unknown:
        raise ValueError(
            "Unsupported FASTNEWTON solver field(s): {}".format(
                ", ".join(unknown)
            )
        )
    required = {"backend", "relative_tol", "max_iter"}
    missing = sorted(required.difference(solver_config))
    if missing:
        raise ValueError("solver config missing: {}".format(", ".join(missing)))

    from solver.fastnewton_adapter import FastNewtonEuclideanProjector

    return FastNewtonEuclideanProjector(
        relative_tolerance=float(solver_config["relative_tol"]),
        max_iterations=int(solver_config["max_iter"]),
        absolute_tolerance=float(solver_config.get("absolute_tol", 1.0e-8)),
        certificate_tolerance=(
            None
            if "certificate_tol" not in solver_config
            else float(solver_config["certificate_tol"])
        ),
        warm_start=solver_config.get("warm_start", True),
    )
