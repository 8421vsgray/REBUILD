from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ProjectionDiagnostics:
    status: str
    iterations: int
    min_value: float
    row_abs_residual: float
    col_abs_residual: float
    row_relative_residual: float
    col_relative_residual: float


@dataclass(frozen=True)
class ProjectionResult:
    plan: torch.Tensor
    diagnostics: ProjectionDiagnostics


def validate_transport_plan(
    plan: torch.Tensor,
    row_marginal: torch.Tensor,
    col_marginal: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-3,
) -> ProjectionDiagnostics:
    """Recompute feasibility diagnostics without trusting a backend status."""
    if plan.ndim != 2:
        return ProjectionDiagnostics("invalid_input", 0, float("nan"), float("inf"), float("inf"), float("inf"), float("inf"))
    if plan.shape != (row_marginal.numel(), col_marginal.numel()):
        return ProjectionDiagnostics("invalid_input", 0, float("nan"), float("inf"), float("inf"), float("inf"), float("inf"))
    if not all(torch.isfinite(value).all().item() for value in (plan, row_marginal, col_marginal)):
        return ProjectionDiagnostics("numerical_failure", 0, float("nan"), float("inf"), float("inf"), float("inf"), float("inf"))
    if torch.any(plan < -atol) or torch.any(row_marginal <= 0) or torch.any(col_marginal <= 0):
        status = "infeasible_output" if torch.any(plan < -atol) else "invalid_input"
    else:
        status = "success"
    row_error = torch.abs(plan.sum(dim=1) - row_marginal)
    col_error = torch.abs(plan.sum(dim=0) - col_marginal)
    row_scale = torch.clamp(torch.abs(row_marginal), min=atol)
    col_scale = torch.clamp(torch.abs(col_marginal), min=atol)
    row_abs = float(row_error.max().item())
    col_abs = float(col_error.max().item())
    row_relative = float((row_error / row_scale).max().item())
    col_relative = float((col_error / col_scale).max().item())
    if row_abs > atol and row_relative > rtol or col_abs > atol and col_relative > rtol:
        status = "infeasible_output"
    return ProjectionDiagnostics(
        status,
        0,
        float(plan.min().item()),
        row_abs,
        col_abs,
        row_relative,
        col_relative,
    )


class TransportProjector:
    """Backend-neutral transport projection contract."""

    def project(
        self,
        candidate: torch.Tensor,
        row_marginal: torch.Tensor,
        col_marginal: torch.Tensor,
    ) -> ProjectionResult:
        raise NotImplementedError
