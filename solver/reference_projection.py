import torch

from .projection_interface import (
    ProjectionDiagnostics,
    ProjectionResult,
    TransportProjector,
    validate_transport_plan,
)


class DykstraEuclideanProjector(TransportProjector):
    """Independent CPU reference for Euclidean projection onto a transport polytope.

    The production DROT adapter must be validated against this backend on small
    problems. This implementation is intentionally simple and is not a
    replacement for a GPU solver.
    """

    def __init__(self, max_iterations=10_000, relative_tolerance=1e-8):
        self.max_iterations = max_iterations
        self.relative_tolerance = relative_tolerance

    def project(self, candidate, row_marginal, col_marginal):
        if candidate.ndim != 2:
            raise ValueError("candidate must be a matrix")
        if candidate.device.type != "cpu":
            raise ValueError("reference projection expects CPU tensors")
        if candidate.shape != (row_marginal.numel(), col_marginal.numel()):
            raise ValueError("candidate and marginal dimensions do not match")
        if not torch.isfinite(candidate).all():
            diagnostics = ProjectionDiagnostics(
                "invalid_input", 0, float("nan"), float("inf"), float("inf"), float("inf"), float("inf")
            )
            return ProjectionResult(candidate.detach().clone(), diagnostics)
        if torch.any(row_marginal <= 0) or torch.any(col_marginal <= 0):
            raise ValueError("marginals must be positive")
        if not torch.isclose(row_marginal.sum(), col_marginal.sum(), rtol=1e-6, atol=1e-8):
            raise ValueError("row and column marginals must have equal mass")

        work = candidate.detach().clone().to(dtype=torch.float64)
        rows = row_marginal.detach().clone().to(dtype=torch.float64)
        cols = col_marginal.detach().clone().to(dtype=torch.float64)
        corrections = [torch.zeros_like(work) for _ in range(3)]
        diagnostics = None

        for iteration in range(1, self.max_iterations + 1):
            shifted = work + corrections[0]
            projected = torch.clamp(shifted, min=0)
            corrections[0] = shifted - projected
            work = projected

            shifted = work + corrections[1]
            projected = shifted + (rows - shifted.sum(dim=1))[:, None] / work.shape[1]
            corrections[1] = shifted - projected
            work = projected

            shifted = work + corrections[2]
            projected = shifted + (cols - shifted.sum(dim=0))[None, :] / work.shape[0]
            corrections[2] = shifted - projected
            work = projected

            diagnostics = validate_transport_plan(
                work, rows, cols, atol=1e-10, rtol=self.relative_tolerance
            )
            if diagnostics.status == "success":
                diagnostics = ProjectionDiagnostics(
                    diagnostics.status,
                    iteration,
                    diagnostics.min_value,
                    diagnostics.row_abs_residual,
                    diagnostics.col_abs_residual,
                    diagnostics.row_relative_residual,
                    diagnostics.col_relative_residual,
                )
                return ProjectionResult(work, diagnostics)

        diagnostics = ProjectionDiagnostics(
            "not_converged",
            self.max_iterations,
            diagnostics.min_value,
            diagnostics.row_abs_residual,
            diagnostics.col_abs_residual,
            diagnostics.row_relative_residual,
            diagnostics.col_relative_residual,
        )
        return ProjectionResult(work, diagnostics)
