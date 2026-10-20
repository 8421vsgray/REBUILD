"""Explicit adapter from Euclidean transport projection to Regularized-DROT.

The adapter deliberately has no fallback solver.  A missing or failing CUDA
extension is surfaced through a non-success diagnostic and must be handled by
the caller before a plan is used for supervision.
"""

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any, Optional

import torch

from .projection_interface import (
    ProjectionDiagnostics,
    ProjectionResult,
    TransportProjector,
    validate_transport_plan,
)


@dataclass(frozen=True)
class DrotInputs:
    cost: torch.Tensor
    row_marginal: torch.Tensor
    col_marginal: torch.Tensor
    scale: float
    regularization_weight: float


def prepare_drot_inputs(
    candidate: torch.Tensor,
    row_marginal: torch.Tensor,
    col_marginal: torch.Tensor,
) -> DrotInputs:
    """Map an Euclidean projection candidate to DROT's cost convention.

    DROT solves ``<C,T> + (m+n) * r_weight / 2 * ||T||²`` over the transport
    polytope.  For ``s = max(candidate) - min(candidate)`` and
    ``C = (max(candidate) - candidate) / s``, setting
    ``r_weight = 1 / ((m+n) * s)`` produces the same optimizer as
    ``1/2 * ||T - candidate||²``.  The constant-candidate case is handled by
    the projector because the scaling is undefined there.
    """
    if candidate.ndim != 2:
        raise ValueError("candidate must be a matrix")
    if row_marginal.ndim != 1 or col_marginal.ndim != 1:
        raise ValueError("marginals must be vectors")
    if candidate.shape != (row_marginal.numel(), col_marginal.numel()):
        raise ValueError("candidate and marginal dimensions do not match")
    if not all(
        torch.isfinite(value).all().item()
        for value in (candidate, row_marginal, col_marginal)
    ):
        raise ValueError("candidate and marginals must be finite")
    if torch.any(row_marginal <= 0) or torch.any(col_marginal <= 0):
        raise ValueError("marginals must be positive")
    if not torch.isclose(
        row_marginal.sum(), col_marginal.sum(), rtol=1e-5, atol=1e-7
    ):
        raise ValueError("row and column marginals must have equal mass")

    maximum = candidate.max()
    minimum = candidate.min()
    scale_tensor = maximum - minimum
    scale = float(scale_tensor.item())
    if scale <= 0.0:
        raise ValueError("constant candidates have no DROT cost scaling")

    cost = (maximum - candidate) / scale_tensor
    regularization_weight = 1.0 / (candidate.shape[0] + candidate.shape[1]) / scale
    return DrotInputs(
        cost=cost,
        row_marginal=row_marginal,
        col_marginal=col_marginal,
        scale=scale,
        regularization_weight=regularization_weight,
    )


def _diagnostics_for_status(
    status: str,
    plan: torch.Tensor,
    iterations: int = 0,
) -> ProjectionDiagnostics:
    minimum = float(plan.min().item()) if plan.numel() else float("nan")
    return ProjectionDiagnostics(
        status,
        iterations,
        minimum,
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
    )


def _load_reg_drot() -> Any:
    """Load the pinned extension from the REBUILD third-party directory.

    The extension is built against the active PyTorch environment and is not
    installed as a system package. Importing ``torch`` above loads its shared
    libraries; adding only the pinned extension directory keeps the loader
    explicit without modifying the third-party source or global driver links.
    """
    try:
        return importlib.import_module("reg_drot")
    except (ImportError, OSError) as first_error:
        extension_dir = Path(__file__).resolve().parent / "Regularized-DROT" / "pytorch"
        if not extension_dir.is_dir():
            raise first_error
        extension_path = str(extension_dir)
        sys.path.insert(0, extension_path)
        try:
            return importlib.import_module("reg_drot")
        finally:
            if sys.path and sys.path[0] == extension_path:
                sys.path.pop(0)


class DrotEuclideanProjector(TransportProjector):
    """Regularized-DROT backend for the REBUILD Euclidean projection contract."""

    def __init__(
        self,
        rho: float = 4.0,
        max_iterations: int = 20_000,
        solver_eps: float = 1e-6,
        relative_tolerance: float = 1e-3,
        absolute_tolerance: float = 1e-8,
        backend: Optional[Any] = None,
    ):
        if rho <= 0:
            raise ValueError("rho must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if solver_eps <= 0 or relative_tolerance <= 0 or absolute_tolerance <= 0:
            raise ValueError("solver tolerances must be positive")
        # The upstream example uses rho=2 and eps=1e-4 for feasibility-only
        # demonstrations.  REBUILD validated rho=4 and eps=1e-6 against an
        # independent Euclidean projection reference, including active-set
        # (boundary) solutions.  Keep both values configurable for audits.
        self.rho = float(rho)
        self.max_iterations = int(max_iterations)
        self.solver_eps = float(solver_eps)
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self._backend = backend
        self._backend_error = None
        if backend is None:
            try:
                self._backend = _load_reg_drot()
            except (ImportError, OSError) as error:
                self._backend_error = error

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def backend_error(self) -> Optional[BaseException]:
        return self._backend_error

    def _invalid_result(self, candidate: torch.Tensor, status: str) -> ProjectionResult:
        return ProjectionResult(
            candidate.detach().clone(),
            _diagnostics_for_status(status, candidate.detach()),
        )

    def project(
        self,
        candidate: torch.Tensor,
        row_marginal: torch.Tensor,
        col_marginal: torch.Tensor,
    ) -> ProjectionResult:
        if candidate.ndim != 2:
            return self._invalid_result(candidate, "invalid_input")
        if candidate.shape != (row_marginal.numel(), col_marginal.numel()):
            return self._invalid_result(candidate, "invalid_input")
        if not all(
            torch.isfinite(value).all().item()
            for value in (candidate, row_marginal, col_marginal)
        ):
            return self._invalid_result(candidate, "numerical_failure")
        if torch.any(row_marginal <= 0) or torch.any(col_marginal <= 0):
            return self._invalid_result(candidate, "invalid_input")
        if not torch.isclose(
            row_marginal.sum(), col_marginal.sum(), rtol=1e-5, atol=1e-7
        ):
            return self._invalid_result(candidate, "invalid_input")

        # A constant candidate has a closed-form projection and does not need
        # the optional extension.  Near-constant candidates still go through
        # DROT; they are never silently replaced by the uniform plan.
        scale = candidate.max() - candidate.min()
        if scale.item() == 0.0:
            total_mass = row_marginal.sum()
            plan = row_marginal[:, None] * col_marginal[None, :] / total_mass
            diagnostics = validate_transport_plan(
                plan,
                row_marginal,
                col_marginal,
                atol=self.absolute_tolerance,
                rtol=self.relative_tolerance,
            )
            return ProjectionResult(plan, diagnostics)

        if self._backend is None:
            return self._invalid_result(candidate, "not_verified")
        if candidate.dtype != torch.float32:
            return self._invalid_result(candidate, "invalid_input")
        if candidate.device.type != "cuda":
            return self._invalid_result(candidate, "invalid_input")
        if row_marginal.device != candidate.device or col_marginal.device != candidate.device:
            return self._invalid_result(candidate, "invalid_input")

        try:
            inputs = prepare_drot_inputs(candidate, row_marginal, col_marginal)
            # REBUILD's eps is a relative marginal tolerance in this explicit
            # adapter path. The upstream API remains available unchanged.
            plan = self._backend.quadratic_drot_relative(
                inputs.cost,
                inputs.col_marginal,
                inputs.row_marginal,
                self.rho,
                inputs.regularization_weight,
                self.max_iterations,
                self.solver_eps,
            )
        except (RuntimeError, ValueError, TypeError) as error:
            self._backend_error = error
            return self._invalid_result(candidate, "numerical_failure")

        if not isinstance(plan, torch.Tensor) or plan.shape != candidate.shape:
            return self._invalid_result(candidate, "infeasible_output")
        diagnostics = validate_transport_plan(
            plan,
            row_marginal,
            col_marginal,
            atol=self.absolute_tolerance,
            rtol=self.relative_tolerance,
        )
        return ProjectionResult(plan, diagnostics)
