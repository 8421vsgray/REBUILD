"""FASTNEWTON adapter for REBUILD's Euclidean transport interface."""

from __future__ import annotations

import importlib
from typing import Any, Optional

import torch

from .projection_interface import (
    ProjectionDiagnostics,
    ProjectionResult,
    TransportProjector,
    validate_transport_plan,
)


def _diagnostics_for_status(
    status: str,
    plan: torch.Tensor,
    *,
    iterations: int = 0,
    backend_status: str = "",
    optimality_verified: bool = False,
) -> ProjectionDiagnostics:
    minimum = float(plan.min().item()) if plan.numel() else float("nan")
    return ProjectionDiagnostics(
        status,
        int(iterations),
        minimum,
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        backend_status,
        bool(optimality_verified),
    )


class FastNewtonEuclideanProjector(TransportProjector):
    """Project uniform square candidates with the installed FASTNEWTON API."""

    def __init__(
        self,
        *,
        relative_tolerance: float = 1.0e-3,
        max_iterations: int = 50,
        absolute_tolerance: float = 1.0e-8,
        certificate_tolerance: Optional[float] = None,
        warm_start: bool = True,
        solver_options: Optional[dict[str, Any]] = None,
    ) -> None:
        if relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if absolute_tolerance <= 0.0:
            raise ValueError("absolute_tolerance must be positive")
        if certificate_tolerance is not None and certificate_tolerance <= 0.0:
            raise ValueError("certificate_tolerance must be positive")
        self.relative_tolerance = float(relative_tolerance)
        self.max_iterations = int(max_iterations)
        self.absolute_tolerance = float(absolute_tolerance)
        self.certificate_tolerance = (
            None if certificate_tolerance is None else float(certificate_tolerance)
        )
        if not isinstance(warm_start, bool):
            raise ValueError("warm_start must be boolean")
        self.warm_start = warm_start
        self.solver_options = dict(solver_options or {})
        self._backend = None
        self._backend_error: Optional[BaseException] = None
        self._dual_warm_start = None
        self._active_warm_start = None
        self._last_info: dict[str, Any] = {}
        try:
            # FASTNEWTON is installed from solver/FASTNEWTON via pyproject.toml.
            # No source-tree sys.path injection is used here.
            self._backend = importlib.import_module("fastnewton")
        except (ImportError, OSError) as error:
            self._backend_error = error

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def backend_error(self) -> Optional[BaseException]:
        return self._backend_error

    @property
    def last_info(self) -> dict[str, Any]:
        return dict(self._last_info)

    def reset_state(self) -> None:
        """Clear warm state before a new independent direction or batch."""
        self._dual_warm_start = None
        self._active_warm_start = None
        self._last_info = {}

    def _invalid_result(self, candidate: torch.Tensor, status: str) -> ProjectionResult:
        self.reset_state()
        if not torch.is_tensor(candidate):
            candidate = torch.empty(0)
        return ProjectionResult(
            candidate.detach().clone(),
            _diagnostics_for_status(status, candidate.detach()),
        )

    @staticmethod
    def _is_uniform_marginal(marginal: torch.Tensor, size: int) -> bool:
        target = marginal.new_full(marginal.shape, 1.0 / float(size))
        return bool(torch.allclose(marginal, target, atol=1.0e-6, rtol=1.0e-4))

    @staticmethod
    def _clone_state(value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            return tuple(item.detach().clone() for item in value)
        if torch.is_tensor(value):
            return value.detach().clone()
        return None

    def project(
        self,
        candidate: torch.Tensor,
        row_marginal: torch.Tensor,
        col_marginal: torch.Tensor,
    ) -> ProjectionResult:
        if not torch.is_tensor(candidate) or candidate.ndim != 2:
            return self._invalid_result(candidate, "invalid_input")
        if candidate.shape[0] != candidate.shape[1]:
            return self._invalid_result(candidate, "invalid_input")
        size = candidate.shape[0]
        if row_marginal.ndim != 1 or col_marginal.ndim != 1:
            return self._invalid_result(candidate, "invalid_input")
        if candidate.shape != (row_marginal.numel(), col_marginal.numel()):
            return self._invalid_result(candidate, "invalid_input")
        if not all(
            torch.isfinite(value).all().item()
            for value in (candidate, row_marginal, col_marginal)
        ):
            self.reset_state()
            return self._invalid_result(candidate, "numerical_failure")
        if not self._is_uniform_marginal(row_marginal, size) or not self._is_uniform_marginal(
            col_marginal, size
        ):
            return self._invalid_result(candidate, "invalid_input")
        if candidate.device != row_marginal.device or candidate.device != col_marginal.device:
            return self._invalid_result(candidate, "invalid_input")
        if candidate.dtype != row_marginal.dtype or candidate.dtype != col_marginal.dtype:
            return self._invalid_result(candidate, "invalid_input")
        if self._backend is None:
            return self._invalid_result(candidate, "not_verified")

        options: dict[str, Any] = {
            "relative_tol": self.relative_tolerance,
            "max_iter": self.max_iterations,
            "return_info": True,
            **self.solver_options,
        }
        if self.certificate_tolerance is not None:
            options["certificate_tolerance"] = self.certificate_tolerance
        if self.warm_start and self._dual_warm_start is not None:
            options["dual_warm_start"] = self._dual_warm_start
        if self.warm_start and self._active_warm_start is not None:
            options["active_warm_start"] = self._active_warm_start

        try:
            plan, info = self._backend.project(candidate, **options)
        except Exception as error:
            self.reset_state()
            self._backend_error = error
            return self._invalid_result(candidate, "numerical_failure")

        info = dict(info)
        self._last_info = info
        backend_status = str(info.get("status", "unknown"))
        iterations = int(info.get("iterations", 0))
        if not isinstance(plan, torch.Tensor) or plan.shape != candidate.shape:
            self.reset_state()
            self._last_info = info
            return ProjectionResult(
                candidate.detach().clone(),
                _diagnostics_for_status(
                    "infeasible_output",
                    candidate.detach(),
                    iterations=iterations,
                    backend_status=backend_status,
                ),
            )
        plan = plan.detach().to(device=candidate.device, dtype=candidate.dtype)
        feasibility = validate_transport_plan(
            plan,
            row_marginal,
            col_marginal,
            atol=self.absolute_tolerance,
            rtol=self.relative_tolerance,
        )
        if backend_status not in {"verified", "feasible_only"} or feasibility.status != "success":
            self.reset_state()
            self._last_info = info
            return ProjectionResult(
                plan,
                _diagnostics_for_status(
                    "failed",
                    plan,
                    iterations=iterations,
                    backend_status=backend_status,
                ),
            )

        # Retain state only after a feasible result when explicitly enabled.
        # The GW caller resets it between P12 and P21 so the two directions
        # cannot contaminate one another.
        if self.warm_start:
            self._dual_warm_start = self._clone_state(info.get("dual_state"))
            self._active_warm_start = self._clone_state(info.get("active_state"))
        else:
            self._dual_warm_start = None
            self._active_warm_start = None
        optimality_verified = bool(info.get("optimality_verified", False))
        status = "success" if backend_status == "verified" else "feasible_only"
        return ProjectionResult(
            plan,
            ProjectionDiagnostics(
                status,
                iterations,
                feasibility.min_value,
                feasibility.row_abs_residual,
                feasibility.col_abs_residual,
                feasibility.row_relative_residual,
                feasibility.col_relative_residual,
                backend_status,
                optimality_verified,
            ),
        )
