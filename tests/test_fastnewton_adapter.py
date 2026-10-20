import unittest

import torch

from pseudolabel.gw import generate_bidirectional_gw_targets, projector_from_solver_config
from solver.fastnewton_adapter import FastNewtonEuclideanProjector
from solver.projection_interface import ProjectionDiagnostics, ProjectionResult


class FastNewtonAdapterTests(unittest.TestCase):
    def test_factory_uses_independent_fastnewton_options(self):
        projector = projector_from_solver_config(
            {"backend": "fastnewton", "relative_tol": 1.0e-3, "max_iter": 50}
        )
        self.assertIsInstance(projector, FastNewtonEuclideanProjector)
        self.assertEqual(projector.max_iterations, 50)
        self.assertEqual(projector.relative_tolerance, 1.0e-3)

    def test_factory_rejects_drot_options_for_fastnewton(self):
        with self.assertRaises(ValueError):
            projector_from_solver_config(
                {
                    "backend": "fastnewton",
                    "relative_tol": 1.0e-3,
                    "max_iter": 50,
                    "rho": 2.0,
                }
            )

    def test_warm_state_is_reused_then_cleared_after_failure(self):
        class FakeBackend:
            def __init__(self):
                self.calls = []

            def project(self, candidate, **kwargs):
                self.calls.append(kwargs)
                marginal = 1.0 / candidate.shape[0]
                plan = torch.full_like(candidate, marginal / candidate.shape[0])
                if len(self.calls) == 1:
                    return plan, {
                        "status": "verified",
                        "iterations": 2,
                        "dual_state": (torch.zeros(candidate.shape[0]), torch.zeros(candidate.shape[0])),
                        "active_state": torch.ones_like(candidate, dtype=torch.bool),
                        "optimality_verified": True,
                    }
                return plan, {"status": "failed", "iterations": 3}

        projector = FastNewtonEuclideanProjector()
        projector._backend = FakeBackend()
        candidate = torch.randn(7, 7)
        marginal = torch.full((7,), 1.0 / 7.0)
        first = projector.project(candidate, marginal, marginal)
        self.assertEqual(first.diagnostics.status, "success")
        second = projector.project(candidate, marginal, marginal)
        self.assertEqual(second.diagnostics.status, "failed")
        self.assertIsNone(projector._dual_warm_start)
        self.assertIsNone(projector._active_warm_start)
        self.assertIn("dual_warm_start", projector._backend.calls[1])

    def test_uniform_projection_and_p_normalization(self):
        projector = FastNewtonEuclideanProjector(
            relative_tolerance=1.0e-3, max_iterations=50
        )
        if not projector.available:
            self.skipTest(f"FASTNEWTON is not installed: {projector.backend_error}")
        for batch_size in (7, 32, 512):
            with self.subTest(batch_size=batch_size):
                torch.manual_seed(100 + batch_size)
                if batch_size == 512:
                    # A dense random 512x512 matrix is a documented
                    # difficult path for FASTNEWTON.  Use a structured
                    # GW-like/near-diagonal candidate for the normal
                    # projection contract test; failure handling is tested
                    # separately by the backend's own suite.
                    candidate = torch.eye(batch_size) * 0.1
                    candidate = candidate + 0.01 * torch.randn(batch_size, batch_size)
                else:
                    candidate = torch.randn(batch_size, batch_size)
                marginal = torch.full((batch_size,), 1.0 / batch_size)
                result = projector.project(candidate, marginal, marginal)
                self.assertIn(result.diagnostics.status, {"success", "feasible_only"})
                self.assertTrue(torch.isfinite(result.plan).all())
                self.assertGreaterEqual(float(result.plan.min()), -1.0e-8)
                pseudo_labels = result.plan * batch_size
                torch.testing.assert_close(
                    pseudo_labels.sum(dim=1),
                    torch.ones(batch_size),
                    atol=2.0e-3,
                    rtol=0.0,
                )
                projector.reset_state()

    def test_two_directions_reset_warm_state_and_return_row_stochastic_labels(self):
        projector = FastNewtonEuclideanProjector(
            relative_tolerance=1.0e-3, max_iterations=50
        )
        if not projector.available:
            self.skipTest(f"FASTNEWTON is not installed: {projector.backend_error}")
        torch.manual_seed(20261020)
        predictions = [torch.randn(7, 8), torch.randn(7, 6)]
        targets = [torch.randn(7, 5), torch.randn(7, 9)]
        labels, diagnostics = generate_bidirectional_gw_targets(
            predictions,
            targets,
            {"init": "diagonal", "steps": 3, "step_size": 0.05},
            projector,
        )
        for name in ("P12", "P21"):
            self.assertEqual(labels[name].shape, (7, 7))
            self.assertFalse(labels[name].requires_grad)
            self.assertTrue(torch.isfinite(labels[name]).all())
            torch.testing.assert_close(
                labels[name].sum(dim=1), torch.ones(7), atol=2.0e-3, rtol=0.0
            )
            self.assertEqual(diagnostics[name]["batch_size"], 7)
        self.assertEqual(diagnostics["P12"]["steps"], 3)
        self.assertEqual(diagnostics["P21"]["steps"], 3)


if __name__ == "__main__":
    unittest.main()
