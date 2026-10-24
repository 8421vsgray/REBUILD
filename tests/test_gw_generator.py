import unittest

import numpy as np
import torch

from pseudolabel.gw import (
    GWProjectionFailure,
    generate_bidirectional_gw_targets,
    generate_gw_targets,
    gw_gradient,
    pointwise_gradient,
    projector_from_solver_config,
    relation_matrix,
)
from solver.drot_adapter import DrotEuclideanProjector
from solver.projection_interface import (
    ProjectionDiagnostics,
    ProjectionResult,
    TransportProjector,
)

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    cp = None
    HAS_CVXPY = False

try:
    import reg_drot  # noqa: F401

    HAS_DROT = True
except (ImportError, OSError):
    HAS_DROT = False


class RecordingProjector(TransportProjector):
    def __init__(self):
        self.candidates = []

    def project(self, candidate, row_marginal, col_marginal):
        self.candidates.append(candidate.detach().clone())
        plan = row_marginal[:, None] * col_marginal[None, :]
        plan = plan / row_marginal.sum()
        diagnostics = ProjectionDiagnostics(
            "success", 1, float(plan.min()), 0.0, 0.0, 0.0, 0.0
        )
        return ProjectionResult(plan, diagnostics)


class FailingProjector(TransportProjector):
    def project(self, candidate, row_marginal, col_marginal):
        diagnostics = ProjectionDiagnostics(
            "not_verified", 1, 0.0, 1.0, 1.0, 1.0, 1.0
        )
        return ProjectionResult(candidate.detach(), diagnostics)


class GWGeneratorTests(unittest.TestCase):
    @staticmethod
    def config(steps=1, step_size=0.05):
        return {"init": "diagonal", "steps": steps, "step_size": step_size}

    def test_relation_and_gradient_match_autograd_reference(self):
        torch.manual_seed(11)
        source = torch.randn(5, 4, dtype=torch.float64)
        target = torch.randn(5, 3, dtype=torch.float64)
        source_relation = relation_matrix(source)
        target_relation = relation_matrix(target)
        transport = torch.rand(5, 5, dtype=torch.float64, requires_grad=True)
        objective = -2.0 * torch.trace(
            source_relation @ transport @ target_relation.t() @ transport.t()
        )
        autograd_gradient = torch.autograd.grad(objective, transport)[0]
        analytic_gradient = gw_gradient(
            source_relation, target_relation, transport.detach()
        )
        torch.testing.assert_close(
            analytic_gradient, autograd_gradient, rtol=1e-10, atol=1e-10
        )

    def test_zero_step_initializes_to_diagonal_and_scales_to_p(self):
        source = torch.randn(7, 4)
        target = torch.randn(7, 3)
        pseudo_labels, diagnostics = generate_gw_targets(
            source, target, self.config(steps=0), RecordingProjector()
        )
        torch.testing.assert_close(pseudo_labels, torch.eye(7), atol=1e-7, rtol=0.0)
        self.assertFalse(pseudo_labels.requires_grad)
        self.assertEqual(diagnostics["status"], "success")
        self.assertEqual(diagnostics["batch_size"], 7)
        self.assertEqual(diagnostics["optimality"], "unverified")

    def test_bidirectional_targets_are_independent(self):
        torch.manual_seed(13)
        predictions = [torch.randn(7, 4), torch.randn(7, 5)]
        target_embeddings = [torch.randn(7, 6), torch.randn(7, 3)]
        targets, diagnostics = generate_bidirectional_gw_targets(
            predictions,
            target_embeddings,
            self.config(steps=1),
            RecordingProjector(),
        )
        self.assertEqual(set(targets), {"P12", "P21"})
        self.assertEqual(set(diagnostics), {"P12", "P21"})
        for name in ("P12", "P21"):
            self.assertTrue(torch.isfinite(targets[name]).all())
            self.assertGreaterEqual(float(targets[name].min()), 0.0)
            torch.testing.assert_close(
                targets[name].sum(dim=1), torch.ones(7), atol=1e-5, rtol=1e-4
            )
            self.assertFalse(targets[name].requires_grad)

    def test_bidirectional_pairs_use_the_intended_views(self):
        class PairRecordingProjector(TransportProjector):
            def __init__(self):
                self.candidates = []

            def project(self, candidate, row_marginal, col_marginal):
                self.candidates.append(candidate.detach().clone())
                plan = row_marginal[:, None] * col_marginal[None, :]
                plan = plan / row_marginal.sum()
                diagnostics = ProjectionDiagnostics(
                    "success", 1, float(plan.min()), 0.0, 0.0, 0.0, 0.0
                )
                return ProjectionResult(plan, diagnostics)

        torch.manual_seed(41)
        predictions = [torch.randn(7, 4), torch.randn(7, 5)]
        target_embeddings = [torch.randn(7, 6), torch.randn(7, 3)]
        projector = PairRecordingProjector()
        generate_bidirectional_gw_targets(
            predictions,
            target_embeddings,
            self.config(steps=1),
            projector,
        )
        expected_p12 = relation_matrix(predictions[0])
        expected_z1 = relation_matrix(target_embeddings[1])
        expected_p21 = relation_matrix(predictions[1])
        expected_z0 = relation_matrix(target_embeddings[0])
        initial = torch.eye(7) / 7.0
        expected_candidate_p12 = initial - 0.05 * gw_gradient(
            expected_p12, expected_z1, initial
        )
        expected_candidate_p21 = initial - 0.05 * gw_gradient(
            expected_p21, expected_z0, initial
        )
        torch.testing.assert_close(
            projector.candidates[0], expected_candidate_p12, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            projector.candidates[1], expected_candidate_p21, atol=1e-5, rtol=1e-5
        )

    def test_steps_and_step_size_reach_projector(self):
        source = torch.randn(7, 4)
        target = torch.randn(7, 4)
        first = RecordingProjector()
        second = RecordingProjector()
        generate_gw_targets(source, target, self.config(steps=2, step_size=0.01), first)
        generate_gw_targets(source, target, self.config(steps=3, step_size=0.07), second)
        self.assertEqual(len(first.candidates), 2)
        self.assertEqual(len(second.candidates), 3)
        self.assertFalse(torch.allclose(first.candidates[0], second.candidates[0]))

    def test_pointwise_gradient_is_used_as_fixed_projection_direction(self):
        torch.manual_seed(47)
        source = torch.randn(7, 4, dtype=torch.float64)
        target = torch.randn(7, 4, dtype=torch.float64)
        projector = RecordingProjector()
        pseudo_labels, diagnostics = generate_gw_targets(
            source,
            target,
            {
                "init": "diagonal",
                "steps": 1,
                "step_size": 0.05,
                "transport_gradient": "pointwise",
            },
            projector,
        )
        expected = torch.eye(7, dtype=torch.float64) / 7.0
        expected = expected - 0.05 * pointwise_gradient(source, target)
        torch.testing.assert_close(
            projector.candidates[0], expected, atol=1e-10, rtol=1e-10
        )
        self.assertEqual(diagnostics["transport_gradient"], "pointwise")
        self.assertFalse(pseudo_labels.requires_grad)

    def test_pointwise_gradient_rejects_mismatched_feature_dimensions(self):
        with self.assertRaises(ValueError):
            pointwise_gradient(torch.randn(7, 4), torch.randn(7, 5))

    def test_unknown_transport_gradient_fails_explicitly(self):
        with self.assertRaises(ValueError):
            generate_gw_targets(
                torch.randn(7, 4),
                torch.randn(7, 4),
                {
                    "init": "diagonal",
                    "steps": 1,
                    "step_size": 0.05,
                    "transport_gradient": "unknown",
                },
                RecordingProjector(),
            )

    def test_solver_parameters_are_explicitly_constructed(self):
        projector = projector_from_solver_config(
            {"backend": "drot", "rho": 3.0, "eps": 1e-7, "max_iter": 1234}
        )
        self.assertIsInstance(projector, DrotEuclideanProjector)
        self.assertEqual(projector.rho, 3.0)
        self.assertEqual(projector.solver_eps, 1e-7)
        self.assertEqual(projector.max_iterations, 1234)

    def test_projection_failure_is_not_replaced(self):
        with self.assertRaises(GWProjectionFailure):
            generate_gw_targets(
                torch.randn(7, 4),
                torch.randn(7, 4),
                self.config(steps=1),
                FailingProjector(),
            )

    def test_batch_sizes_do_not_assume_512(self):
        for batch_size in (7, 32, 512):
            with self.subTest(batch_size=batch_size):
                pseudo_labels, diagnostics = generate_gw_targets(
                    torch.randn(batch_size, 4),
                    torch.randn(batch_size, 5),
                    self.config(steps=0),
                    RecordingProjector(),
                )
                self.assertEqual(tuple(pseudo_labels.shape), (batch_size, batch_size))
                self.assertEqual(diagnostics["batch_size"], batch_size)

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            generate_gw_targets(
                torch.randn(7, 4), torch.randn(6, 4), self.config(), RecordingProjector()
            )
        bad_source = torch.randn(7, 4)
        bad_source[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            generate_gw_targets(
                bad_source, torch.randn(7, 4), self.config(), RecordingProjector()
            )
        with self.assertRaises(ValueError):
            generate_gw_targets(
                torch.randn(7, 4),
                torch.randn(7, 4),
                {"init": "identity", "steps": 1, "step_size": 0.05},
                RecordingProjector(),
            )

    @unittest.skipUnless(HAS_CVXPY, "cvxpy is not installed")
    def test_cvxpy_reference_is_available_at_small_sizes(self):
        torch.manual_seed(19)
        for batch_size in (2, 4, 8, 16, 32):
            with self.subTest(batch_size=batch_size):
                candidate = torch.randn(batch_size, batch_size, dtype=torch.float64)
                marginal = torch.full(
                    (batch_size,), 1.0 / batch_size, dtype=torch.float64
                )
                variable = cp.Variable((batch_size, batch_size), nonneg=True)
                candidate_np = candidate.numpy()
                constraints = [
                    variable @ np.ones(batch_size) == np.full(batch_size, 1.0 / batch_size),
                    variable.T @ np.ones(batch_size) == np.full(batch_size, 1.0 / batch_size),
                ]
                problem = cp.Problem(
                    cp.Minimize(0.5 * cp.sum_squares(variable - candidate_np)),
                    constraints,
                )
                problem.solve(
                    solver=cp.CLARABEL,
                    tol_gap_abs=1e-9,
                    tol_gap_rel=1e-9,
                    tol_feas=1e-9,
                )
                self.assertEqual(problem.status, cp.OPTIMAL)
                expected = torch.from_numpy(variable.value)
                self.assertEqual(tuple(expected.shape), (batch_size, batch_size))
                self.assertGreaterEqual(float(expected.min()), -1e-7)
                torch.testing.assert_close(
                    expected.sum(dim=1), marginal, atol=1e-7, rtol=1e-7
                )

    @unittest.skipUnless(
        HAS_DROT and torch.cuda.is_available(), "DROT CUDA extension is unavailable"
    )
    def test_drot_projection_matches_cvxpy_at_small_sizes(self):
        torch.manual_seed(29)
        projector = DrotEuclideanProjector(
            rho=4.0, solver_eps=1e-6, max_iterations=20_000
        )
        for batch_size in (2, 4, 8, 16, 32):
            with self.subTest(batch_size=batch_size):
                candidate = torch.randn(batch_size, batch_size, device="cuda")
                marginal = torch.full(
                    (batch_size,), 1.0 / batch_size, device="cuda"
                )
                result = projector.project(candidate, marginal, marginal)
                self.assertEqual(result.diagnostics.status, "success")
                variable = cp.Variable((batch_size, batch_size), nonneg=True)
                candidate_np = candidate.cpu().numpy()
                vector = np.full(batch_size, 1.0 / batch_size)
                problem = cp.Problem(
                    cp.Minimize(0.5 * cp.sum_squares(variable - candidate_np)),
                    [variable @ np.ones(batch_size) == vector, variable.T @ np.ones(batch_size) == vector],
                )
                problem.solve(
                    solver=cp.CLARABEL,
                    tol_gap_abs=1e-9,
                    tol_gap_rel=1e-9,
                    tol_feas=1e-9,
                )
                self.assertEqual(problem.status, cp.OPTIMAL)
                expected = torch.from_numpy(variable.value)
                self.assertLessEqual(
                    float((result.plan.cpu() - expected).abs().max()), 1e-3
                )


if __name__ == "__main__":
    unittest.main()
