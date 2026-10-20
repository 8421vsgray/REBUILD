import unittest

import torch

from solver.reference_projection import DykstraEuclideanProjector
from solver.projection_interface import validate_transport_plan


class ReferenceProjectionTests(unittest.TestCase):
    def test_diagonal_candidate_projects_to_feasible_transport_plan(self):
        candidate = torch.eye(2, dtype=torch.float64)
        marginal = torch.full((2,), 0.5, dtype=torch.float64)
        result = DykstraEuclideanProjector().project(candidate, marginal, marginal)

        self.assertEqual(result.diagnostics.status, "success")
        torch.testing.assert_close(
            result.plan,
            torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float64),
            rtol=0.0,
            atol=1e-8,
        )

    def test_random_projection_has_nonnegative_entries_and_target_marginals(self):
        torch.manual_seed(123)
        candidate = torch.randn(8, 8, dtype=torch.float64)
        marginal = torch.full((8,), 1.0 / 8.0, dtype=torch.float64)
        result = DykstraEuclideanProjector().project(candidate, marginal, marginal)

        self.assertEqual(result.diagnostics.status, "success")
        self.assertGreaterEqual(float(result.plan.min()), -1e-10)
        self.assertLessEqual(result.diagnostics.row_relative_residual, 1e-8)
        self.assertLessEqual(result.diagnostics.col_relative_residual, 1e-8)

    def test_constant_candidate_returns_uniform_plan(self):
        candidate = torch.full((4, 4), 7.0, dtype=torch.float64)
        marginal = torch.full((4,), 0.25, dtype=torch.float64)
        result = DykstraEuclideanProjector().project(candidate, marginal, marginal)

        expected = torch.full((4, 4), 1.0 / 16.0, dtype=torch.float64)
        self.assertEqual(result.diagnostics.status, "success")
        torch.testing.assert_close(result.plan, expected, rtol=0.0, atol=1e-8)

    def test_validator_rejects_infeasible_plan(self):
        plan = torch.zeros(2, 2, dtype=torch.float64)
        marginal = torch.full((2,), 0.5, dtype=torch.float64)
        diagnostics = validate_transport_plan(plan, marginal, marginal)

        self.assertEqual(diagnostics.status, "infeasible_output")
        self.assertGreater(diagnostics.row_relative_residual, 0.0)


if __name__ == "__main__":
    unittest.main()
