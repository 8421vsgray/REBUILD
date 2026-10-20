import unittest

import torch

from solver.drot_adapter import DrotEuclideanProjector, prepare_drot_inputs


class _FakeDrot:
    def __init__(self):
        self.calls = []

    def quadratic_drot(self, cost, p, q, rho, r_weight, max_iter, eps):
        self.calls.append((cost, p, q, rho, r_weight, max_iter, eps))
        return q[:, None] * p[None, :]

    def quadratic_drot_relative(self, cost, p, q, rho, r_weight, max_iter, eps):
        return self.quadratic_drot(cost, p, q, rho, r_weight, max_iter, eps)


class DrotAdapterTests(unittest.TestCase):
    def test_candidate_mapping_matches_square_euclidean_scaling(self):
        candidate = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        marginal = torch.full((2,), 0.5, dtype=torch.float32)
        inputs = prepare_drot_inputs(candidate, marginal, marginal)
        self.assertTrue(torch.allclose(inputs.cost, torch.tensor([[0.0, 1.0], [1.0, 0.5]])))
        self.assertAlmostEqual(inputs.scale, 2.0)
        self.assertAlmostEqual(inputs.regularization_weight, 0.125)

    def test_constant_candidate_has_closed_form_projection(self):
        candidate = torch.full((3, 3), 7.0, dtype=torch.float32)
        marginal = torch.full((3,), 1.0 / 3.0)
        result = DrotEuclideanProjector(backend=_FakeDrot()).project(
            candidate, marginal, marginal
        )
        self.assertEqual(result.diagnostics.status, "success")
        self.assertTrue(torch.allclose(result.plan, torch.full((3, 3), 1.0 / 9.0)))

    def test_missing_backend_is_not_verified(self):
        candidate = torch.tensor([[1.0, 0.0], [0.0, 0.5]], dtype=torch.float32)
        marginal = torch.full((2,), 0.5, dtype=torch.float32)
        projector = DrotEuclideanProjector(backend=object())
        projector._backend = None
        result = projector.project(candidate, marginal, marginal)
        self.assertEqual(result.diagnostics.status, "not_verified")

    def test_fake_backend_output_is_revalidated(self):
        if not torch.cuda.is_available():
            self.skipTest("DROT adapter backend contract test requires CUDA")
        backend = _FakeDrot()
        candidate = torch.tensor(
            [[1.0, 0.0], [0.0, 0.5]], dtype=torch.float32, device="cuda"
        )
        marginal = torch.full((2,), 0.5, dtype=torch.float32, device="cuda")
        result = DrotEuclideanProjector(backend=backend).project(
            candidate, marginal, marginal
        )
        self.assertEqual(result.diagnostics.status, "success")
        self.assertEqual(len(backend.calls), 1)
        self.assertAlmostEqual(backend.calls[0][4], 0.25)


if __name__ == "__main__":
    unittest.main()
