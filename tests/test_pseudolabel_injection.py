import copy
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import model as model_module
from model import CorreGenBaselineModel, MaxLikelihoodLoss


class ExternalPseudoLabelTests(unittest.TestCase):
    @staticmethod
    def make_model():
        model = CorreGenBaselineModel(
            n_views=2,
            layer_dims=[[4, 8, 3], [4, 8, 3]],
            temperature=0.7,
            n_classes=3,
            drop_rate=0.0,
        )
        model.eval()
        return model

    @staticmethod
    def make_targets(batch_size):
        p12 = torch.eye(batch_size, dtype=torch.float32)
        p12 = torch.roll(p12, shifts=1, dims=1)
        return {"P12": p12, "P21": p12.t().contiguous()}

    @staticmethod
    def make_batch(batch_size):
        torch.manual_seed(17)
        return [torch.randn(batch_size, 4), torch.randn(batch_size, 4)]

    def test_external_targets_change_loss_and_online_gradients(self):
        batch_size = 7
        data = self.make_batch(batch_size)
        diagonal = {"P12": torch.eye(batch_size), "P21": torch.eye(batch_size)}
        permuted = self.make_targets(batch_size)
        training_state = {"mmt": 1.0, "dist": None}
        training_phase = {"rectify": True}

        model_diag = self.make_model()
        model_perm = copy.deepcopy(model_diag)
        with mock.patch.object(
            model_diag,
            "_cross_view_loss",
            wraps=model_diag._cross_view_loss,
        ) as cross_loss:
            loss_diag = model_diag(
                data, training_state, training_phase, external_targets=diagonal
            )
        loss_perm = model_perm(
            data, training_state, training_phase, external_targets=permuted
        )

        self.assertEqual(cross_loss.call_count, 1)
        used_targets = cross_loss.call_args.args[2]
        self.assertTrue(torch.equal(used_targets[0], diagonal["P12"]))
        self.assertTrue(torch.equal(used_targets[1], diagonal["P21"]))
        self.assertFalse(torch.allclose(loss_diag, loss_perm))

        loss_diag.backward()
        loss_perm.backward()
        diag_grads = [
            parameter.grad.detach().clone()
            for parameter in model_diag.online_encoder.parameters()
            if parameter.grad is not None
        ]
        perm_grads = [
            parameter.grad.detach().clone()
            for parameter in model_perm.online_encoder.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(diag_grads)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in diag_grads))
        self.assertTrue(all(torch.isfinite(grad).all() for grad in perm_grads))
        self.assertTrue(any(not torch.allclose(left, right) for left, right in zip(diag_grads, perm_grads)))
        self.assertTrue(
            all(parameter.grad is None for parameter in model_diag.target_encoder.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model_perm.target_encoder.parameters())
        )

    def test_external_target_loss_matches_independent_reference(self):
        torch.manual_seed(23)
        batch_size = 7
        temperature = 0.7
        predictions = [torch.randn(batch_size, 3), torch.randn(batch_size, 3)]
        target_embeddings = [torch.randn(batch_size, 3), torch.randn(batch_size, 3)]
        targets = self.make_targets(batch_size)
        loss_module = MaxLikelihoodLoss(temperature)
        model = self.make_model()

        actual = model._cross_view_loss(
            predictions, target_embeddings, [targets["P12"], targets["P21"]]
        )

        def reference(x_q, x_k, target):
            x_q = F.normalize(x_q, dim=1)
            x_k = F.normalize(x_k, dim=1)
            similarity = (x_q @ x_k.t()) / temperature
            probability = torch.softmax(similarity.reshape(-1), dim=0) * batch_size
            return (-torch.log(probability.reshape(batch_size, batch_size)) * target).mean()

        expected = (
            reference(predictions[0], target_embeddings[1], targets["P12"])
            + reference(predictions[1], target_embeddings[0], targets["P21"])
        ) / 2
        direct = (
            loss_module(predictions[0], target_embeddings[1], targets["P12"])
            + loss_module(predictions[1], target_embeddings[0], targets["P21"])
        ) / 2
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(actual, direct, atol=1e-7, rtol=1e-6))

    def test_external_mode_does_not_call_partial_ot(self):
        batch_size = 7
        model = self.make_model()
        data = self.make_batch(batch_size)
        targets = self.make_targets(batch_size)
        training_state = {"mmt": 1.0, "dist": None}
        training_phase = {"rectify": False}

        def fake_intra(cost, reg):
            return torch.eye(cost.shape[0], dtype=cost.dtype, device=cost.device)

        with mock.patch.object(model_module, "Partial_ot", side_effect=AssertionError("Partial_ot called")), mock.patch.object(
            model_module, "entropic_wasserstein", side_effect=fake_intra
        ):
            loss = model(
                data, training_state, training_phase, external_targets=targets
            )
        self.assertTrue(torch.isfinite(loss))

    def test_none_mode_retains_original_partial_ot_path(self):
        batch_size = 7
        model = self.make_model()
        data = self.make_batch(batch_size)
        training_state = {"mmt": 1.0, "dist": None}
        training_phase = {"rectify": False}

        def fake_transport(cost, reg, rho):
            return torch.eye(cost.shape[0], dtype=cost.dtype, device=cost.device)

        def fake_intra(cost, reg):
            return torch.eye(cost.shape[0], dtype=cost.dtype, device=cost.device)

        with mock.patch.object(
            model_module, "Partial_ot", side_effect=fake_transport
        ) as partial_ot, mock.patch.object(
            model_module, "entropic_wasserstein", side_effect=fake_intra
        ):
            loss = model(data, training_state, training_phase)
        self.assertEqual(partial_ot.call_count, 2)
        self.assertTrue(torch.isfinite(loss))

    def test_validation_rejects_malformed_external_targets(self):
        model = self.make_model()
        device = torch.device("cpu")
        dtype = torch.float32
        valid = self.make_targets(7)

        cases = [
            ({"P12": valid["P12"]}, "exactly P12 and P21"),
            ({"P12": torch.eye(6), "P21": torch.eye(6)}, "shape"),
            ({"P12": valid["P12"].clone(), "P21": valid["P21"]}, ""),
        ]
        cases[2][0]["P12"][0, 0] = float("nan")
        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaises((TypeError, ValueError)):
                    model._validate_external_targets(value, 7, device, dtype)

        negative = self.make_targets(7)
        negative["P12"][0, 0] = -0.1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            model._validate_external_targets(negative, 7, device, dtype)

        bad_sum = self.make_targets(7)
        bad_sum["P12"][0] = 0.0
        with self.assertRaisesRegex(ValueError, "rows must sum"):
            model._validate_external_targets(bad_sum, 7, device, dtype)

        requires_grad = self.make_targets(7)
        requires_grad["P12"].requires_grad_()
        with self.assertRaisesRegex(ValueError, "gradients"):
            model._validate_external_targets(requires_grad, 7, device, dtype)

        with self.assertRaisesRegex(ValueError, "dtype"):
            model._validate_external_targets(valid, 7, device, torch.float64)
        with self.assertRaisesRegex(ValueError, "must be on"):
            model._validate_external_targets(valid, 7, torch.device("cuda"), dtype)

    def test_validation_supports_actual_batch_sizes(self):
        model = self.make_model()
        for batch_size in (7, 32, 512):
            with self.subTest(batch_size=batch_size):
                targets = self.make_targets(batch_size)
                p12, p21 = model._validate_external_targets(
                    targets, batch_size, torch.device("cpu"), torch.float32
                )
                self.assertEqual(tuple(p12.shape), (batch_size, batch_size))
                self.assertTrue(torch.allclose(p12.sum(dim=1), torch.ones(batch_size)))
                self.assertTrue(torch.allclose(p21, p12.t()))


if __name__ == "__main__":
    unittest.main()
