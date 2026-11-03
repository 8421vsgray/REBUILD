import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.utils.data import DataLoader, Dataset

import engine_train
from engine_train import train_one_epoch
from model import CorreGenBaselineModel


class TinyViews(Dataset):
    def __init__(self, size=21):
        generator = torch.Generator().manual_seed(20261103)
        self.views = [
            torch.randn(size, 3, generator=generator),
            torch.randn(size, 3, generator=generator),
        ]

    def __len__(self):
        return self.views[0].shape[0]

    def __getitem__(self, index):
        label = torch.tensor(index % 2, dtype=torch.long)
        return (
            torch.tensor(index, dtype=torch.long),
            [self.views[0][index], self.views[1][index]],
            label,
            label,
            torch.tensor(index, dtype=torch.long),
            torch.tensor(index, dtype=torch.long),
            torch.tensor(True),
        )


class PriorGateFallbackTests(unittest.TestCase):
    def test_gate_uses_recovered_mass_not_failure_alone(self):
        torch.manual_seed(20261103)
        model = CorreGenBaselineModel(
            n_views=2,
            layer_dims=[[3, 16, 16, 4], [3, 16, 16, 4]],
            temperature=0.5,
            n_classes=2,
            drop_rate=0.0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        loader = DataLoader(TinyViews(size=21), batch_size=7, shuffle=False)
        args = SimpleNamespace(
            algorithm="stride_gw",
            print_this_epoch=False,
            print_freq=1,
            start_rectify_epoch=0,
            momentum=0.98,
            rho=0.2,
            n_views=2,
            batch_size=7,
            start_guide_epoch=100,
            init="diagonal",
            steps=3,
            step_size=0.05,
            solver={"backend": "fastnewton", "relative_tol": 1.0e-3, "max_iter": 50},
            prior_gate={"lock": True, "lockmass": 0.2},
        )

        identity = torch.eye(7)
        shifted = torch.roll(identity, shifts=1, dims=1)
        diffused = 0.7 * identity + 0.3 * shifted
        recovered_low_diffusion = 0.9 * identity + 0.1 * shifted

        def generator_result(plan, projection_failed=False):
            diagnostics = {
                "non_diagonal_mass": 1.0 - float(plan.diagonal().mean()),
                "prior_gate": False,
                "projection_failed": projection_failed,
                "failed_step": 1 if projection_failed else None,
                "last_valid_step": 0 if projection_failed else 2,
            }
            return (
                {"P12": plan.clone(), "P21": plan.clone()},
                {"P12": diagnostics.copy(), "P21": diagnostics.copy()},
            )

        captured_targets = []
        original_loss = model.loss_from_training_features

        def capture_loss(z, p, z_t, inter_targets):
            captured_targets.append(
                [target.detach().clone() for target in inter_targets]
            )
            return original_loss(z, p, z_t, inter_targets)

        model.loss_from_training_features = capture_loss
        with mock.patch.object(
            engine_train,
            "generate_bidirectional_gw_targets",
            side_effect=[
                generator_result(diffused),
                generator_result(diffused, projection_failed=True),
                generator_result(recovered_low_diffusion, projection_failed=True),
            ],
        ) as generate_targets:
            train_one_epoch(
                model,
                loader,
                loader,
                optimizer,
                torch.device("cpu"),
                None,
                0,
                scheduler=None,
                args=args,
                projector=object(),
            )

        self.assertEqual(len(captured_targets), 3)
        torch.testing.assert_close(captured_targets[0][0], diffused)
        torch.testing.assert_close(captured_targets[0][1], diffused)
        # A solver failure alone does not lock if the returned fallback diffused enough.
        torch.testing.assert_close(captured_targets[1][0], diffused)
        torch.testing.assert_close(captured_targets[1][1], diffused)
        # The last-valid fallback is gated only when its measured mass is low.
        torch.testing.assert_close(captured_targets[2][0], identity)
        torch.testing.assert_close(captured_targets[2][1], identity)
        self.assertEqual(generate_targets.call_count, 3)
        for call in generate_targets.call_args_list:
            self.assertNotIn("prior_gate_precheck", call.args[2])


if __name__ == "__main__":
    unittest.main()
