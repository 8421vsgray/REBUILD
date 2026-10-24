import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader, Dataset

import main_train
import engine_train
from engine_train import train_one_epoch
from model import CorreGenBaselineModel
from solver.fastnewton_adapter import FastNewtonEuclideanProjector


class TinyViews(Dataset):
    def __init__(self, size=7):
        generator = torch.Generator().manual_seed(20261020)
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


class TrainingIntegrationTests(unittest.TestCase):
    @staticmethod
    def _stride_args(start_rectify_epoch=0):
        return SimpleNamespace(
            algorithm="stride_gw",
            print_this_epoch=False,
            print_freq=1,
            start_rectify_epoch=start_rectify_epoch,
            momentum=0.98,
            rho=0.2,
            n_views=2,
            batch_size=7,
            start_guide_epoch=100,
            init="diagonal",
            steps=3,
            step_size=0.05,
            solver={"backend": "fastnewton", "relative_tol": 1.0e-3, "max_iter": 50},
        )

    @staticmethod
    def _tiny_model(drop_rate=0.2):
        return CorreGenBaselineModel(
            n_views=2,
            layer_dims=[[3, 16, 16, 4], [3, 16, 16, 4]],
            temperature=0.5,
            n_classes=2,
            drop_rate=drop_rate,
        )

    def test_stride_engine_reuses_one_training_forward_and_keeps_train_mode(self):
        torch.manual_seed(20261020)
        model = self._tiny_model(drop_rate=0.2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        loader = DataLoader(TinyViews(), batch_size=7, shuffle=False)
        args = self._stride_args(start_rectify_epoch=0)
        events = []
        counts = {"online0": 0, "online1": 0, "decoder0": 0, "decoder1": 0,
                  "target0": 0, "target1": 0}
        train_modes = {name: [] for name in counts}
        observed = {}

        def hook_for(name):
            def hook(module, inputs, output):
                counts[name] += 1
                train_modes[name].append(module.training)
                events.append(name)
            return hook

        hooks = []
        for index in range(2):
            hooks.append(model.online_encoder[index].register_forward_hook(
                hook_for("online{}".format(index))))
            hooks.append(model.cross_view_decoder[index].register_forward_hook(
                hook_for("decoder{}".format(index))))
            hooks.append(model.target_encoder[index].register_forward_hook(
                hook_for("target{}".format(index))))

        original_update = model._update_target_branch

        def tracked_update(momentum):
            events.append("ema")
            return original_update(momentum)

        model._update_target_branch = tracked_update
        original_loss = model.loss_from_training_features

        def tracked_loss(z, p, z_t, inter_targets):
            observed["loss_p_ids"] = [id(item) for item in p]
            observed["loss_target_ids"] = [id(item) for item in z_t]
            return original_loss(z, p, z_t, inter_targets)

        model.loss_from_training_features = tracked_loss

        def fake_generator(predictions, target_embeddings, gw_config, projector):
            observed["gw_p_ids"] = [id(item) for item in predictions]
            observed["gw_target_ids"] = [id(item) for item in target_embeddings]
            batch_size = predictions[0].shape[0]
            identity = torch.eye(
                batch_size, dtype=predictions[0].dtype, device=predictions[0].device
            )
            diagnostics = {
                "P12": {"non_diagonal_mass": 0.0},
                "P21": {"non_diagonal_mass": 0.0},
            }
            return {"P12": identity, "P21": identity.clone()}, diagnostics

        try:
            with mock.patch.object(
                engine_train,
                "generate_bidirectional_gw_targets",
                side_effect=fake_generator,
            ):
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
        finally:
            for hook in hooks:
                hook.remove()

        for name, count in counts.items():
            self.assertEqual(count, 1, name)
            self.assertEqual(train_modes[name], [True], name)
        self.assertEqual(events[0], "ema")
        self.assertLess(events.index("ema"), events.index("online0"))
        self.assertEqual(observed["gw_p_ids"], observed["loss_p_ids"])
        self.assertEqual(observed["gw_target_ids"], observed["loss_target_ids"])
        self.assertTrue(all(torch.isfinite(value).all() for value in model.parameters()))

    def test_stride_warmup_uses_identity_without_gw_generator(self):
        torch.manual_seed(20261020)
        model = self._tiny_model(drop_rate=0.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        loader = DataLoader(TinyViews(), batch_size=7, shuffle=False)
        args = self._stride_args(start_rectify_epoch=1)
        with mock.patch.object(
            engine_train,
            "generate_bidirectional_gw_targets",
            side_effect=AssertionError("GW generator must not run during diagonal warmup"),
        ):
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

    def test_fastnewton_yaml_is_accepted_and_drot_fields_are_rejected(self):
        config = main_train.load_config_file(
            main_train.resolve_config_path("config/scene15.yaml")
        )
        self.assertEqual(config["solver"]["backend"], "fastnewton")
        with self.assertRaisesRegex(ValueError, "drot.*removed"):
            main_train._validate_standard_config(
                {**config, "solver": {"backend": "drot", "rho": 2.0, "eps": 1e-6, "max_iter": 100}}
            )
        mixed = copy.deepcopy(config)
        mixed["solver"]["rho"] = 2.0
        with self.assertRaises(ValueError):
            main_train._validate_standard_config(mixed)

    def test_prior_gate_lock_and_lockmass_controls(self):
        config = main_train.load_config_file(
            main_train.resolve_config_path("config/scene15.yaml")
        )
        config = copy.deepcopy(config)
        config["prior_gate"] = {
            "lock": False,
            "lockmass": 2.0e-3,
        }
        main_train._validate_standard_config(config)

        invalid_lock = copy.deepcopy(config)
        invalid_lock["prior_gate"]["lock"] = "false"
        with self.assertRaises(ValueError):
            main_train._validate_standard_config(invalid_lock)

        invalid_mass = copy.deepcopy(config)
        invalid_mass["prior_gate"]["lockmass"] = -1.0
        with self.assertRaises(ValueError):
            main_train._validate_standard_config(invalid_mass)

    def test_stride_engine_generates_gw_targets_and_updates_model(self):
        torch.manual_seed(20261020)
        model = CorreGenBaselineModel(
            n_views=2,
            layer_dims=[[3, 16, 16, 4], [3, 16, 16, 4]],
            temperature=0.5,
            n_classes=2,
            drop_rate=0.0,
        )
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        projector = FastNewtonEuclideanProjector(
            relative_tolerance=1.0e-4, max_iterations=50
        )
        if not projector.available:
            self.skipTest(f"FASTNEWTON is not installed: {projector.backend_error}")
        args = self._stride_args(start_rectify_epoch=0)
        loader = DataLoader(TinyViews(), batch_size=7, shuffle=False)
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
            projector=projector,
        )
        self.assertEqual(projector.last_info.get("status"), "verified")
        self.assertTrue(
            any(
                not torch.equal(before[name], value)
                for name, value in model.state_dict().items()
            )
        )
        self.assertTrue(
            all(torch.isfinite(value).all() for value in model.parameters())
        )


if __name__ == "__main__":
    unittest.main()
