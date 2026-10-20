import copy
from types import SimpleNamespace
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

import main_train
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
    def test_fastnewton_yaml_is_accepted_and_drot_fields_are_rejected(self):
        config = main_train.load_config_file(
            main_train.resolve_config_path("config/Scene15_MR08_FASTNEWTON.yaml")
        )
        self.assertEqual(config["solver"]["backend"], "fastnewton")
        mixed = copy.deepcopy(config)
        mixed["solver"]["rho"] = 2.0
        with self.assertRaises(ValueError):
            main_train._validate_standard_config(mixed)

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
            solver={"backend": "fastnewton", "relative_tol": 1.0e-4, "max_iter": 50},
        )
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
