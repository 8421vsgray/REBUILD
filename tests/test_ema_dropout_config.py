import unittest
from types import SimpleNamespace

import torch
from torch import nn

import main_train
from model import CorreGenBaselineModel


class EmaDropoutConfigTests(unittest.TestCase):
    @staticmethod
    def make_model(*, drop_rate=0.4, ema_dropout=None, target_drop_rate=None):
        return CorreGenBaselineModel(
            n_views=2,
            layer_dims=[[4, 8, 8, 2], [4, 8, 8, 2]],
            temperature=0.5,
            n_classes=2,
            drop_rate=drop_rate,
            ema_dropout=ema_dropout,
            target_drop_rate=target_drop_rate,
        )

    @staticmethod
    def dropout_rates(module):
        return [layer.p for layer in module.modules() if isinstance(layer, nn.Dropout)]

    def test_missing_ema_dropout_inherits_online_rate(self):
        model = self.make_model(drop_rate=0.4)
        self.assertEqual(self.dropout_rates(model.online_encoder), [0.4, 0.4])
        self.assertEqual(self.dropout_rates(model.target_encoder), [0.4, 0.4])
        self.assertEqual(model.ema_dropout, 0.4)

    def test_explicit_zero_disables_only_ema_dropout(self):
        model = self.make_model(drop_rate=0.4, ema_dropout=0.0)
        self.assertEqual(self.dropout_rates(model.online_encoder), [0.4, 0.4])
        self.assertEqual(self.dropout_rates(model.target_encoder), [0.0, 0.0])

    def test_explicit_value_overrides_ema_branch(self):
        model = self.make_model(drop_rate=0.2, ema_dropout=0.6)
        self.assertEqual(self.dropout_rates(model.online_encoder), [0.2, 0.2])
        self.assertEqual(self.dropout_rates(model.target_encoder), [0.6, 0.6])

    def test_legacy_target_drop_rate_alias_is_supported(self):
        model = self.make_model(drop_rate=0.4, target_drop_rate=0.1)
        self.assertEqual(model.ema_dropout, 0.1)
        self.assertEqual(model.target_drop_rate, 0.1)
        with self.assertRaisesRegex(ValueError, "cannot disagree"):
            self.make_model(drop_rate=0.4, ema_dropout=0.0, target_drop_rate=0.1)

    def test_finalize_config_inherits_and_accepts_explicit_zero(self):
        inherited = SimpleNamespace(
            ema_dropout=None, target_drop_rate=None, drop_rate=0.3
        )
        main_train.finalize_ema_dropout(inherited)
        self.assertEqual(inherited.ema_dropout, 0.3)

        explicit_zero = SimpleNamespace(
            ema_dropout=0.0, target_drop_rate=None, drop_rate=0.3
        )
        main_train.finalize_ema_dropout(explicit_zero)
        self.assertEqual(explicit_zero.ema_dropout, 0.0)

    def test_finalize_config_accepts_legacy_alias_and_rejects_conflicts(self):
        legacy = SimpleNamespace(
            ema_dropout=None, target_drop_rate=0.1, drop_rate=0.3
        )
        main_train.finalize_ema_dropout(legacy)
        self.assertEqual(legacy.ema_dropout, 0.1)
        self.assertIsNone(legacy.target_drop_rate)

        conflict = SimpleNamespace(
            ema_dropout=0.0, target_drop_rate=0.1, drop_rate=0.3
        )
        with self.assertRaisesRegex(ValueError, "cannot disagree"):
            main_train.finalize_ema_dropout(conflict)

    def test_invalid_ema_dropout_is_rejected(self):
        for value in (-0.01, 1.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "ema_dropout"):
                    self.make_model(drop_rate=0.3, ema_dropout=value)

    def test_canonical_yaml_configs_all_use_max_iter_200(self):
        config_dir = main_train.Path(__file__).resolve().parents[1] / "config"
        expected = {"scene15.yaml", "handwritten.yaml", "ccv20.yaml", "food101.yaml", "reuters.yaml"}
        actual = {path.name for path in config_dir.glob("*.yaml")}
        self.assertEqual(actual, expected)
        for name in sorted(expected):
            with self.subTest(config=name):
                config = main_train.load_config_file(config_dir / name)
                self.assertEqual(config["solver"]["backend"], "fastnewton")
                self.assertEqual(config["solver"]["max_iter"], 200)
                self.assertEqual(config["num_runs"], 5)


if __name__ == "__main__":
    unittest.main()
