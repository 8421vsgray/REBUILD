import unittest

import torch

from model import CorreGenBaselineModel, OursModel


class NamingRefactorTests(unittest.TestCase):
    @staticmethod
    def make_model():
        return CorreGenBaselineModel(
            n_views=2,
            layer_dims=[[3, 5, 2], [3, 5, 2]],
            temperature=0.5,
            n_classes=2,
            drop_rate=0.0,
            beta=0.15,
            reg=0.05,
            rho=0.2,
        )

    def test_legacy_class_name_is_a_compatibility_alias(self):
        self.assertIs(OursModel, CorreGenBaselineModel)

    def test_renamed_class_preserves_state_dict_and_forward_contract(self):
        torch.manual_seed(123)
        named_model = self.make_model().eval()
        torch.manual_seed(123)
        legacy_model = OursModel(
            n_views=2,
            layer_dims=[[3, 5, 2], [3, 5, 2]],
            temperature=0.5,
            n_classes=2,
            drop_rate=0.0,
            beta=0.15,
            reg=0.05,
            rho=0.2,
        ).eval()

        self.assertEqual(list(named_model.state_dict()), list(legacy_model.state_dict()))
        for key, value in named_model.state_dict().items():
            self.assertTrue(torch.equal(value, legacy_model.state_dict()[key]))

        torch.manual_seed(456)
        samples = [torch.randn(4, 3), torch.randn(4, 3)]
        named_features = named_model.forward_features(samples)
        legacy_features = legacy_model.forward_features(samples)
        for named, legacy in zip(named_features, legacy_features):
            torch.testing.assert_close(named, legacy, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
