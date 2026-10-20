import os
import unittest
from types import SimpleNamespace
import warnings

import numpy as np

from dataset_loader import load_dataset, load_mat


class DataProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
        if not os.path.exists(os.path.join(cls.data_path, "Scene_15.mat")):
            raise unittest.SkipTest("Scene15 fixture is not available")

    @staticmethod
    def make_args(c_ratio=0.0):
        return SimpleNamespace(
            dataset="Scene15",
            data_path=DataProtocolTests.data_path,
            data_norm="standard",
            n_views=2,
            m_ratio=0.2,
            c_ratio=c_ratio,
        )

    def test_mismatch_construction_is_repeatable_and_indexed(self):
        args = self.make_args()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = load_mat(args)
            second = load_mat(args)

        data, _, labels1, labels2, id1, id2, aligned, _, _ = first
        data_again, _, labels1_again, labels2_again, id1_again, id2_again, aligned_again, _, _ = second
        sample_count = data[0].shape[0]
        selected_count = int(args.m_ratio * sample_count)

        self.assertEqual([view.shape for view in data], [(sample_count, 59), (sample_count, 20)])
        self.assertTrue(all(np.array_equal(left, right) for left, right in zip(data, data_again)))
        self.assertTrue(np.array_equal(labels1, labels1_again))
        self.assertTrue(np.array_equal(labels2, labels2_again))
        self.assertTrue(np.array_equal(id1, id1_again))
        self.assertTrue(np.array_equal(id2, id2_again))
        self.assertTrue(np.array_equal(aligned, aligned_again))
        self.assertEqual(int((aligned == 0).sum()), selected_count)
        self.assertTrue(np.array_equal(id1, np.arange(sample_count)))
        self.assertEqual(len(np.unique(id2)), sample_count)

    def test_cr_changes_training_features_but_not_clean_evaluation_features(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean_train, clean_test = load_dataset(self.make_args(c_ratio=0.0))
            noisy_train, noisy_test = load_dataset(self.make_args(c_ratio=0.1))

        self.assertEqual(len(clean_train), len(clean_test))
        self.assertEqual(len(noisy_train), len(noisy_test))
        self.assertTrue(all(np.array_equal(clean_train.data[i], clean_test.data[i]) for i in range(2)))
        self.assertFalse(all(np.array_equal(noisy_train.data[i], noisy_test.data[i]) for i in range(2)))
        sample = noisy_train[0]
        self.assertEqual(sample[0], 0)
        self.assertEqual(sample[2].item(), int(noisy_train.label1[0]))
        self.assertEqual(sample[3].item(), int(noisy_train.label2[0]))


if __name__ == "__main__":
    unittest.main()
