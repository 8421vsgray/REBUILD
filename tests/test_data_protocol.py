import os
import unittest
from types import SimpleNamespace
import warnings
import tempfile

import numpy as np

from dataset_loader import load_dataset, load_mat
from main_train import build_run_output_dir, load_config_file


class DataProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
        if not os.path.exists(os.path.join(cls.data_path, "Scene_15.mat")):
            raise unittest.SkipTest("Scene15 fixture is not available")

    @staticmethod
    def make_args(m_ratio=0.2, view_indices=None, encoder_dim=None):
        args = SimpleNamespace(
            dataset="Scene15",
            data_path=DataProtocolTests.data_path,
            data_norm="standard",
            n_views=2,
            m_ratio=m_ratio,
        )
        if view_indices is not None:
            args.view_indices = view_indices
        if encoder_dim is not None:
            args.encoder_dim = encoder_dim
        return args

    def test_mismatch_construction_is_repeatable_and_indexed(self):
        args = self.make_args()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = load_mat(args)
            second = load_mat(args)

        data, labels1, labels2, id1, id2, aligned, _, _ = first
        data_again, labels1_again, labels2_again, id1_again, id2_again, aligned_again, _, _ = second
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

    def test_zero_mismatch_keeps_anchor_and_identity_alignment(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zero = load_mat(self.make_args(m_ratio=0.0))
            mismatch = load_mat(self.make_args(m_ratio=0.2))

        data, labels1, labels2, id1, id2, aligned, _, _ = zero
        mismatch_data, mismatch_labels1, _, mismatch_id1, _, _, _, _ = mismatch
        self.assertTrue(np.array_equal(data[0], mismatch_data[0]))
        self.assertTrue(np.array_equal(labels1, mismatch_labels1))
        self.assertTrue(np.array_equal(id1, id2))
        self.assertTrue(np.array_equal(id1, mismatch_id1))
        self.assertTrue(np.array_equal(labels1, labels2))
        self.assertTrue(np.all(aligned == 1))

    def test_train_and_evaluation_share_the_same_mr_data(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_dataset, test_dataset = load_dataset(self.make_args())

        self.assertEqual(len(train_dataset), len(test_dataset))
        self.assertTrue(all(np.array_equal(train_dataset.data[i], test_dataset.data[i]) for i in range(2)))
        self.assertTrue(np.array_equal(train_dataset.id2, test_dataset.id2))
        self.assertTrue(np.array_equal(train_dataset.label2, test_dataset.label2))

    def test_explicit_view_indices_preserve_scene15_anchor_order(self):
        args = self.make_args(view_indices=[1, 0])
        data, _, _, _, _, _, _, _ = load_mat(args)
        self.assertEqual([view.shape for view in data], [(4485, 59), (4485, 20)])

    def test_encoder_input_dimensions_are_checked(self):
        args = self.make_args(
            view_indices=[1, 0],
            encoder_dim=[[58, 1024], [20, 1024]],
        )
        with self.assertRaisesRegex(ValueError, "encoder_dim input dimensions"):
            load_dataset(args)

    def test_removed_corruption_ratio_config_is_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config:
            config.write("c_ratio: 0\nm_ratio: 0.2\n")
            config_path = config.name
        try:
            with self.assertRaisesRegex(ValueError, "c_ratio"):
                load_config_file(config_path)
        finally:
            os.unlink(config_path)

    def test_legacy_train_time_config_is_explicitly_mapped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config:
            config.write("train_time: 7\nm_ratio: 0.2\n")
            config_path = config.name
        try:
            parsed = load_config_file(config_path)
            self.assertEqual(parsed["num_runs"], 7)
            self.assertNotIn("train_time", parsed)
        finally:
            os.unlink(config_path)

    def test_legacy_and_new_run_count_names_cannot_conflict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config:
            config.write("train_time: 7\nnum_runs: 5\n")
            config_path = config.name
        try:
            with self.assertRaisesRegex(ValueError, "cannot both be set"):
                load_config_file(config_path)
        finally:
            os.unlink(config_path)

    def test_unknown_configuration_fields_are_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config:
            config.write("m_ratio: 0.2\nmisspelled_batch_size: 512\n")
            config_path = config.name
        try:
            with self.assertRaisesRegex(ValueError, "Unknown configuration field"):
                load_config_file(config_path)
        finally:
            os.unlink(config_path)

    def test_standard_scene15_configuration_is_validated(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "scene15.yaml",
        )
        parsed = load_config_file(config_path)
        self.assertEqual(parsed["view_indices"], [1, 0])
        self.assertEqual(parsed["num_runs"], 5)
        self.assertEqual(parsed["start_rectify_epoch"], 20)
        self.assertEqual(parsed["solver"]["backend"], "fastnewton")

    def test_run_output_directory_is_seeded_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as output_dir:
            first = build_run_output_dir(output_dir, "run_fixed", 1)
            first.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                build_run_output_dir(output_dir, "run_fixed", 1)
            self.assertTrue(str(first).endswith(os.path.join("run_fixed", "seed_1")))


if __name__ == "__main__":
    unittest.main()
