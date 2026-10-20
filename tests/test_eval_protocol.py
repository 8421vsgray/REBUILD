import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from main_train import build_data_loaders, resolve_batch_sizes


class IndexDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


def make_args(batch_size=4, eval_batch_size=None):
    return SimpleNamespace(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=0,
        pin_mem=False,
    )


class EvaluationProtocolTests(unittest.TestCase):
    def test_default_evaluation_batch_size_follows_training_configuration(self):
        args = make_args(batch_size=4)
        train_loader, eval_loader = build_data_loaders(
            IndexDataset(9), IndexDataset(11), args
        )

        self.assertEqual(train_loader.batch_size, 4)
        self.assertEqual(eval_loader.batch_size, 4)
        self.assertEqual(args.eval_batch_size, 4)

    def test_explicit_evaluation_batch_size_is_independent(self):
        args = make_args(batch_size=8, eval_batch_size=3)
        train_loader, eval_loader = build_data_loaders(
            IndexDataset(9), IndexDataset(11), args
        )

        self.assertEqual(train_loader.batch_size, 8)
        self.assertEqual(eval_loader.batch_size, 3)
        self.assertEqual(args.eval_batch_size, 3)

    def test_evaluation_random_sampler_covers_each_sample_once(self):
        args = make_args(batch_size=8, eval_batch_size=3)
        _, eval_loader = build_data_loaders(
            IndexDataset(9), IndexDataset(11), args
        )
        observed = torch.cat([batch for batch in eval_loader]).tolist()

        self.assertEqual(len(observed), 11)
        self.assertEqual(sorted(observed), list(range(11)))

    def test_batch_sizes_are_capped_without_dropping_samples(self):
        args = make_args(batch_size=32, eval_batch_size=64)
        train_batch_size, eval_batch_size = resolve_batch_sizes(args, 5, 7)

        self.assertEqual(train_batch_size, 5)
        self.assertEqual(eval_batch_size, 7)

    def test_non_positive_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_batch_sizes(make_args(batch_size=0), 5, 7)
        with self.assertRaises(ValueError):
            resolve_batch_sizes(make_args(batch_size=4, eval_batch_size=0), 5, 7)


if __name__ == "__main__":
    unittest.main()
