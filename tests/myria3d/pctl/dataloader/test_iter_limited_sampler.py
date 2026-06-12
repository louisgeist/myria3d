import pytest
import torch
from torch.utils.data import Dataset

from myria3d.pctl.dataloader.iter_limited_sampler import IterLimitedSampler


class _DummyDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


@pytest.mark.parametrize("dataset_size", [100, 10])
def test_iter_limited_sampler_length(dataset_size):
    dataset = _DummyDataset(dataset_size)
    sampler = IterLimitedSampler(dataset, iter_per_epoch=5, batch_size=4, seed=0)
    assert len(sampler) == 5 * 4


def test_iter_limited_sampler_without_replacement():
    dataset = _DummyDataset(100)
    sampler = IterLimitedSampler(dataset, iter_per_epoch=5, batch_size=4, seed=0)
    indices = list(sampler)
    assert len(indices) == 20
    assert len(set(indices)) == 20
    assert all(0 <= i < 100 for i in indices)


def test_iter_limited_sampler_with_replacement():
    dataset = _DummyDataset(10)
    sampler = IterLimitedSampler(dataset, iter_per_epoch=5, batch_size=4, seed=0)
    assert sampler.needs_replacement is True
    indices = list(sampler)
    assert len(indices) == 20
    assert len(set(indices)) < 20


def test_iter_limited_sampler_set_epoch_changes_indices():
    dataset = _DummyDataset(100)
    sampler = IterLimitedSampler(dataset, iter_per_epoch=5, batch_size=4, seed=0)
    sampler.set_epoch(0)
    indices_epoch_0 = list(sampler)
    sampler.set_epoch(1)
    indices_epoch_1 = list(sampler)
    assert indices_epoch_0 != indices_epoch_1


def test_iter_limited_sampler_same_epoch_is_reproducible():
    dataset = _DummyDataset(100)
    sampler_a = IterLimitedSampler(dataset, iter_per_epoch=5, batch_size=4, seed=42)
    sampler_b = IterLimitedSampler(dataset, iter_per_epoch=5, batch_size=4, seed=42)
    sampler_a.set_epoch(3)
    sampler_b.set_epoch(3)
    assert list(sampler_a) == list(sampler_b)
