import os

import numpy as np
import torch
from torch_geometric.transforms import Center, GridSampling

from myria3d.pctl.datamodule.downstream import DownstreamNpyDatamodule
from myria3d.pctl.transforms.transforms import (
    MaximumNumNodes,
    MinimumNumNodes,
    NullifyLowestZ,
    SubtileCrop,
)


def _write_scene(root, split, name, n=400):
    scene_dir = os.path.join(root, split, name)
    os.makedirs(scene_dir, exist_ok=True)
    pos = (np.random.rand(n, 3) * 100).astype(np.float32)
    np.save(os.path.join(scene_dir, "coord.npy"), pos)
    np.save(
        os.path.join(scene_dir, "strength.npy"),
        np.random.rand(n).astype(np.float32),
    )
    np.save(
        os.path.join(scene_dir, "segment.npy"),
        np.random.randint(0, 5, size=(n,)).astype(np.int64),
    )
    return scene_dir


def _transforms_dict():
    train_prep = [
        SubtileCrop(tile_width=100, subtile_width=50, random=True, min_points=1),
        GridSampling(0.5),
        MinimumNumNodes(4),
        MaximumNumNodes(4000),
        Center(),
        NullifyLowestZ(),
    ]
    eval_prep = [
        SubtileCrop(tile_width=100, subtile_width=50, random=False),
        GridSampling(0.5),
        MinimumNumNodes(4),
        MaximumNumNodes(4000),
        Center(),
        NullifyLowestZ(),
    ]
    return {
        "preparations_train_list": train_prep,
        "preparations_eval_list": eval_prep,
        "augmentations_list": [],
        "normalizations_list": [],
    }


def test_dales_datamodule_end_to_end_batches(tmp_path):
    _write_scene(str(tmp_path), "train", "tile_a")
    _write_scene(str(tmp_path), "train", "tile_b")
    _write_scene(str(tmp_path), "test", "tile_c")

    dm = DownstreamNpyDatamodule(
        data_root=str(tmp_path),
        dataset_target="myria3d.pctl.dataset.downstream.dales.DalesDataset",
        train_dir="train",
        val_dir="test",  # DALES: val == test.
        test_dir="test",
        tile_width=100,
        subtile_width=50,
        batch_size=2,
        num_workers=1,
        transforms=_transforms_dict(),
    )
    dm.setup()

    train_batch = next(iter(dm.train_dataloader()))
    assert train_batch is not None
    assert train_batch.x.shape[1] == 5
    # DALES has no color: all points get the all-True color_mask.
    assert torch.equal(train_batch.color_mask, torch.ones_like(train_batch.color_mask))

    test_loader = dm.test_dataloader()
    # 1 test scene * 4 subtiles (100m / 50m) worth of dataset entries.
    assert len(dm.test_dataset) == 4
    test_batch = next(iter(test_loader))
    assert test_batch is not None


def _write_labeled_scene(root, split, name, segment, n=400):
    scene_dir = _write_scene(root, split, name, n=n)
    np.save(os.path.join(scene_dir, "segment.npy"), np.asarray(segment, dtype=np.int32))
    return scene_dir


def test_opengf_datamodule_applies_dataset_kwargs_per_split(tmp_path):
    rng = np.random.default_rng(0)
    seg_with_outliers = rng.integers(0, 3, size=400)
    assert (seg_with_outliers == 2).any()
    for split in ("train", "val"):
        _write_labeled_scene(str(tmp_path), split, "S1_1_0-0", seg_with_outliers)
    _write_labeled_scene(str(tmp_path), "test", "T1_0-0", seg_with_outliers)
    _write_labeled_scene(str(tmp_path), "test", "T2_0-0", seg_with_outliers)

    dm = DownstreamNpyDatamodule(
        data_root=str(tmp_path),
        dataset_target="myria3d.pctl.dataset.downstream.opengf.OpenGFDataset",
        tile_width=100,
        subtile_width=50,
        batch_size=1,
        num_workers=0,
        transforms=_transforms_dict(),
        train_dataset_kwargs={"remap_labels": {2: 1}},
        val_dataset_kwargs={"remap_labels": {2: 1}},
        test_dataset_kwargs={"include_names": "T2", "drop_labels": [2]},
    )
    dm.setup()

    # Train / val: outliers merged into Non-ground -> only labels {0, 1} remain.
    for dataset in (dm.train_dataset, dm.val_dataset):
        labels = set(dataset[0].y.tolist()) if dataset[0] is not None else set()
        assert labels <= {0, 1}
    # Test: T2 only (4 mosaic subtiles for a single 100 m / 50 m tile), outliers deleted.
    assert {os.path.basename(d) for d, _ in dm.test_dataset.entries} == {"T2_0-0"}
    assert len(dm.test_dataset) == 4
    test_labels = set()
    for i in range(len(dm.test_dataset)):
        item = dm.test_dataset[i]
        if item is not None:
            test_labels |= set(item.y.tolist())
    assert test_labels <= {0, 1}
    assert dm.test_dataset.drop_labels == [2]
    assert dm.train_dataset.drop_labels is None


def test_common_dataset_kwargs_still_apply_to_every_split(tmp_path):
    _write_scene(str(tmp_path), "train", "a")
    _write_scene(str(tmp_path), "test", "b")

    dm = DownstreamNpyDatamodule(
        data_root=str(tmp_path),
        dataset_target="myria3d.pctl.dataset.downstream.dales.DalesDataset",
        val_dir="test",
        tile_width=100,
        subtile_width=50,
        num_workers=0,
        transforms=_transforms_dict(),
        dataset_kwargs={"label_key": "segment"},
    )

    for dataset in (dm.train_dataset, dm.val_dataset, dm.test_dataset):
        assert dataset.label_key == "segment"
