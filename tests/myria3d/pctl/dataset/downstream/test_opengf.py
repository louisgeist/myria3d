import os

import numpy as np
import torch

from myria3d.pctl.dataset.downstream.opengf import OpenGFDataset


def _write_scene(root, split, name, segment=None, n=8):
    if segment is not None:
        n = len(segment)  # one label per point
    scene_dir = os.path.join(root, split, name)
    os.makedirs(scene_dir, exist_ok=True)
    np.save(os.path.join(scene_dir, "coord.npy"), np.random.rand(n, 3).astype(np.float32))
    np.save(os.path.join(scene_dir, "strength.npy"), np.random.rand(n).astype(np.float32))
    if segment is None:
        segment = np.random.randint(0, 2, size=(n,))
    np.save(os.path.join(scene_dir, "segment.npy"), np.asarray(segment, dtype=np.int32))
    return scene_dir


def test_include_names_keeps_only_matching_scene_prefix(tmp_path):
    for name in ("T1_0-0", "T1_0-1", "T2_0-0", "T2_1-0", "T20_0-0", "T3_0-0"):
        _write_scene(str(tmp_path), "test", name)

    dataset = OpenGFDataset(str(tmp_path), "test", include_names="T2")

    kept = sorted(os.path.basename(scene_dir) for scene_dir, _ in dataset.entries)
    assert kept == ["T2_0-0", "T2_1-0"]  # not T1_*, T3_*, nor T20_*


def test_include_names_accepts_a_list_of_prefixes(tmp_path):
    for name in ("T1_0-0", "T2_0-0", "T3_0-0"):
        _write_scene(str(tmp_path), "test", name)

    dataset = OpenGFDataset(str(tmp_path), "test", include_names=["T1", "T3"])

    kept = sorted(os.path.basename(scene_dir) for scene_dir, _ in dataset.entries)
    assert kept == ["T1_0-0", "T3_0-0"]


def test_include_names_none_keeps_every_scene(tmp_path):
    for name in ("T1_0-0", "T2_0-0"):
        _write_scene(str(tmp_path), "test", name)

    dataset = OpenGFDataset(str(tmp_path), "test")

    assert len(dataset) == 2


def test_dataset_has_no_color_and_has_strength(tmp_path):
    _write_scene(str(tmp_path), "train", "S1_1_0-0")

    data = OpenGFDataset(str(tmp_path), "train", pre_filter=None)[0]

    assert torch.equal(data.color_mask, torch.ones(data.x.size(0), dtype=torch.bool))
    assert not hasattr(data, "strength_mask")
    assert data.x[:, 0].abs().sum() > 0  # real intensity is kept


def test_remap_labels_merges_outliers_into_non_ground_and_keeps_every_point(tmp_path):
    _write_scene(str(tmp_path), "train", "S1_1_0-0", segment=[0, 1, 2, 2, 1, 0])

    data = OpenGFDataset(str(tmp_path), "train", remap_labels={2: 1}, pre_filter=None)[0]

    assert data.y.tolist() == [0, 1, 1, 1, 1, 0]
    assert data.pos.size(0) == 6


def test_drop_labels_physically_removes_points_from_every_per_point_attribute(tmp_path):
    _write_scene(str(tmp_path), "test", "T2_0-0", segment=[0, 2, 1, 2, 0, 1])

    data = OpenGFDataset(str(tmp_path), "test", drop_labels=[2], pre_filter=None)[0]

    assert data.y.tolist() == [0, 1, 0, 1]
    assert data.pos.size(0) == 4
    assert data.x.size(0) == 4
    assert data.color_mask.size(0) == 4
    # idx_in_original_cloud still points back at the surviving points' original rows.
    assert data.idx_in_original_cloud.tolist() == [0, 2, 4, 5]


def test_drop_labels_runs_before_the_transform(tmp_path):
    _write_scene(str(tmp_path), "test", "T2_0-0", segment=[0, 2, 1, 2, 0, 1])
    seen = []

    def spy(data):
        seen.append(data.y.tolist())
        return data

    OpenGFDataset(str(tmp_path), "test", drop_labels=[2], pre_filter=None, transform=spy)[0]

    assert seen == [[0, 1, 0, 1]]


def test_labels_are_untouched_by_default(tmp_path):
    _write_scene(str(tmp_path), "test", "T2_0-0", segment=[0, 2, 1, 2])

    data = OpenGFDataset(str(tmp_path), "test", pre_filter=None)[0]

    assert data.y.tolist() == [0, 2, 1, 2]
