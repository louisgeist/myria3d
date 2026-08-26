import csv
import os

import numpy as np
import pytest
import torch

from myria3d.pctl.dataset.pointcept_npy import build_scene_list, load_pointcept_scene


def _write_manifest(path: str, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "patch_id", "dept_year", "roi", "LIDARHD"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_scene(data_root: str, split: str, patch_id: str, dept_year: str, roi: str):
    scene_dir = os.path.join(data_root, split, f"{dept_year}_LIDARHD", roi, patch_id)
    os.makedirs(scene_dir, exist_ok=True)
    coord = np.zeros((10, 3), dtype=np.float32)
    np.save(os.path.join(scene_dir, "coord.npy"), coord)


@pytest.fixture
def pointcept_manifest_tree(tmp_path):
    data_root = tmp_path / "data"
    manifest = tmp_path / "manifest.csv"
    rows = [
        {
            "split": "train",
            "patch_id": "D067-2021_UU-S1-31_1-1",
            "dept_year": "D067-2021",
            "roi": "UU-S1-31",
            "LIDARHD": "True",
        },
        {
            "split": "val",
            "patch_id": "D067-2021_UU-S1-31_1-2",
            "dept_year": "D067-2021",
            "roi": "UU-S1-31",
            "LIDARHD": "True",
        },
    ]
    _write_manifest(str(manifest), rows)
    _write_scene(str(data_root), "train", rows[0]["patch_id"], rows[0]["dept_year"], rows[0]["roi"])
    _write_scene(str(data_root), "val", rows[1]["patch_id"], rows[1]["dept_year"], rows[1]["roi"])
    return str(data_root), str(manifest)


def test_build_scene_list_expands_val_into_four_subtiles(pointcept_manifest_tree):
    data_root, manifest = pointcept_manifest_tree
    scenes = build_scene_list(
        data_root=data_root,
        csv_manifest=manifest,
        tile_width=100,
        subtile_width=50,
    )
    train_entries = [s for s in scenes if s[1] == "train"]
    val_entries = [s for s in scenes if s[1] == "val"]

    assert len(train_entries) == 1
    assert train_entries[0][2] is None

    assert len(val_entries) == 4
    assert [entry[2] for entry in val_entries] == [0, 1, 2, 3]
    assert len({entry[0] for entry in val_entries}) == 1


def test_load_pointcept_scene_merges_buildings_and_maps_void(tmp_path):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    np.save(scene_dir / "coord.npy", np.zeros((4, 3), dtype=np.float32))
    np.save(scene_dir / "segment.npy", np.array([0, 15, 16, -1], dtype=np.int32))

    data = load_pointcept_scene(str(scene_dir))

    assert torch.equal(data.y, torch.tensor([0, 0, 0, 15]))
