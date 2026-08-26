import csv
import json
import os

import numpy as np
import pytest
import torch

from myria3d.pctl.dataset.pointcept_npy import (
    PointceptNpyDataset,
    build_scene_list,
    load_pointcept_scene,
)


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


def test_pointcept_npy_dataset_sets_patch_id(pointcept_manifest_tree):
    data_root, manifest = pointcept_manifest_tree
    dataset = PointceptNpyDataset(
        data_root=data_root,
        csv_manifest=manifest,
        tile_width=100,
        subtile_width=50,
        train_transform=None,
        eval_transform=None,
    )
    data = dataset[0]
    assert data.patch_id == "D067-2021_UU-S1-31_1-1"


def test_load_pointcept_scene_merges_buildings_and_maps_void(tmp_path):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    np.save(scene_dir / "coord.npy", np.zeros((4, 3), dtype=np.float32))
    np.save(scene_dir / "segment.npy", np.array([0, 15, 16, -1], dtype=np.int32))

    data = load_pointcept_scene(str(scene_dir))

    assert torch.equal(data.y, torch.tensor([0, 0, 0, 15]))


def _write_raster_meta(scene_dir, *, origin_x=0.0, origin_y=0.0, pixel_m=1.0):
    forest = np.array([[[0, 1], [2, 1]]], dtype=np.uint8)  # (1, 2, 2)
    network = np.zeros((3, 2, 2), dtype=np.uint8)
    network[0, 0, 1] = 1  # ROADS at (row=0, col=1)
    np.save(scene_dir / "forest_2d.npy", forest)
    np.save(scene_dir / "network.npy", network)
    meta = {
        "forest_2d": {
            "origin_x": origin_x,
            "origin_y": origin_y,
            "pixel_m": pixel_m,
        },
        "network": {
            "origin_x": origin_x,
            "origin_y": origin_y,
            "pixel_m": pixel_m,
            "channel_order": ["ROADS", "RAILROADS", "TRANSMISSION_LINES"],
        },
    }
    with open(scene_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def test_load_pointcept_scene_pixel_semantic_and_nathab_axes(tmp_path):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    # Four points sitting in the four 1 m raster cells of a 2x2 grid starting at (0, 0).
    pos = np.array(
        [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 1.5, 0.0], [1.5, 1.5, 0.0]],
        dtype=np.float32,
    )
    np.save(scene_dir / "coord.npy", pos)
    _write_raster_meta(scene_dir)
    # Raw CarHab: open-temperate-acid-humid (0) and void-sentinel (43).
    np.save(scene_dir / "natural_habitat.npy", np.array([0, 0, 43, 43], dtype=np.int64))

    data = load_pointcept_scene(str(scene_dir))

    assert torch.equal(data.y_forest_2d, torch.tensor([0, 1, 2, 1]))
    assert torch.equal(data.forest_2d_cell_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(data.forest_2d_raster_h, torch.tensor([2]))
    assert torch.equal(data.forest_2d_raster_w, torch.tensor([2]))
    assert torch.equal(data.y_roads, torch.tensor([0, 1, 0, 0]))
    assert torch.equal(data.roads_cell_id, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(data.roads_raster_h, torch.tensor([2]))
    assert torch.equal(data.roads_raster_w, torch.tensor([2]))
    # Axis 0 (open / humid / acid / temperate) vs void on every axis.
    assert torch.equal(data.y_nathab_habitat_type, torch.tensor([0, 0, 4, 4]))
    assert torch.equal(data.y_nathab_moisture_regime, torch.tensor([0, 0, 3, 3]))
    assert torch.equal(data.y_nathab_soil_chemistry, torch.tensor([0, 0, 2, 2]))
    assert torch.equal(data.y_nathab_bioclimatic_zone, torch.tensor([0, 0, 3, 3]))


def test_load_pointcept_scene_missing_raster_and_nathab_fallback(tmp_path):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    np.save(scene_dir / "coord.npy", np.zeros((3, 3), dtype=np.float32))

    data = load_pointcept_scene(str(scene_dir))

    assert torch.equal(data.forest_2d_cell_id, torch.tensor([-1, -1, -1]))
    assert torch.equal(data.y_forest_2d, torch.tensor([2, 2, 2]))
    assert torch.equal(data.forest_2d_raster_h, torch.tensor([0]))
    assert torch.equal(data.forest_2d_raster_w, torch.tensor([0]))
    assert torch.equal(data.roads_cell_id, torch.tensor([-1, -1, -1]))
    assert torch.equal(data.y_roads, torch.tensor([2, 2, 2]))
    assert torch.equal(data.roads_raster_h, torch.tensor([0]))
    assert torch.equal(data.roads_raster_w, torch.tensor([0]))
    # Missing natural_habitat.npy is filled with raw id 43 (void on every axis).
    assert torch.equal(data.y_nathab_habitat_type, torch.tensor([4, 4, 4]))
    assert torch.equal(data.y_nathab_moisture_regime, torch.tensor([3, 3, 3]))
    assert torch.equal(data.y_nathab_soil_chemistry, torch.tensor([2, 2, 2]))
    assert torch.equal(data.y_nathab_bioclimatic_zone, torch.tensor([3, 3, 3]))


def test_load_pointcept_scene_uses_coord_translation_for_raster_cells(tmp_path):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    np.save(scene_dir / "coord.npy", np.array([[0.5, 0.5, 0.0]], dtype=np.float32))
    np.save(scene_dir / "coord_translation.npy", np.array([1000.0, 2000.0, 0.0], dtype=np.float64))
    np.save(scene_dir / "forest_2d.npy", np.array([[[7]]], dtype=np.uint8))
    with open(scene_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "forest_2d": {
                    "origin_x": 1000.0,
                    "origin_y": 2000.0,
                    "pixel_m": 1.0,
                }
            },
            f,
        )

    data = load_pointcept_scene(str(scene_dir))

    assert torch.equal(data.forest_2d_cell_id, torch.tensor([0]))
    assert torch.equal(data.y_forest_2d, torch.tensor([7]))

