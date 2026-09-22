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
    _write_scene(
        str(data_root), "train", rows[0]["patch_id"], rows[0]["dept_year"], rows[0]["roi"]
    )
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


def _pointcept_manifest_tree_with_n_val_tiles(tmp_path, n_val_tiles):
    data_root = tmp_path / "data"
    manifest = tmp_path / "manifest.csv"
    rows = [
        {
            "split": "val",
            "patch_id": f"D067-2021_UU-S1-31_1-{i}",
            "dept_year": "D067-2021",
            "roi": "UU-S1-31",
            "LIDARHD": "True",
        }
        for i in range(n_val_tiles)
    ]
    _write_manifest(str(manifest), rows)
    for row in rows:
        _write_scene(str(data_root), "val", row["patch_id"], row["dept_year"], row["roi"])
    return str(data_root), str(manifest)


def test_build_scene_list_caps_val_tiles_deterministically(tmp_path):
    data_root, manifest = _pointcept_manifest_tree_with_n_val_tiles(tmp_path, n_val_tiles=10)

    scenes_a = build_scene_list(
        data_root=data_root,
        csv_manifest=manifest,
        tile_width=100,
        subtile_width=50,
        max_val_tiles=3,
        val_tiles_seed=42,
    )
    scenes_b = build_scene_list(
        data_root=data_root,
        csv_manifest=manifest,
        tile_width=100,
        subtile_width=50,
        max_val_tiles=3,
        val_tiles_seed=42,
    )

    val_tiles_a = {s[0] for s in scenes_a if s[1] == "val"}
    val_tiles_b = {s[0] for s in scenes_b if s[1] == "val"}
    assert len(val_tiles_a) == 3
    # 3 tiles x 4 subtiles/tile (100 m tile / 50 m subtile) each.
    assert len([s for s in scenes_a if s[1] == "val"]) == 12
    assert val_tiles_a == val_tiles_b  # same seed -> same subset


def test_build_scene_list_does_not_cap_val_tiles_by_default(tmp_path):
    data_root, manifest = _pointcept_manifest_tree_with_n_val_tiles(tmp_path, n_val_tiles=10)

    scenes = build_scene_list(
        data_root=data_root,
        csv_manifest=manifest,
        tile_width=100,
        subtile_width=50,
    )

    val_tiles = {s[0] for s in scenes if s[1] == "val"}
    assert len(val_tiles) == 10


def test_build_scene_list_pins_val_tiles_to_manifest(tmp_path):
    data_root, manifest = _pointcept_manifest_tree_with_n_val_tiles(tmp_path, n_val_tiles=10)
    val_tiles_manifest = tmp_path / "val_dev_subset_2000.csv"
    pinned_patch_ids = [f"D067-2021_UU-S1-31_1-{i}" for i in (2, 5, 7)]
    with open(val_tiles_manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "patch_id"])
        writer.writeheader()
        for patch_id in pinned_patch_ids:
            writer.writerow({"split": "val", "patch_id": patch_id})

    scenes = build_scene_list(
        data_root=data_root,
        csv_manifest=manifest,
        tile_width=100,
        subtile_width=50,
        val_tiles_manifest=str(val_tiles_manifest),
    )

    val_patch_ids = {os.path.basename(s[0]) for s in scenes if s[1] == "val"}
    assert val_patch_ids == set(pinned_patch_ids)
    assert len([s for s in scenes if s[1] == "val"]) == 12  # 3 tiles x 4 subtiles


def test_build_scene_list_falls_back_to_max_val_tiles_when_manifest_missing(tmp_path):
    data_root, manifest = _pointcept_manifest_tree_with_n_val_tiles(tmp_path, n_val_tiles=10)

    scenes = build_scene_list(
        data_root=data_root,
        csv_manifest=manifest,
        tile_width=100,
        subtile_width=50,
        val_tiles_manifest=str(tmp_path / "does_not_exist.csv"),
        max_val_tiles=3,
        val_tiles_seed=42,
    )

    val_tiles = {s[0] for s in scenes if s[1] == "val"}
    assert len(val_tiles) == 3


def test_build_scene_list_manifest_ignores_patch_ids_not_locally_available(tmp_path):
    data_root, manifest = _pointcept_manifest_tree_with_n_val_tiles(tmp_path, n_val_tiles=3)
    val_tiles_manifest = tmp_path / "val_dev_subset_2000.csv"
    with open(val_tiles_manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "patch_id"])
        writer.writeheader()
        writer.writerow({"split": "val", "patch_id": "D067-2021_UU-S1-31_1-0"})
        writer.writerow({"split": "val", "patch_id": "some_tile_not_present_here"})

    scenes = build_scene_list(
        data_root=data_root,
        csv_manifest=manifest,
        tile_width=100,
        subtile_width=50,
        val_tiles_manifest=str(val_tiles_manifest),
    )

    val_patch_ids = {os.path.basename(s[0]) for s in scenes if s[1] == "val"}
    assert val_patch_ids == {"D067-2021_UU-S1-31_1-0"}


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


def test_load_pointcept_scene_passes_segment_through_unchanged(tmp_path):
    """segment.npy is already remapped to train ids by Pointcept's own preprocessing
    (v20: identity 0-15, 15=Void) — myria3d must not re-remap it a second time, or
    genuinely-Void points get silently relabeled Building(0)."""
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    np.save(scene_dir / "coord.npy", np.zeros((4, 3), dtype=np.float32))
    np.save(scene_dir / "segment.npy", np.array([0, 7, 15, 3], dtype=np.int32))

    data = load_pointcept_scene(str(scene_dir))

    assert torch.equal(data.y, torch.tensor([0, 7, 15, 3]))


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
    # Pointcept-computed final class ids, one column per axis (habitat_type,
    # moisture_regime, soil_chemistry, bioclimatic_zone) -- not raw CarHab ids.
    # Points 0-1 are class 0 on every axis; points 2-3 are void (== each axis's own
    # ignore_index: 4, 3, 2, 3).
    np.save(
        scene_dir / "natural_habitat.npy",
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [4, 3, 2, 3], [4, 3, 2, 3]], dtype=np.uint8),
    )

    data = load_pointcept_scene(str(scene_dir))

    assert torch.equal(data.y_forest_2d, torch.tensor([0, 1, 2, 1]))
    assert torch.equal(data.forest_2d_cell_id, torch.tensor([0, 1, 2, 3]))
    assert (data.forest_2d_raster_h, data.forest_2d_raster_w) == (2, 2)
    assert torch.equal(data.y_roads, torch.tensor([0, 1, 0, 0]))
    assert torch.equal(data.roads_cell_id, torch.tensor([0, 1, 2, 3]))
    assert (data.roads_raster_h, data.roads_raster_w) == (2, 2)
    assert torch.equal(data.y_nathab_habitat_type, torch.tensor([0, 0, 4, 4]))
    assert torch.equal(data.y_nathab_moisture_regime, torch.tensor([0, 0, 3, 3]))
    assert torch.equal(data.y_nathab_soil_chemistry, torch.tensor([0, 0, 2, 2]))
    assert torch.equal(data.y_nathab_bioclimatic_zone, torch.tensor([0, 0, 3, 3]))


def test_load_pointcept_scene_rejects_wrong_natural_habitat_shape(tmp_path):
    """Regression: natural_habitat.npy used to be read as raw CarHab ids via
    `.reshape(-1)`, which silently corrupted -- rather than rejected -- files already
    holding the newer (N, 4) per-axis class-id format (mismatched length vs. num_points
    made the tensor skip every later crop/GridSampling step untouched)."""
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    np.save(scene_dir / "coord.npy", np.zeros((3, 3), dtype=np.float32))
    np.save(scene_dir / "natural_habitat.npy", np.zeros((3, 5), dtype=np.uint8))

    with pytest.raises(ValueError, match="natural_habitat.npy"):
        load_pointcept_scene(str(scene_dir))


def test_load_pointcept_scene_missing_raster_and_nathab_fallback(tmp_path):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    np.save(scene_dir / "coord.npy", np.zeros((3, 3), dtype=np.float32))

    data = load_pointcept_scene(str(scene_dir))

    assert torch.equal(data.forest_2d_cell_id, torch.tensor([-1, -1, -1]))
    assert torch.equal(data.y_forest_2d, torch.tensor([2, 2, 2]))
    assert (data.forest_2d_raster_h, data.forest_2d_raster_w) == (0, 0)
    assert torch.equal(data.roads_cell_id, torch.tensor([-1, -1, -1]))
    assert torch.equal(data.y_roads, torch.tensor([2, 2, 2]))
    assert (data.roads_raster_h, data.roads_raster_w) == (0, 0)
    # Missing natural_habitat.npy is filled with each axis's own ignore_index (void).
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


def test_degenerate_gridsampled_scene_still_batches_with_a_normal_scene(tmp_path):
    """Regression: a subtile crop that leaves num_nodes == 1 used to make GridSampling
    mean-reduce (float-cast) the (1,) raster-hw tensors, so batching that scene with a
    normal one blew up with "torch.cat(): input types can't be cast to Long"."""
    from torch_geometric.data import Batch
    from torch_geometric.transforms import GridSampling

    small = tmp_path / "small"
    small.mkdir()
    np.save(small / "coord.npy", np.zeros((1, 3), dtype=np.float32))
    big = tmp_path / "big"
    big.mkdir()
    np.save(big / "coord.npy", np.random.rand(64, 3).astype(np.float32))

    grid = GridSampling(0.1)
    a = grid(load_pointcept_scene(str(small)))  # collapses to a single node
    b = grid(load_pointcept_scene(str(big)))
    assert a.num_nodes == 1

    batch = Batch.from_data_list([a, b])
    for key in ("forest_2d_raster_h", "roads_raster_w"):
        assert batch[key].dtype == torch.long
        assert batch[key].numel() == 2
