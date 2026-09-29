import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bench_inference_speed as bench  # noqa: E402

from myria3d.pctl.dataset.pointcept_npy import PointceptNpyDataset  # noqa: E402

PATCH_IDS = [f"D067-2021_UU-S1-31_1-{i}" for i in range(5)]
NUM_POINTS = 4000
PATCH_WIDTH = 102.4


def _write_manifest(path: Path, split: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "patch_id", "dept_year", "roi", "LIDARHD"])
        writer.writeheader()
        for patch_id in PATCH_IDS:
            writer.writerow(
                dict(
                    split=split,
                    patch_id=patch_id,
                    dept_year="D067-2021",
                    roi="UU-S1-31",
                    LIDARHD="True",
                )
            )


def _write_scene(data_root: Path, split: str, patch_id: str, seed: int):
    scene_dir = data_root / split / "D067-2021_LIDARHD" / "UU-S1-31" / patch_id
    scene_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    coord = rng.uniform(0.0, PATCH_WIDTH, size=(NUM_POINTS, 3)).astype(np.float32)
    coord[:, 2] = rng.uniform(0.0, 20.0, size=NUM_POINTS)
    # Pin the min corner so the 4 quadrants of the 104/52 mosaic are all populated.
    coord[0, :2] = 0.0
    np.save(scene_dir / "coord.npy", coord)
    np.save(scene_dir / "color.npy", rng.integers(0, 255, (NUM_POINTS, 3), dtype=np.uint8))
    np.save(scene_dir / "strength.npy", rng.uniform(0, 1, NUM_POINTS).astype(np.float32))
    np.save(scene_dir / "segment.npy", rng.integers(0, 15, NUM_POINTS).astype(np.int32))
    return scene_dir


@pytest.fixture
def synthetic_tree(tmp_path):
    data_root = tmp_path / "data"
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, split="test")
    for seed, patch_id in enumerate(PATCH_IDS):
        _write_scene(data_root, "test", patch_id, seed)
    return str(data_root), str(manifest)


def _dataset(data_root, manifest):
    return PointceptNpyDataset(
        data_root=data_root,
        csv_manifest=manifest,
        tile_width=104,
        subtile_width=52,
        eval_transform=None,
    )


def test_select_tiles_groups_the_four_subtiles_of_each_scene(synthetic_tree):
    data_root, manifest = synthetic_tree
    dataset = _dataset(data_root, manifest)

    tiles = bench.select_tiles(dataset, split="test", num_tiles=3, seed=42, mode="first")

    assert [t.patch_id for t in tiles] == PATCH_IDS[:3]
    for tile in tiles:
        assert len(tile.entry_indices) == 4
        assert [dataset._scenes[i][2] for i in tile.entry_indices] == [0, 1, 2, 3]
        assert {dataset._scenes[i][0] for i in tile.entry_indices} == {tile.scene_dir}
        # raw count is read from coord.npy (same convention as Pointcept's bench tiles)
        assert tile.num_points_raw == NUM_POINTS


def test_select_tiles_random_matches_pointcept_sampling_rule(synthetic_tree):
    """Same seeded permutation of the CSV-ordered split as Pointcept's bench, so both
    benches pick the same tiles when their test lists agree."""
    data_root, manifest = synthetic_tree
    dataset = _dataset(data_root, manifest)

    tiles = bench.select_tiles(dataset, split="test", num_tiles=3, seed=42, mode="random")

    perm = np.random.default_rng(42).permutation(len(PATCH_IDS))
    assert [t.patch_id for t in tiles] == [PATCH_IDS[int(i)] for i in perm[:3]]


def test_select_tiles_honours_explicit_tile_names_and_fails_on_unknown(synthetic_tree):
    data_root, manifest = synthetic_tree
    dataset = _dataset(data_root, manifest)
    wanted = [PATCH_IDS[3], PATCH_IDS[1]]

    tiles = bench.select_tiles(
        dataset, split="test", num_tiles=2, seed=0, mode="random", tile_names=wanted
    )
    assert [t.patch_id for t in tiles] == wanted

    with pytest.raises(RuntimeError, match="missing"):
        bench.select_tiles(
            dataset, split="test", num_tiles=1, seed=0, mode="random", tile_names=["nope"]
        )


def test_select_tiles_rejects_more_tiles_than_available(synthetic_tree):
    data_root, manifest = synthetic_tree
    dataset = _dataset(data_root, manifest)
    with pytest.raises(ValueError, match="Only 5 tiles"):
        bench.select_tiles(dataset, split="test", num_tiles=6, seed=0, mode="first")


def _record(tile_idx, subtile, *, stall, xfer, fwd, knn, voxel, net, oom=False, empty=False):
    return dict(
        tile_idx=tile_idx,
        subtile=subtile,
        patch_id=f"tile{tile_idx}",
        stall_ms=stall,
        transfer_ms=xfer,
        forward_ms=fwd,
        knn_ms=knn,
        num_points_voxel=voxel,
        num_points_net=net,
        num_points_interp=voxel,
        oom=oom,
        empty=empty,
    )


def _tile_records(tile_idx, per_subtile_ms, voxel=1000, net=800):
    return [
        _record(tile_idx, s, stall=ms, xfer=0.0, fwd=0.0, knn=0.0, voxel=voxel, net=net)
        for s, ms in enumerate(per_subtile_ms)
    ]


def test_summarize_reports_raw_and_voxel_points_per_second_and_skips_warmup():
    records = []
    records += _tile_records(0, [1000.0] * 4)  # warmup tile: must be ignored
    records += _tile_records(1, [100.0] * 4)  # 400 ms per tile, 4000 voxels
    records += _tile_records(2, [100.0] * 4)
    tile_raw_points = {0: 999_999, 1: 40_000, 2: 60_000}

    s = bench.summarize(records, tile_raw_points, num_warmup=1)

    assert s["n_measured"] == 2 and s["n_failed"] == 0
    assert s["total_ms_mean"] == pytest.approx(400.0)
    assert s["pts_per_sec_raw"] == pytest.approx(100_000 / 0.8)
    assert s["pts_per_sec_voxel"] == pytest.approx(8000 / 0.8)
    assert s["tiles_per_sec"] == pytest.approx(2 / 0.8)
    assert s["num_points_raw_mean"] == pytest.approx(50_000)


def test_summarize_excludes_oom_tiles_and_reports_cap_pressure():
    records = []
    records += _tile_records(0, [10.0] * 4)  # warmup
    records += _tile_records(1, [100.0] * 4, voxel=50_000, net=40_000)  # capped
    records += _tile_records(2, [100.0] * 4, voxel=30_000, net=30_000)  # not capped
    records[8]["oom"] = True  # one OOM subtile fails the whole of tile 2
    tile_raw_points = {0: 1, 1: 10, 2: 10}

    s = bench.summarize(records, tile_raw_points, num_warmup=1)

    assert s["n_measured"] == 1 and s["n_failed"] == 1
    assert s["frac_subtiles_capped"] == pytest.approx(1.0)


def test_summarize_all_failed_returns_counts_only():
    records = _tile_records(1, [1.0] * 4)
    records[0]["oom"] = True
    s = bench.summarize(records, {1: 5}, num_warmup=0)
    assert s == dict(n_measured=0, n_failed=1, mode="pipeline")


@pytest.mark.slow
def test_main_smoke_on_cpu_writes_per_tile_csv_and_summary(synthetic_tree, tmp_path):
    data_root, manifest = synthetic_tree
    out_dir = tmp_path / "out"

    summary = bench.main(
        [
            "--data-root",
            data_root,
            "--csv-manifest",
            manifest,
            "--split",
            "test",
            "--num-tiles",
            "3",
            "--num-warmup",
            "1",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--no-cache-warmup",
            "--out-dir",
            str(out_dir),
        ]
    )

    stats = summary["summaries"]["randlanet_multitask"]["pipeline"]
    assert stats["n_measured"] == 2
    assert stats["pts_per_sec_raw"] > 0
    assert stats["knn_ms_mean"] > 0  # KNN interpolation is part of the measured forward
    # voxel count is taken before the 40k cap, net after it
    assert stats["num_points_voxel_mean"] >= stats["num_points_net_mean"] > 0

    per_subtile = list(csv.DictReader(open(out_dir / "per_subtile.csv")))
    assert len(per_subtile) == 3 * 4
    per_tile = list(csv.DictReader(open(out_dir / "per_tile.csv")))
    assert len(per_tile) == 3
    assert [r["warmup"] for r in per_tile] == ["True", "False", "False"]
    assert all(int(r["num_points_raw"]) == NUM_POINTS for r in per_tile)

    saved = json.load(open(out_dir / "summary.json"))
    assert saved["args"]["num_workers"] == "0"
    assert len(saved["tile_names"]) == 3
