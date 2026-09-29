#!/usr/bin/env python3
"""Inference-speed benchmark of the Flair3D+ multitask RandLA-Net (myria3d), aligned on
Pointcept's ``scripts/bench_inference_speed.py`` (pipeline pass only).

Measures test-time throughput in **raw points / second** on the same seeded tiles as the
Pointcept bench, from ``.npy`` on disk to per-point predictions:

- DataLoader with ``--num-workers`` workers (default 8), ``batch_size=1``,
  ``prefetch_factor=1``, ``pin_memory`` -- the pipeline pass of the Pointcept bench.
  ``stall_ms`` is the time spent waiting on ``next(loader)``; CPU work hidden behind the GPU
  does not show up, exactly as in Pointcept's ``pipeline`` mode.
- One benchmark unit is a **tile** (one ~102.4 m Pointcept patch). myria3d's test mosaic
  (``eval_tile_width/eval_subtile_width`` = 104/52) cuts it into 4 subtiles that go through
  the network one by one; a tile's time is the sum over its 4 subtiles of
  ``stall + H2D + forward``. That is the counterpart of Pointcept's 1 tile = 1 fragment.
- Everything the real test run does between disk and predictions is inside the timed
  region: ``load_pointcept_scene`` (features, raster cells), ``SubtileCrop``,
  ``CategoricalGridSampling(0.1)``, ``MaximumNumNodes`` (the 40k cap of the Flair3D+ setup),
  normalizations, the network **and the KNN interpolation back to all raw points**
  (``MultiTaskModel.forward`` with ``interpolate=True``; ``--no-interpolate`` to drop it).
  Bookkeeping (loss, metrics, Pointcept ``*_logits_network.npy`` dumps) is not measured,
  like Pointcept's bench skips mIoU/APLS.
- Weights are randomly initialized: none of the ops branch on weight values.

Point counts reported per tile (all untimed):

- ``num_points_raw``: rows of ``coord.npy``. This is Pointcept's "raw" count, the numerator of
  ``pts_per_sec_raw`` (the headline figure).
- ``num_points_voxel``: points after ``CategoricalGridSampling(0.1)`` and *before* the
  ``MaximumNumNodes`` cap, summed over the 4 subtiles. Counterpart of the ``coord.shape[0]``
  Pointcept's bench divides by (``pts_per_sec_voxel``); voxels are per 52 m subtile here,
  per whole tile there, so the two differ slightly on subtile borders.
- ``num_points_net``: points actually fed to the network (after the cap), informative only;
  ``frac_subtiles_capped`` tells how often the cap bites.

Examples::

  # Local dry run on Hecate (D067 has no local test split -- use val).
  python scripts/bench_inference_speed.py \\
    --csv-manifest /data/geist/Pointcept/data/flair3d_plus/raw/scene_split_manifest_D067.csv \\
    --split val --num-tiles 15 --num-warmup 5

  # Real run on A100 (Jean Zay, national manifest, test split), same tiles as a Pointcept run.
  python scripts/bench_inference_speed.py --split test --num-tiles 200 --num-warmup 10 \\
    --tile-names-from <pointcept>/stats/flair3d/inference_speed_bench/<jobid>/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "randlanet_multitask"

PER_SUBTILE_FIELDS = [
    "tile_idx",
    "subtile",
    "patch_id",
    "warmup",
    "stall_ms",
    "transfer_ms",
    "forward_ms",
    "knn_ms",
    "num_points_voxel",
    "num_points_net",
    "num_points_interp",
    "empty",
    "oom",
]
PER_TILE_FIELDS = [
    "tile_idx",
    "patch_id",
    "warmup",
    "num_points_raw",
    "num_points_voxel",
    "num_points_net",
    "stall_ms",
    "transfer_ms",
    "forward_ms",
    "knn_ms",
    "total_ms",
    "oom",
]


@dataclass
class Tile:
    patch_id: str
    scene_dir: str
    entry_indices: List[int]  # dataset entries of the tile's subtiles, in mosaic order
    num_points_raw: int  # rows of coord.npy


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment",
        default="flair3d_plus/multitask_200k",
        help="Hydra experiment composed for datamodule/model (the real multitask test setup).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override datamodule.data_root (default: FLAIR3D_DATA_ROOT env / config).",
    )
    parser.add_argument(
        "--csv-manifest",
        default=None,
        help="Override datamodule.csv_manifest (default: FLAIR3D_CSV_MANIFEST env / config).",
    )
    parser.add_argument("--split", default="test", help="Manifest split to sample tiles from.")
    parser.add_argument(
        "--num-tiles",
        type=int,
        default=60,
        help="Total tiles benchmarked, including warmup tiles.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Leading tiles run but excluded from the stats (kernel/allocator warmup).",
    )
    parser.add_argument(
        "--tile-sample",
        choices=("random", "first"),
        default="random",
        help="Seeded shuffle of the split (default, same rule as Pointcept's bench) or "
        "manifest order.",
    )
    parser.add_argument(
        "--tile-names-from",
        default=None,
        metavar="SUMMARY_JSON",
        help="Use the first --num-tiles `tile_names` of a Pointcept bench summary.json, "
        "to benchmark exactly the same tiles.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument(
        "--cache-warmup",
        dest="cache_warmup",
        action="store_true",
        default=True,
        help="CPU-only read of the sampled tiles before timing, so the page cache is hot "
        "(default: on).",
    )
    parser.add_argument("--no-cache-warmup", dest="cache_warmup", action="store_false")
    parser.add_argument(
        "--interpolate",
        dest="interpolate",
        action="store_true",
        default=True,
        help="Include the KNN interpolation to raw points, as at test/predict time "
        "(default: on).",
    )
    parser.add_argument("--no-interpolate", dest="interpolate", action="store_false")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Wrap the forward in torch.autocast (default: off, fp32).",
    )
    parser.add_argument(
        "--matmul-precision",
        default="high",
        choices=("highest", "high", "medium"),
        help="torch.set_float32_matmul_precision; run.py uses 'high' (TF32), so does the "
        "real test run.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Default: stats/flair3d/inference_speed_bench/<timestamp>/",
    )
    parser.add_argument(
        "--override",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra overrides, e.g. model.num_workers=8.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- setup


class RecordVoxelCount:
    """Store the post-voxelization / pre-cap point count on the sample (untimed bookkeeping,
    a python int so that no transform touches it and the collater stacks it to (B,))."""

    def __call__(self, data):
        data.num_points_voxel = int(data.num_nodes)
        return data


def build_cfg(args):
    from hydra import compose, initialize_config_dir

    overrides = [
        f"experiment={args.experiment}",
        f"work_dir={REPO_ROOT}",
        # bound-free split: tiles are sampled here, not capped by the validation subset.
        "datamodule.max_val_tiles=null",
        "datamodule.val_tiles_manifest=null",
    ]
    if args.data_root:
        overrides.append(f"datamodule.data_root={args.data_root}")
    if args.csv_manifest:
        overrides.append(f"datamodule.csv_manifest={args.csv_manifest}")
    overrides += list(args.override)
    os.environ.setdefault("LOGS_DIR", str(REPO_ROOT / "logs"))
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), job_name="bench"):
        return compose(config_name="config", overrides=overrides)


def instrument_eval_transform(datamodule) -> int:
    """Insert RecordVoxelCount right before MaximumNumNodes; return the cap."""
    from myria3d.pctl.transforms.transforms import MaximumNumNodes

    # Hydra hands back ListConfigs; `eval_transform` concatenates them, which needs plain
    # lists once one of them holds an inserted transform.
    transforms = list(datamodule.preparation_eval_transform)
    for i, transform in enumerate(transforms):
        if isinstance(transform, MaximumNumNodes):
            transforms.insert(i, RecordVoxelCount())
            datamodule.preparation_eval_transform = transforms
            datamodule.normalization_transform = list(datamodule.normalization_transform)
            return int(transform.num)
    raise RuntimeError("No MaximumNumNodes in the eval preparations: cannot count voxels.")


class KnnTimer:
    """Accumulates the wall time spent in MultiTaskModel._interpolate_outputs."""

    def __init__(self, model):
        self.total_ms = 0.0
        original = model._interpolate_outputs

        def timed(*args, **kwargs):
            # The network runs asynchronously and the first `.cpu()` inside would wait for
            # it: drain the GPU first so only the interpolation itself is attributed here.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.total_ms += (time.perf_counter() - t0) * 1000.0

        model._interpolate_outputs = timed


# --------------------------------------------------------------------------- tiles


def _num_rows(scene_dir: str) -> int:
    return int(np.load(os.path.join(scene_dir, "coord.npy"), mmap_mode="r").shape[0])


def select_tiles(dataset, split, num_tiles, seed, mode, tile_names=None) -> List[Tile]:
    """Pick `num_tiles` scenes of `split` and group their subtile dataset entries.

    Same rule as Pointcept's bench: names in manifest order, then a seeded
    ``default_rng(seed).permutation`` (``mode="random"``) or the first N (``"first"``).
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for entry_idx, (scene_dir, entry_split, subtile_index) in enumerate(dataset._scenes):
        if entry_split == split and subtile_index is not None:
            groups[scene_dir].append(entry_idx)
    scene_dirs = list(groups)
    patch_id = lambda d: os.path.basename(os.path.normpath(d))  # noqa: E731

    if tile_names is not None:
        by_name = {patch_id(d): d for d in scene_dirs}
        wanted = list(tile_names)[:num_tiles]
        missing = [n for n in wanted if n not in by_name]
        if missing:
            raise RuntimeError(
                f"{len(missing)} requested tile(s) missing from split={split!r} "
                f"(first missing: {missing[0]!r})"
            )
        chosen = [by_name[n] for n in wanted]
    else:
        if len(scene_dirs) < num_tiles:
            raise ValueError(
                f"Only {len(scene_dirs)} tiles available for split={split!r}, need "
                f"--num-tiles={num_tiles}."
            )
        if mode == "first":
            chosen = scene_dirs[:num_tiles]
        elif mode == "random":
            perm = np.random.default_rng(seed).permutation(len(scene_dirs))
            chosen = [scene_dirs[int(i)] for i in perm[:num_tiles]]
        else:
            raise ValueError(f"Unknown tile sampling mode {mode!r}")

    return [Tile(patch_id(d), d, groups[d], _num_rows(d)) for d in chosen]


def warmup_page_cache(tiles: List[Tile]):
    from myria3d.pctl.dataset.pointcept_npy import load_pointcept_scene

    print(f"[bench] cache warmup: loading {len(tiles)} tiles (CPU only, discarded) ...")
    for i, tile in enumerate(tiles):
        load_pointcept_scene(tile.scene_dir)
        if (i + 1) % 20 == 0 or i + 1 == len(tiles):
            print(f"[bench] cache warmup {i + 1}/{len(tiles)} {tile.patch_id}")


# --------------------------------------------------------------------------- timing


def _timed_h2d_and_forward(model, batch, knn_timer, args, device):
    use_cuda = device.type == "cuda"
    if use_cuda:
        start_ev, end_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start_ev.record()
        batch = batch.to(device, non_blocking=True)
        end_ev.record()
        torch.cuda.synchronize()
        transfer_ms = start_ev.elapsed_time(end_ev)
    else:
        t0 = time.perf_counter()
        batch = batch.to(device)
        transfer_ms = (time.perf_counter() - t0) * 1000.0

    knn_timer.total_ms = 0.0
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=args.amp):
            model(batch, interpolate=args.interpolate)
    if use_cuda:
        torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - t0) * 1000.0

    return dict(
        transfer_ms=transfer_ms,
        forward_ms=forward_ms,
        knn_ms=knn_timer.total_ms,
        num_points_voxel=int(batch.num_points_voxel.sum()),
        num_points_net=int(batch.pos.shape[0]),
        num_points_interp=int(batch.copies["pos_copy"].shape[0]),
    )


def benchmark_pipeline(model, dataset, tiles, args, device) -> List[dict]:
    """One record per subtile, in tile order (tile_idx, subtile = position in the mosaic)."""
    from myria3d.pctl.dataloader.dataloader import GeometricNoneProofCollater

    flat = [
        (tile_idx, subtile, entry_idx)
        for tile_idx, tile in enumerate(tiles)
        for subtile, entry_idx in enumerate(tile.entry_indices)
    ]
    subset = torch.utils.data.Subset(dataset, [entry_idx for *_, entry_idx in flat])
    pin_memory = bool(args.pin_memory and device.type == "cuda")
    loader_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=GeometricNoneProofCollater(),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["persistent_workers"] = True
    loader = torch.utils.data.DataLoader(subset, **loader_kwargs)
    print(
        f"[bench][pipeline] DataLoader num_workers={args.num_workers} "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')} "
        f"pin_memory={pin_memory} interpolate={args.interpolate}"
    )

    knn_timer = KnnTimer(model)
    records = []
    loader_iter = iter(loader)
    try:
        for tile_idx, subtile, _ in flat:
            record = dict(
                tile_idx=tile_idx,
                subtile=subtile,
                patch_id=tiles[tile_idx].patch_id,
                warmup=tile_idx < args.num_warmup,
                stall_ms=None,
                transfer_ms=0.0,
                forward_ms=0.0,
                knn_ms=0.0,
                num_points_voxel=0,
                num_points_net=0,
                num_points_interp=0,
                empty=False,
                oom=False,
            )
            try:
                t0 = time.perf_counter()
                batch = next(loader_iter)
                record["stall_ms"] = (time.perf_counter() - t0) * 1000.0
                if batch is None:  # empty crop: the loader still cost some wait time
                    record["empty"] = True
                else:
                    record.update(_timed_h2d_and_forward(model, batch, knn_timer, args, device))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                record["oom"] = True
                print(f"[bench][pipeline] OOM on tile {tile_idx} subtile {subtile} -- skipped")
            if record["stall_ms"] is None:
                record["stall_ms"] = 0.0
            records.append(record)
            if subtile == len(tiles[tile_idx].entry_indices) - 1:
                _print_tile_progress(records, tile_idx, len(tiles), tiles[tile_idx])
    finally:
        del loader_iter
        del loader
    return records


def _print_tile_progress(records, tile_idx, num_tiles, tile):
    rows = [r for r in records if r["tile_idx"] == tile_idx]
    total = sum(r["stall_ms"] + r["transfer_ms"] + r["forward_ms"] for r in rows)
    tag = "warmup" if rows[0]["warmup"] else "      "
    print(
        f"[bench][pipeline] tile {tile_idx + 1:3d}/{num_tiles} [{tag}] {tile.patch_id:35s} "
        f"raw={tile.num_points_raw:9,d} vox={sum(r['num_points_voxel'] for r in rows):8,d} "
        f"net={sum(r['num_points_net'] for r in rows):8,d} "
        f"stall={sum(r['stall_ms'] for r in rows):8.1f}ms "
        f"fwd={sum(r['forward_ms'] for r in rows):8.1f}ms "
        f"(knn {sum(r['knn_ms'] for r in rows):7.1f}ms) total={total:8.1f}ms"
        + ("  OOM" if any(r["oom"] for r in rows) else "")
    )


# --------------------------------------------------------------------------- stats


def aggregate_tiles(records, tile_raw_points, num_warmup) -> List[dict]:
    """Sum the per-subtile records of each tile (a tile fails if any subtile OOMed)."""
    by_tile = defaultdict(list)
    for r in records:
        by_tile[r["tile_idx"]].append(r)
    rows = []
    for tile_idx, subtiles in by_tile.items():
        sums = {
            k: sum(r[k] for r in subtiles)
            for k in ("stall_ms", "transfer_ms", "forward_ms", "knn_ms")
        }
        rows.append(
            dict(
                tile_idx=tile_idx,
                patch_id=subtiles[0]["patch_id"],
                warmup=tile_idx < num_warmup,
                num_points_raw=tile_raw_points[tile_idx],
                num_points_voxel=sum(r["num_points_voxel"] for r in subtiles),
                num_points_net=sum(r["num_points_net"] for r in subtiles),
                total_ms=sums["stall_ms"] + sums["transfer_ms"] + sums["forward_ms"],
                oom=any(r["oom"] for r in subtiles),
                n_empty_subtiles=sum(1 for r in subtiles if r["empty"]),
                n_capped_subtiles=sum(
                    1 for r in subtiles if r["num_points_voxel"] > r["num_points_net"]
                ),
                n_nonempty_subtiles=sum(1 for r in subtiles if not r["empty"] and not r["oom"]),
                **sums,
            )
        )
    return rows


def summarize(records, tile_raw_points, num_warmup) -> dict:
    tiles = [t for t in aggregate_tiles(records, tile_raw_points, num_warmup) if not t["warmup"]]
    measured = [t for t in tiles if not t["oom"]]
    n_failed = len(tiles) - len(measured)
    if not measured:
        return dict(n_measured=0, n_failed=n_failed, mode="pipeline")

    def col(key):
        return np.array([t[key] for t in measured], dtype=np.float64)

    out = dict(n_measured=len(measured), n_failed=n_failed, mode="pipeline")
    for key in ("stall_ms", "transfer_ms", "forward_ms", "knn_ms", "total_ms"):
        values = col(key)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_median"] = float(np.median(values))
        out[f"{key}_std"] = float(values.std())
    for key in ("num_points_raw", "num_points_voxel", "num_points_net"):
        values = col(key)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_min"] = float(values.min())
        out[f"{key}_max"] = float(values.max())
    total_s = col("total_ms").sum() / 1000.0
    out["pts_per_sec_raw"] = float(col("num_points_raw").sum() / total_s)
    out["pts_per_sec_voxel"] = float(col("num_points_voxel").sum() / total_s)
    out["pts_per_sec_net"] = float(col("num_points_net").sum() / total_s)
    out["tiles_per_sec"] = float(len(measured) / total_s)
    nonempty = col("n_nonempty_subtiles").sum()
    out["frac_subtiles_capped"] = (
        float(col("n_capped_subtiles").sum() / nonempty) if nonempty else 0.0
    )
    out["n_empty_subtiles"] = int(col("n_empty_subtiles").sum())
    return out


def _mean_std(s, key, decimals):
    return f"{s[f'{key}_mean']:.{decimals}f}±{s[f'{key}_std']:.{decimals}f}"


def print_summary(stats, excluded):
    print(f"\n=== Summary pipeline ({excluded}) ===")
    if stats["n_measured"] == 0:
        print(f"{MODEL_NAME}: all measured tiles failed ({stats['n_failed']})")
        return
    print(
        f"{MODEL_NAME}: n_ok={stats['n_measured']} n_fail={stats['n_failed']}\n"
        f"  per tile (mean±std): stall {_mean_std(stats, 'stall_ms', 1)} ms | "
        f"xfer {_mean_std(stats, 'transfer_ms', 2)} ms | "
        f"forward {_mean_std(stats, 'forward_ms', 1)} ms (of which KNN "
        f"{_mean_std(stats, 'knn_ms', 1)} ms) | total {_mean_std(stats, 'total_ms', 1)} ms\n"
        f"  points per tile (mean): raw {stats['num_points_raw_mean']:,.0f} | "
        f"voxel {stats['num_points_voxel_mean']:,.0f} | net {stats['num_points_net_mean']:,.0f} "
        f"(subtiles hitting the cap: {stats['frac_subtiles_capped']:.1%})\n"
        f"  pts/s(raw) = {stats['pts_per_sec_raw']:,.0f}   "
        f"pts/s(voxel) = {stats['pts_per_sec_voxel']:,.0f}   "
        f"tiles/s = {stats['tiles_per_sec']:.3f}"
    )


# --------------------------------------------------------------------------- main


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> dict:
    import hydra

    import myria3d  # noqa: F401  (registers the `get_method` OmegaConf resolver)

    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available (use --device cpu for a smoke test).")
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(args.seed)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT
        / "stats"
        / "flair3d"
        / "inference_speed_bench"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    max_num_nodes = instrument_eval_transform(datamodule)
    dataset = datamodule.dataset

    tile_names = None
    if args.tile_names_from:
        with open(args.tile_names_from) as f:
            tile_names = json.load(f)["tile_names"]
    print(
        f"[bench] sampling {args.num_tiles} tiles of split={args.split!r} "
        f"(tile_sample={args.tile_sample!r}, seed={args.seed}, "
        f"tile_names_from={args.tile_names_from!r}) ..."
    )
    tiles = select_tiles(
        dataset, args.split, args.num_tiles, args.seed, args.tile_sample, tile_names
    )
    dept_years = sorted({t.patch_id.split("_")[0] for t in tiles})
    print(
        f"[bench] {len(tiles)} tiles from {len(dept_years)} dept_year "
        f"(first 3: {[t.patch_id for t in tiles[:3]]}); "
        f"{sum(len(t.entry_indices) for t in tiles)} subtiles, cap={max_num_nodes} pts/subtile"
    )

    if args.cache_warmup:
        warmup_page_cache(tiles)

    model = hydra.utils.instantiate(cfg.model).to(device)
    model.eval()
    print(f"\n[bench] === {MODEL_NAME} ({args.experiment}) ===")
    records = benchmark_pipeline(model, dataset, tiles, args, device)

    tile_raw_points = {i: t.num_points_raw for i, t in enumerate(tiles)}
    stats = summarize(records, tile_raw_points, args.num_warmup)
    print_summary(stats, f"warmup tiles excluded: first {args.num_warmup}/{args.num_tiles}")

    _write_csv(
        out_dir / "per_subtile.csv",
        PER_SUBTILE_FIELDS,
        records,
    )
    _write_csv(
        out_dir / "per_tile.csv",
        PER_TILE_FIELDS,
        aggregate_tiles(records, tile_raw_points, args.num_warmup),
    )
    summary = dict(
        args={k: str(v) for k, v in vars(args).items()},
        experiment=args.experiment,
        tile_sample=args.tile_sample,
        seed=args.seed,
        tile_names=[t.patch_id for t in tiles],
        dept_years=dept_years,
        setup=dict(
            device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            torch=torch.__version__,
            max_num_nodes=max_num_nodes,
            voxel=float(cfg.datamodule.transforms.preparations.voxel),
            eval_tile_width=cfg.datamodule.transforms.preparations.eval_tile_width,
            eval_subtile_width=cfg.datamodule.transforms.preparations.eval_subtile_width,
            num_workers=args.num_workers,
            interpolate=args.interpolate,
            float32_matmul_precision=args.matmul_precision,
            amp=args.amp,
        ),
        summaries={MODEL_NAME: dict(pipeline=stats)},
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[bench] wrote {out_dir}/per_subtile.csv, per_tile.csv, summary.json")
    return summary


if __name__ == "__main__":
    main()
