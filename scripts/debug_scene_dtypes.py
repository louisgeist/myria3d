"""Track down PyG ``Batch.from_data_list`` collate failures on Flair3D+ multitask data.

The symptom on Jean Zay / Hecate is a DataLoader worker crash like::

    RuntimeError: torch.cat(): input types can't be cast to the desired output type Long

That means one ``Data`` attribute has a *different dtype* on different scenes that land
in the same batch: PyG sizes the output buffer from the first scene in the batch, so a
batch whose first scene has key ``K`` as int64 blows up as soon as another scene has
``K`` as float (and vice-versa, intermittently, depending on shuffle order).

On Flair3D+ this is almost always caused by a per-point label ``.npy`` whose row count
does not match ``coord.npy``. PyG's ``GridSampling`` only mean-reduces (float-casts) a
tensor when ``tensor.size(0) == num_nodes``; a size-mismatched label tensor is silently
skipped and stays int64, while every well-formed scene's copy of that key becomes
float32. Hence ``--check-npy-counts`` below, which finds the root cause directly.

Usage
-----
Fast root-cause scan (no torch model / Hydra needed), full split::

    python scripts/debug_scene_dtypes.py \
        --data-root "$FLAIR3D_DATA_ROOT" \
        --csv-manifest "$FLAIR3D_CSV_MANIFEST" \
        --split train --limit 0

Faithful reproduction -- apply the *real* training transform pipeline (composed from the
experiment config) before comparing dtypes, and try an actual collate of any mismatch::

    python scripts/debug_scene_dtypes.py \
        --data-root "$FLAIR3D_DATA_ROOT" \
        --csv-manifest "$FLAIR3D_CSV_MANIFEST" \
        --split train --limit 0 \
        --transform train --experiment flair3d_plus/multitask_200k

On Jean Zay the two env vars are exported by
``scripts/jz/run_flair3d_plus_multitask_200k.slurm``.
"""

import argparse
import os
import os.path as osp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Allow running as `python scripts/debug_scene_dtypes.py` from anywhere -- put the repo
# root (this file's grandparent) on sys.path so `myria3d` is importable without needing
# PYTHONPATH set or `python -m`.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from myria3d.pctl.dataset.pointcept_npy import PointceptNpyDataset  # noqa: E402

# Per-point .npy assets written by the Pointcept preprocessing. Each of these, when
# present, must have exactly as many rows as coord.npy or the collate dtype bug below
# is triggered (see module docstring).
PER_POINT_NPY_ASSETS = (
    "color",
    "strength",
    "segment",
    "elevation",
    "natural_habitat",
)


def _dtype_label(value) -> str:
    if isinstance(value, torch.Tensor):
        return f"torch.{str(value.dtype).replace('torch.', '')}"
    if isinstance(value, np.ndarray):
        return f"numpy:{value.dtype}"
    if hasattr(value, "dtype"):
        return f"other:{value.dtype}"
    return type(value).__name__


def _shape_label(value) -> str:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        shp = tuple(value.shape)
        # Trailing dims only -- the leading (points) dim legitimately varies per scene.
        return f"ndim={len(shp)} trailing={shp[1:]}"
    return "-"


def check_npy_counts(data_root: str, csv_manifest: str, tile_width, subtile_width,
                     split: str, limit: int) -> bool:
    """Directly compare row counts of every per-point .npy against coord.npy.

    Returns True if any inconsistency was found. This needs no transforms and is the
    fastest way to the root cause.
    """
    from myria3d.pctl.dataset.pointcept_npy import build_scene_list

    scenes = build_scene_list(
        data_root=data_root,
        csv_manifest=csv_manifest,
        tile_width=tile_width,
        subtile_width=subtile_width,
    )
    # De-duplicate scene dirs (val/test yield one entry per subtile).
    seen = {}
    for scene_dir, scene_split, _ in scenes:
        if scene_split == split:
            seen.setdefault(scene_dir, None)
    scene_dirs = list(seen)
    if limit:
        scene_dirs = scene_dirs[:limit]

    print(f"\n=== Per-point .npy row-count check ({len(scene_dirs)} {split!r} scenes) ===")
    bad = []
    for i, scene_dir in enumerate(scene_dirs):
        coord_path = osp.join(scene_dir, "coord.npy")
        if not osp.isfile(coord_path):
            continue
        n_coord = int(np.load(coord_path, mmap_mode="r").shape[0])
        for asset in PER_POINT_NPY_ASSETS:
            asset_path = osp.join(scene_dir, f"{asset}.npy")
            if not osp.isfile(asset_path):
                continue
            n_asset = int(np.load(asset_path, mmap_mode="r").reshape(-1, 1).shape[0]) \
                if asset in ("segment", "elevation", "natural_habitat") \
                else int(np.load(asset_path, mmap_mode="r").shape[0])
            if n_asset != n_coord:
                bad.append((scene_dir, asset, n_coord, n_asset))
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(scene_dirs)}")

    if bad:
        print(f"\n  !!! {len(bad)} scene/asset pairs with a row-count mismatch vs coord.npy:")
        for scene_dir, asset, n_coord, n_asset in bad[:40]:
            print(f"    {asset:16s} rows={n_asset:>9d}  coord rows={n_coord:>9d}  {scene_dir}")
        if len(bad) > 40:
            print(f"    ... and {len(bad) - 40} more")
        print(
            "\n  These scenes are the cause of the collate dtype error. Options:\n"
            "    - exclude them via the too_small / excluded_tiles manifest, or\n"
            "    - fix / re-run the Pointcept preprocessing for the affected tiles, or\n"
            "    - make load_pointcept_scene reconcile label length with coord length."
        )
    else:
        print("  OK -- every per-point .npy matches coord.npy row count.")
    return bool(bad)


def build_transform(experiment: str, which: str, data_root: str, csv_manifest: str):
    """Compose the experiment config and return its train/eval CustomCompose transform."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    overrides = [
        f"experiment={experiment}",
        "logger=csv",  # avoid pulling comet / wandb just to read transforms
        f"datamodule.data_root={data_root}",
        f"datamodule.csv_manifest={csv_manifest}",
    ]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="config", overrides=overrides)
    datamodule = instantiate(cfg.datamodule)
    return datamodule.train_transform if which == "train" else datamodule.eval_transform


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.environ.get("FLAIR3D_DATA_ROOT"),
                   help="defaults to $FLAIR3D_DATA_ROOT")
    p.add_argument("--csv-manifest", default=os.environ.get("FLAIR3D_CSV_MANIFEST"),
                   help="defaults to $FLAIR3D_CSV_MANIFEST")
    p.add_argument("--tile-width", type=float, default=100)
    p.add_argument("--subtile-width", type=float, default=50)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=500, help="max scenes to scan (0 = all)")
    p.add_argument("--transform", default="none", choices=["none", "train", "eval"],
                   help="apply the real transform pipeline before comparing dtypes")
    p.add_argument("--experiment", default="flair3d_plus/multitask_200k",
                   help="experiment config to compose the transform pipeline from")
    p.add_argument("--check-npy-counts", dest="check_npy_counts", action="store_true",
                   default=True, help="(default on) compare per-point .npy row counts")
    p.add_argument("--no-check-npy-counts", dest="check_npy_counts", action="store_false")
    args = p.parse_args()

    if not args.data_root or not args.csv_manifest:
        p.error("--data-root/--csv-manifest required (or set FLAIR3D_DATA_ROOT / "
                "FLAIR3D_CSV_MANIFEST)")

    root_cause_found = False
    if args.check_npy_counts:
        root_cause_found = check_npy_counts(
            args.data_root, args.csv_manifest, args.tile_width, args.subtile_width,
            args.split, args.limit,
        )

    transform = None
    if args.transform != "none":
        print(f"\nComposing '{args.transform}' transform from experiment={args.experiment!r} ...")
        transform = build_transform(
            args.experiment, args.transform, args.data_root, args.csv_manifest
        )
        print(f"  pipeline: {[type(t).__name__ for t in transform.transforms]}")

    dataset = PointceptNpyDataset(
        data_root=args.data_root,
        csv_manifest=args.csv_manifest,
        tile_width=args.tile_width,
        subtile_width=args.subtile_width,
        # Same transform on both hooks so whichever split we scan gets it.
        train_transform=transform,
        eval_transform=transform,
    )
    indices = dataset._indices_for_split(args.split)
    if args.limit:
        indices = indices[: args.limit]
    print(f"\nScanning {len(indices)} scenes from split={args.split!r} "
          f"(transform={args.transform}) ...")

    combos_seen = defaultdict(set)  # key -> {(dtype_label, shape_label)}
    keys_seen = defaultdict(int)
    examples = defaultdict(lambda: defaultdict(list))  # key -> combo -> [scene_dir]
    first_data_by_combo = {}  # (key, combo) -> Data (kept for a real collate attempt)

    for i, idx in enumerate(indices):
        data = dataset[idx]
        if data is None:
            continue
        scene_dir = dataset._scenes[idx][0]
        for key in data.keys():
            value = data[key]
            keys_seen[key] += 1
            combo = (_dtype_label(value), _shape_label(value))
            combos_seen[key].add(combo)
            if len(examples[key][combo]) < 5:
                examples[key][combo].append(scene_dir)
            first_data_by_combo.setdefault((key, combo), data)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(indices)}")

    print("\n=== Per-key (dtype, trailing-shape) observed  [MISMATCH if >1] ===")
    mismatched_keys = []
    for key in sorted(combos_seen):
        combos = sorted(combos_seen[key])
        flag = "   <-- MISMATCH" if len(combos) > 1 else ""
        if len(combos) > 1:
            mismatched_keys.append(key)
        print(f"{key:28s} in {keys_seen[key]:6d}/{len(indices)}  {combos}{flag}")

    if mismatched_keys:
        print("\n=== Example scenes per mismatched key ===")
        for key in mismatched_keys:
            print(f"\n{key}:")
            for combo in sorted(combos_seen[key]):
                print(f"  {combo}")
                for scene_dir in examples[key][combo]:
                    print(f"      {scene_dir}")

        # Reproduce the actual worker-side collate RuntimeError for the first mismatched
        # key. PyG only allocates the pre-typed `out=` buffer (from values[0]'s dtype)
        # inside a DataLoader worker -- in the main process torch.cat silently upcasts --
        # so emulate that buffer here. The crash depends on batch order (which scene is
        # values[0]), so both orders are tried.
        for key in mismatched_keys:
            tensors = []
            for c in sorted(combos_seen[key]):
                t = first_data_by_combo[(key, c)][key]
                if isinstance(t, torch.Tensor):
                    tensors.append(t.reshape(-1))
            if len(tensors) < 2:
                continue
            a, b = tensors[0], tensors[1]
            print(f"\n=== Worker-collate repro for key {key!r} ===")
            for lead, trail in ((a, b), (b, a)):
                try:
                    out = torch.empty(lead.numel() + trail.numel(), dtype=lead.dtype)
                    torch.cat([lead, trail], out=out)
                    print(f"  values[0] dtype {str(lead.dtype):13s} -> ok")
                except RuntimeError as e:
                    print(f"  values[0] dtype {str(lead.dtype):13s} -> RuntimeError: {e}")
                    print("    ^ this is the DataLoader worker crash from Jean Zay.")
            break
    elif not root_cause_found:
        print("\nNo dtype/shape mismatch in the scanned scenes. Try --limit 0, a different "
              "--split, or --transform train to apply the real pipeline.")


if __name__ == "__main__":
    main()
