"""Scan a PointceptNpyDataset for per-attribute dtype inconsistencies across scenes.

A PyG `Batch.from_data_list` collate error such as
``RuntimeError: torch.cat(): input types can't be cast to the desired output type Long``
means some `Data` attribute has a different dtype on different scenes in the same
batch. This walks every scene in a split and reports, per attribute key, every dtype
observed and a few example offending scene directories.

Usage:
  python scripts/debug_scene_dtypes.py \
    --data-root /path/to/flair3d_plus \
    --csv-manifest /path/to/flair3d_plus/raw/scene_split_manifest.csv \
    --split train --limit 0
"""

import argparse
from collections import defaultdict

import torch

from myria3d.pctl.dataset.pointcept_npy import PointceptNpyDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--csv-manifest", required=True)
    p.add_argument("--tile-width", type=float, default=100)
    p.add_argument("--subtile-width", type=float, default=50)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=500, help="max scenes to scan (0 = all)")
    args = p.parse_args()

    dataset = PointceptNpyDataset(
        data_root=args.data_root,
        csv_manifest=args.csv_manifest,
        tile_width=args.tile_width,
        subtile_width=args.subtile_width,
        train_transform=None,
        eval_transform=None,
    )
    indices = dataset._indices_for_split(args.split)
    if args.limit:
        indices = indices[: args.limit]
    print(f"Scanning {len(indices)} scenes from split={args.split!r} ...")

    dtypes_seen = defaultdict(set)
    keys_seen = defaultdict(int)
    bad_examples = defaultdict(list)

    for i, idx in enumerate(indices):
        data = dataset[idx]
        if data is None:
            continue
        scene_dir = dataset._scenes[idx][0]
        for key in data.keys():
            value = data[key]
            keys_seen[key] += 1
            if isinstance(value, torch.Tensor):
                dt = str(value.dtype)
            elif hasattr(value, "dtype"):
                dt = f"numpy:{value.dtype}"
            else:
                dt = type(value).__name__
            if dtypes_seen[key] and dt not in dtypes_seen[key]:
                bad_examples[key].append((scene_dir, dt))
            dtypes_seen[key].add(dt)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(indices)}")

    print("\n=== Per-key dtypes observed (flag if >1) ===")
    any_bad = False
    for key in sorted(dtypes_seen):
        dts = dtypes_seen[key]
        n_present = keys_seen[key]
        flag = " <-- MISMATCH" if len(dts) > 1 else ""
        if len(dts) > 1:
            any_bad = True
        print(
            f"{key:30s} present in {n_present:5d}/{len(indices)} scenes  "
            f"dtypes={sorted(dts)}{flag}"
        )

    if any_bad:
        print("\n=== Example offending scenes (first few per mismatched key) ===")
        for key, examples in bad_examples.items():
            print(f"\n{key}:")
            for scene_dir, dt in examples[:5]:
                print(f"  {dt:20s} {scene_dir}")
    else:
        print(
            "\nNo dtype mismatch found in the scanned scenes. Try --limit 0 (full scan) "
            "or a different --split."
        )


if __name__ == "__main__":
    main()
