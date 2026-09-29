#!/usr/bin/env python3
"""Extract per-tile mean/max-pooled frozen-backbone hypercolumn features on PureForest.

Plain script, not a Hydra task: PureForest's actual linear probe is trained back in
the sibling Pointcept repo (on Hecate), via its existing
``scripts/probe_pureforest_sklearn.py`` -- this script only extracts and saves
features, in exactly Pointcept's ``{split}.npz`` schema, so the ``.npz`` files can be
copied there and read unmodified (pass ``--channel-blocks 32 128 256 512`` explicitly
on the Pointcept side; myria3d has no ``Config.fromfile``-loadable config for that
script's usual meta.json auto-resolution path). See readme_linear_probing.md.

For each tile: one frozen-backbone forward (deterministic, no augmentation) ->
``_forward_encoder_stages`` + ``hypercolumn_concat`` -> per-point hypercolumn feature
-> mean pool (float32 accumulation, nan_to_num-safe) and max pool (native dtype),
both cast to float16 for storage.

Usage::

    python scripts/extract_pureforest_pooled_features.py \\
        --ckpt-path /path/to/flair3d_plus_multitask.ckpt \\
        --data-root data/pureforest \\
        --output-dir stats/pureforest_embeddings \\
        --splits train val test --batch-size 8 --num-workers 6 --point-budget 600000

Async loading + point-budget batching (ported from the sibling Pointcept repo's
``scripts/extract_pureforest_pooled_embeddings.py``, which needed the same fix): tile
load + transform (disk read, GridSampling voxelization) runs in ``--num-workers``
DataLoader worker processes so it overlaps with the previous batch's GPU forward,
instead of running serially in the main process between forwards. Real PureForest
tiles vary ~10x in point count and are listed alphabetically (clustering same-forest
tiles together), so a fixed ``--batch-size`` can silently pack several of a split's
densest tiles into one batch and OOM even after a smaller sample looked safe -- pass
``--point-budget`` to pack batches by total raw point count instead (First-Fit-
Decreasing; ``--batch-size`` then becomes the packer's max-tiles-per-batch cap).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Center
from torch_scatter import scatter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch_geometric.transforms import GridSampling  # noqa: E402

from myria3d.models.grid_probe_model import load_frozen_backbone  # noqa: E402
from myria3d.models.modules.hypercolumn import (  # noqa: E402
    DEFAULT_SCALES,
    STAGE_CHANNELS,
    hypercolumn_concat,
)
from myria3d.pctl.dataset.downstream.pureforest import (  # noqa: E402
    CLASS_NAMES,
    PureForestDataset,
)
from myria3d.pctl.transforms.compose import CustomCompose  # noqa: E402
from myria3d.pctl.transforms.transforms import (  # noqa: E402
    NormalizePos,
    NullifyLowestZ,
    StandardizeRGBAndIntensity,
    resolve_x_feature_names,
)

CHANNEL_BLOCKS = tuple(STAGE_CHANNELS[s] for s in DEFAULT_SCALES)
FEAT_DIM = sum(CHANNEL_BLOCKS)


def build_extract_transform(voxel: float) -> CustomCompose:
    """Deterministic, augmentation-free pipeline matching how the backbone was
    pretrained (Center/NullifyLowestZ/NormalizePos/StandardizeRGBAndIntensity, same
    voxel size as `configs/datamodule/transforms/preparations/points_budget_downstream.yaml`).
    PureForest tiles are already ~50 m single plots (see readme_linear_probing.md) --
    no SubtileCrop needed."""
    return CustomCompose(
        [
            GridSampling(voxel),
            Center(),
            NullifyLowestZ(),
            NormalizePos(subtile_width=50),
            StandardizeRGBAndIntensity(),
        ]
    )


def fill_strength_mask_value(
    x: torch.Tensor, names: List[str], strength_mask_value
) -> torch.Tensor:
    """PureForest has no intensity asset at all (`strength_mask` is all-True, see
    `load_pureforest_scene`) -- every point's Intensity column must come from the
    frozen backbone's own learned fill-in value, not a hand-picked zero. Mirrors
    `GridProbeModel._fill_masked_features`'s strength-only case."""
    if strength_mask_value is None or "Intensity" not in names:
        return x
    idx = names.index("Intensity")
    fill = strength_mask_value.to(dtype=x.dtype, device=x.device)[:, 0].reshape(())
    out = x.clone()
    out[:, idx] = fill
    return out


@torch.no_grad()
def extract_tile_features(backbone, batch: Batch, mask_values: dict) -> torch.Tensor:
    names = resolve_x_feature_names(batch)
    x = fill_strength_mask_value(batch.x, names, mask_values.get("strength_mask_value"))
    stages = backbone._forward_encoder_stages(x, batch.pos, batch.batch, batch.ptr)
    return hypercolumn_concat(stages, batch.pos, batch.batch, scales=DEFAULT_SCALES, k=1)


def pool_mean_max(feat: torch.Tensor, batch_index: torch.Tensor, num_graphs: int):
    mean_feat = scatter(feat.float(), batch_index, dim=0, dim_size=num_graphs, reduce="mean")
    mean_feat = torch.nan_to_num(mean_feat, nan=0.0, posinf=0.0, neginf=0.0)
    max_feat = scatter(feat, batch_index, dim=0, dim_size=num_graphs, reduce="max")
    return mean_feat, max_feat


class _TileDataset(torch.utils.data.Dataset):
    """Wraps `PureForestDataset[idx]` + `transform` so a DataLoader's worker
    processes can do the CPU-bound part (disk read, GridSampling voxelization)
    ahead of time, overlapping it with the previous batch's GPU forward instead of
    running it serially in the main process between forwards."""

    def __init__(self, dataset: PureForestDataset, transform: CustomCompose):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        raw = self.dataset[idx]
        name = raw.patch_id
        data = self.transform(raw)
        if data is None or data.num_nodes == 0:
            print(f"[extract] skipping empty tile after transform: {name}")
            return None
        return data, name, int(data.category)


def _extract_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    data_list, names, cats = zip(*batch)
    return Batch.from_data_list(list(data_list)), list(names), list(cats)


def _compute_raw_sizes(dataset: PureForestDataset) -> List[int]:
    """Per-tile raw point count, read from `coord.npy`'s header via mmap (no full
    load) -- a safe upper bound for the post-GridSample point count (GridSample(0.1)
    only removes a modest fraction of points), used only to pack batches by point
    budget below."""
    sizes = []
    for name in dataset.names:
        coord_path = Path(dataset.data_root) / dataset.split / name / "coord.npy"
        sizes.append(int(np.load(coord_path, mmap_mode="r").shape[0]))
    return sizes


def pack_indices_by_point_budget(
    sizes: List[int], point_budget: int, max_batch_size: int
) -> List[List[int]]:
    """First-Fit-Decreasing bin packing by point count, so a batch's total point
    count stays under `point_budget` regardless of how densely tiles cluster in
    dataset order. Oversized single tiles become singleton batches. Ported from the
    sibling Pointcept repo's `pointcept/datasets/utils.py::pack_indices_by_voxel_budget`
    -- PureForest tiles vary ~10x in point count and `PureForestDataset` lists them
    alphabetically (clustering same-forest tiles together), so a fixed tile-count
    batch can silently pack several of a split's densest tiles into one batch and OOM
    even after a smaller sample looked safe.
    See: https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing
    """
    if point_budget <= 0:
        raise ValueError(f"point_budget must be > 0, got {point_budget}")
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be > 0, got {max_batch_size}")

    indexed = sorted(enumerate(sizes), key=lambda item: (-int(item[1]), item[0]))
    batches: List[List[int]] = []
    batch_sums: List[int] = []
    for index, size in indexed:
        size = int(size)
        if size > point_budget:
            batches.append([index])
            batch_sums.append(size)
            continue
        placed = False
        for b, (batch, batch_sum) in enumerate(zip(batches, batch_sums)):
            if len(batch) >= max_batch_size:
                continue
            if batch_sum + size > point_budget:
                continue
            batch.append(index)
            batch_sums[b] = batch_sum + size
            placed = True
            break
        if not placed:
            batches.append([index])
            batch_sums.append(size)
    return batches


def run_split(
    *,
    backbone,
    mask_values: dict,
    dataset: PureForestDataset,
    transform: CustomCompose,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    point_budget: int = None,
) -> dict:
    names: List[str] = []
    categories: List[int] = []
    mean_chunks: List[np.ndarray] = []
    max_chunks: List[np.ndarray] = []

    n = len(dataset)
    tile_dataset = _TileDataset(dataset, transform)
    loader_kwargs = dict(
        collate_fn=_extract_collate,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    if point_budget is not None:
        sizes = _compute_raw_sizes(dataset)
        loader_kwargs["batch_sampler"] = pack_indices_by_point_budget(
            sizes, point_budget, max_batch_size=batch_size
        )
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = False

    loader = torch.utils.data.DataLoader(tile_dataset, **loader_kwargs)

    processed = 0
    for out in loader:
        if out is None:
            continue
        batch, batch_names, batch_cats = out
        batch = batch.to(device)
        feat = extract_tile_features(backbone, batch, mask_values)
        mean_feat, max_feat = pool_mean_max(feat, batch.batch, len(batch_names))

        mean_chunks.append(mean_feat.cpu().numpy().astype(np.float16))
        max_chunks.append(max_feat.cpu().numpy().astype(np.float16))
        names.extend(batch_names)
        categories.extend(batch_cats)
        processed += len(batch_names)
        print(f"[extract] {dataset.split}: {processed}/{n}")

    return {
        "names": np.asarray(names),
        "category": np.asarray(categories, dtype=np.int64),
        "mean_feat": np.concatenate(mean_chunks, axis=0),
        "max_feat": np.concatenate(max_chunks, axis=0),
    }


def save_split_npz(output_path: Path, payload: dict, meta: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        names=payload["names"],
        category=payload["category"],
        mean_feat=payload["mean_feat"],
        max_feat=payload["max_feat"],
        class_names=np.asarray(CLASS_NAMES),
        # meta may also carry "class_names" (e.g. main()'s meta_common, also reused
        # for meta.json) -- excluded here to avoid a duplicate-keyword TypeError.
        **{k: np.asarray(v) for k, v in meta.items() if k != "class_names"},
    )
    size_mb = output_path.stat().st_size / 2**20
    n, c = payload["mean_feat"].shape
    print(f"[extract] wrote {output_path}  ({n:,} tiles x {c}ch mean/max, {size_mb:.1f} MB)")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt-path", required=True, help="Flair3D+ multitask checkpoint.")
    parser.add_argument("--data-root", required=True, help="e.g. data/pureforest")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for {split}.npz + meta.json"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Fixed tiles/batch (default), or the max tiles/batch cap when "
        "--point-budget is set.",
    )
    parser.add_argument(
        "--point-budget",
        type=int,
        default=None,
        help="Pack batches by total raw point count instead of a fixed tile count "
        "(First-Fit-Decreasing); --batch-size becomes the packer's max-tiles-per-batch "
        "cap. Strongly recommended for the real (non-toy) dataset -- PureForest tiles "
        "vary ~10x in point count and are listed alphabetically (clustering "
        "same-forest tiles together), so a fixed --batch-size can OOM on an unlucky "
        "dense cluster even if it survived a smaller sample.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="DataLoader workers for async load+transform (0 = main-process, "
        "synchronous). Overlaps disk I/O + GridSampling voxelization with the "
        "previous batch's GPU forward.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per worker (only used when --num-workers > 0).",
    )
    parser.add_argument("--voxel", type=float, default=0.1)
    parser.add_argument("--num-features", type=int, default=5)
    parser.add_argument("--num-neighbors", type=int, default=16)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    parser.add_argument("--max-tiles", type=int, default=None, help="Smoke-test cap per split.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[extract] device={device}  ckpt={args.ckpt_path}  channel_blocks={CHANNEL_BLOCKS}")

    backbone, mask_values = load_frozen_backbone(
        args.ckpt_path,
        num_features=args.num_features,
        num_neighbors=args.num_neighbors,
        decimation=args.decimation,
    )
    backbone = backbone.to(device)

    transform = build_extract_transform(args.voxel)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_common = {
        # "weight" (not "ckpt_path"): matches Pointcept's own
        # extract_pureforest_pooled_embeddings.py metadata key name.
        "weight": str(args.ckpt_path),
        "data_root": str(args.data_root),
        "feat_dim": FEAT_DIM,
        "channel_blocks": list(CHANNEL_BLOCKS),
        "voxel": args.voxel,
        "class_names": CLASS_NAMES,
    }

    for split in args.splits:
        dataset = PureForestDataset(
            args.data_root,
            split,
            # Test-only leakage filter (fact 9): drop test tiles whose forest polygon
            # overlaps a Flair3D+ train/val tile, never applied to train/val -- matches
            # every Pointcept PureForest config (`exclude_flair3d_leakage_tiles=True`
            # set only on `data.test`).
            exclude_flair3d_leakage_tiles=(split == "test"),
        )
        if args.max_tiles is not None:
            dataset.names = dataset.names[: int(args.max_tiles)]
            print(f"[extract] capped {split} at {len(dataset.names)} tiles")

        payload = run_split(
            backbone=backbone,
            mask_values=mask_values,
            dataset=dataset,
            transform=transform,
            batch_size=max(1, int(args.batch_size)),
            device=device,
            num_workers=max(0, int(args.num_workers)),
            prefetch_factor=max(1, int(args.prefetch_factor)),
            point_budget=None if args.point_budget is None else int(args.point_budget),
        )
        save_split_npz(
            output_dir / f"{split}.npz",
            payload,
            {**meta_common, "split": split, "num_tiles": int(payload["category"].shape[0])},
        )

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({**meta_common, "splits": list(args.splits)}, f, indent=2)
    print(f"[extract] wrote {meta_path}")


if __name__ == "__main__":
    main()
