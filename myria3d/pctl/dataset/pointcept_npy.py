"""Load Flair3D+ scenes preprocessed by Pointcept (.npy folders) for myria3d training."""

from __future__ import annotations

import csv
import os
import os.path as osp
from numbers import Number
from typing import Callable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from myria3d.pctl.dataset.flair3d import load_excluded_tiles_from_details_csv
from myria3d.pctl.dataset.flair3d_label_remap import apply_remap, get_definition
from myria3d.pctl.dataset.utils import (
    SPLIT_TYPE,
    get_num_subtiles,
    pre_filter_below_n_points,
)
from myria3d.utils import utils

log = utils.get_logger(__name__)

SceneEntry = Tuple[str, SPLIT_TYPE, Optional[int]]

REQUIRED_ASSETS = ("coord",)
MULTITASK_TARGET_FILES = (
    ("segment", "y"),
    ("forest", "y_forest"),
    ("land_use", "y_land_use"),
    ("natural_habitat", "y_natural_habitat"),
    ("elevation", "y_elevation"),
)
# Void / missing fills aligned with configs/dataset_description/flair3d_plus_multitask.yaml
# and flair3d.py raster preprocessing (missing GeoTIFF → all ignore_index).
MULTITASK_MISSING_FILLS = {
    "y": 15,
    "y_forest": 2,
    "y_land_use": 10,
    "y_natural_habitat": 11,
}


def load_too_small_tiles_from_csv(csv_path: Optional[str]) -> Set[Tuple[str, str]]:
    """Load (split, patch_id) train tiles flagged as too small (train split only)."""
    if not csv_path:
        return set()
    if not osp.isfile(csv_path):
        log.warning(
            "Too-small tiles CSV not found: %s. Continuing with empty set.",
            csv_path,
        )
        return set()

    too_small: Set[Tuple[str, str]] = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get("split") or "").strip().lower()
            patch_id = (row.get("patch_id") or "").strip()
            if split == "train" and patch_id:
                too_small.add((split, patch_id))
    return too_small


def build_scene_list(
    data_root: str,
    csv_manifest: str,
    excluded_tiles_details_csv: Optional[str] = None,
    too_small_tiles_manifest: Optional[str] = None,
    splits: Sequence[str] = ("train", "val", "test"),
    tile_width: Number = 100,
    subtile_width: Number = 50,
    subtile_overlap: Number = 0,
) -> List[SceneEntry]:
    """Build scene directories from a Pointcept-style split manifest CSV.

    Train scenes yield one entry (random subtile at transform time). Val/test scenes
    yield one entry per mosaic subtile (subtile_index 0..N-1).
    """
    eval_subtiles_per_scene = get_num_subtiles(
        tile_width, subtile_width, subtile_overlap=subtile_overlap
    )
    if not osp.isfile(csv_manifest):
        raise FileNotFoundError(f"CSV manifest not found: {csv_manifest}")

    splits_set = set(splits)
    excluded = load_excluded_tiles_from_details_csv(excluded_tiles_details_csv)
    too_small = load_too_small_tiles_from_csv(too_small_tiles_manifest)
    excluded_tiles = excluded | too_small

    scenes: List[SceneEntry] = []
    skipped_missing_coord = 0
    with open(csv_manifest, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get("split") or "").strip().lower()
            patch_id = (row.get("patch_id") or "").strip()
            if split not in splits_set or row.get("LIDARHD") != "True":
                continue
            if (split, patch_id) in excluded_tiles:
                continue

            dept_year = row.get("dept_year") or patch_id.split("_")[0]
            roi = row.get("roi") or patch_id.split("_")[1]
            scene_dir = osp.join(
                data_root, split, f"{dept_year}_LIDARHD", roi, patch_id
            )
            coord_path = osp.join(scene_dir, "coord.npy")
            if not osp.isfile(coord_path):
                skipped_missing_coord += 1
                continue
            if split == "train":
                scenes.append((scene_dir, split, None))
            else:
                for subtile_index in range(eval_subtiles_per_scene):
                    scenes.append((scene_dir, split, subtile_index))

    log.info(
        "PointceptNpy: %d dataset entries from manifest (skipped %d without coord.npy).",
        len(scenes),
        skipped_missing_coord,
    )
    return scenes


def _build_feature_matrix(
    color: np.ndarray, strength: Optional[np.ndarray], num_points: int
) -> Tuple[np.ndarray, List[str]]:
    """Build x features matching Flair3D+ _flair3d_feature_tensors (RGB 0-255, no /255)."""
    if strength is not None:
        intensity = strength.astype(np.float32, copy=False)
    else:
        intensity = np.zeros(num_points, dtype=np.float32)

    if color is not None and color.shape[0] == num_points:
        red = color[:, 0].astype(np.float32, copy=False)
        green = color[:, 1].astype(np.float32, copy=False)
        blue = color[:, 2].astype(np.float32, copy=False)
        rgb_avg = (red + green + blue) / 3.0
    else:
        red = green = blue = rgb_avg = np.zeros(num_points, dtype=np.float32)

    x = np.stack(
        [intensity, red, green, blue, rgb_avg],
        axis=1,
    ).astype(np.float32, copy=False)
    x_features_names = ["Intensity", "Red", "Green", "Blue", "rgb_avg"]
    return x, x_features_names


def load_pointcept_scene(scene_dir: str) -> Data:
    """Load one Pointcept-preprocessed scene folder into a PyG Data object."""
    coord_path = osp.join(scene_dir, "coord.npy")
    if not osp.isfile(coord_path):
        raise FileNotFoundError(f"Missing coord.npy in {scene_dir}")

    pos = np.load(coord_path).astype(np.float32, copy=False)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"coord.npy must be (N, 3), got {pos.shape} in {scene_dir}")
    num_points = pos.shape[0]

    color_path = osp.join(scene_dir, "color.npy")
    color = np.load(color_path) if osp.isfile(color_path) else None

    strength_path = osp.join(scene_dir, "strength.npy")
    strength = np.load(strength_path) if osp.isfile(strength_path) else None

    x, x_features_names = _build_feature_matrix(color, strength, num_points)

    kwargs = dict(
        pos=torch.from_numpy(pos),
        x=torch.from_numpy(x),
        x_features_names=x_features_names,
        idx_in_original_cloud=np.arange(num_points, dtype=np.int32),
    )

    for asset_name, data_key in MULTITASK_TARGET_FILES:
        asset_path = osp.join(scene_dir, f"{asset_name}.npy")
        if osp.isfile(asset_path):
            values = np.load(asset_path).reshape(-1)
            if data_key == "y_elevation":
                kwargs[data_key] = torch.from_numpy(values.astype(np.float32, copy=False))
            elif data_key == "y":
                remapped = apply_remap(
                    values.astype(np.int64, copy=False),
                    get_definition("segment", "default"),
                )
                kwargs[data_key] = torch.from_numpy(remapped.astype(np.int64, copy=False))
            else:
                kwargs[data_key] = torch.from_numpy(values.astype(np.int64, copy=False))
        elif data_key == "y_elevation":
            kwargs[data_key] = torch.full(
                (num_points,), float("nan"), dtype=torch.float32
            )
        elif data_key in MULTITASK_MISSING_FILLS:
            kwargs[data_key] = torch.full(
                (num_points,),
                MULTITASK_MISSING_FILLS[data_key],
                dtype=torch.int64,
            )

    return Data(**kwargs)


class PointceptNpyDataset(Dataset):
    """Dataset over Pointcept-preprocessed Flair3D+ .npy scene folders."""

    def __init__(
        self,
        data_root: str,
        csv_manifest: str,
        excluded_tiles_details_csv: Optional[str] = None,
        too_small_tiles_manifest: Optional[str] = None,
        tile_width: Number = 100,
        subtile_width: Number = 50,
        subtile_overlap: Number = 0,
        pre_filter: Callable[[Data], bool] = pre_filter_below_n_points,
        train_transform: Optional[List[Callable]] = None,
        eval_transform: Optional[List[Callable]] = None,
    ):
        self.pre_filter = pre_filter
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self._scenes = build_scene_list(
            data_root=data_root,
            csv_manifest=csv_manifest,
            excluded_tiles_details_csv=excluded_tiles_details_csv,
            too_small_tiles_manifest=too_small_tiles_manifest,
            tile_width=tile_width,
            subtile_width=subtile_width,
            subtile_overlap=subtile_overlap,
        )

    def __len__(self) -> int:
        return len(self._scenes)

    def __getitem__(self, idx: int) -> Optional[Data]:
        scene_dir, split, subtile_index = self._scenes[idx]
        data = load_pointcept_scene(scene_dir)

        if self.pre_filter and self.pre_filter(data):
            return None

        if subtile_index is not None:
            data.subtile_index = subtile_index

        transform = self.train_transform
        if split in ("val", "test"):
            transform = self.eval_transform
        if transform:
            data = transform(data)

        if not data or (self.pre_filter and self.pre_filter(data)):
            return None

        return data

    def _indices_for_split(self, split: SPLIT_TYPE) -> List[int]:
        return [
            idx
            for idx, (_, scene_split, _) in enumerate(self._scenes)
            if scene_split == split
        ]

    @property
    def traindata(self):
        return torch.utils.data.Subset(self, self._indices_for_split("train"))

    @property
    def valdata(self):
        return torch.utils.data.Subset(self, self._indices_for_split("val"))

    @property
    def testdata(self):
        return torch.utils.data.Subset(self, self._indices_for_split("test"))
