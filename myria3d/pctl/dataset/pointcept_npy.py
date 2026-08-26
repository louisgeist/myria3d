"""Load Flair3D+ scenes preprocessed by Pointcept (.npy folders) for myria3d training."""

from __future__ import annotations

import csv
import json
import os
import os.path as osp
from numbers import Number
from typing import Callable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from myria3d.pctl.dataset.flair3d import load_excluded_tiles_from_details_csv
from myria3d.pctl.dataset.flair3d_label_remap import (
    NATURAL_HABITAT_AXIS_DEFINITIONS,
    apply_remap,
    get_definition,
)
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
    ("elevation", "y_elevation"),
)
# Void / missing fills aligned with configs/dataset_description/flair3d_plus_multitask.yaml.
MULTITASK_MISSING_FILLS = {
    "y": 15,
}

# Rasters preprocessed by Pointcept's own rasterize_forest.py / rasterize_network.py
# scripts (run separately, outside myria3d): (npy filename stem, meta.json key, myria3d
# task name, raster channel name or None for a single-channel raster). Loaded as
# pixel_semantic targets — see MultiTaskModel / pixel_pooling.py for how per-point
# predictions get pooled back up to raster-cell resolution.
PIXEL_SEMANTIC_TARGET_FILES = (
    ("forest_2d", "forest_2d", "forest_2d", None),
    ("network", "network", "roads", "ROADS"),
)
# ignore_index per pixel_semantic task, aligned with configs/dataset_description/flair3d_plus_multitask.yaml.
PIXEL_SEMANTIC_MISSING_FILLS = {
    "forest_2d": 2,
    "roads": 2,
}

# Raw CarHab id used to fill natural_habitat when natural_habitat.npy is absent — void in
# every axis LUT (see flair3d_label_remap.py), matching Pointcept's own missing-fill sentinel.
NATURAL_HABITAT_MISSING_FILL_RAW_ID = 43


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


COLOR_NORMALIZATION_MAX_VALUE = 255.0


def _build_feature_matrix(
    color: np.ndarray, strength: Optional[np.ndarray], num_points: int
) -> Tuple[np.ndarray, List[str]]:
    """Build x features matching myria3d's classic RandLA-Net convention (lidar_hd_pre_transform):
    color channels scaled to ~[0, 1], not left at raw 0-255 (Pointcept's `color.npy` is 8-bit,
    hence /255 here vs lidar_hd_pre_transform's /65280 for 16-bit Lidar-HD colors)."""
    if strength is not None:
        intensity = strength.astype(np.float32, copy=False)
    else:
        intensity = np.zeros(num_points, dtype=np.float32)

    if color is not None and color.shape[0] == num_points:
        red = color[:, 0].astype(np.float32, copy=False) / COLOR_NORMALIZATION_MAX_VALUE
        green = color[:, 1].astype(np.float32, copy=False) / COLOR_NORMALIZATION_MAX_VALUE
        blue = color[:, 2].astype(np.float32, copy=False) / COLOR_NORMALIZATION_MAX_VALUE
        rgb_avg = (red + green + blue) / 3.0
    else:
        red = green = blue = rgb_avg = np.zeros(num_points, dtype=np.float32)

    x = np.stack(
        [intensity, red, green, blue, rgb_avg],
        axis=1,
    ).astype(np.float32, copy=False)
    x_features_names = ["Intensity", "Red", "Green", "Blue", "rgb_avg"]
    return x, x_features_names


def _load_scene_meta(scene_dir: str) -> dict:
    meta_path = osp.join(scene_dir, "meta.json")
    if not osp.isfile(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _assign_raster_cells(
    pos_xy: np.ndarray,
    raster_2d: np.ndarray,
    raster_meta: dict,
    ignore_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map absolute Lambert (x, y) coordinates to a raster cell id + that cell's label.

    `raster_2d` is a single-channel (H, W) array. Points outside the raster grid (or
    when the raster/meta is entirely absent, via a zero-size raster_2d) get cell_id=-1
    and label=ignore_index — pool_points_by_cell drops cell_id=-1 points.
    """
    num_points = pos_xy.shape[0]
    cell_id = np.full(num_points, -1, dtype=np.int64)
    label = np.full(num_points, ignore_index, dtype=np.int64)
    if raster_2d.size == 0:
        return cell_id, label

    origin_x = float(raster_meta["origin_x"])
    origin_y = float(raster_meta["origin_y"])
    pixel_m = float(raster_meta["pixel_m"])
    height, width = raster_2d.shape

    col = np.floor((pos_xy[:, 0] - origin_x) / pixel_m).astype(np.int64)
    row = np.floor((pos_xy[:, 1] - origin_y) / pixel_m).astype(np.int64)
    in_grid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    if np.any(in_grid):
        r, c = row[in_grid], col[in_grid]
        cell_id[in_grid] = r * width + c
        label[in_grid] = raster_2d[r, c]
    return cell_id, label


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
            else:
                # segment.npy is written by Pointcept's own preprocessing (already
                # remapped to train ids via its "v20" LUT, which is a pass-through for
                # 0-15 with 15=Void) — do not re-apply a myria3d-side segment remap
                # here, or genuinely-Void points get silently relabeled Building(0).
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

    # Pointcept stores local `coord.npy` (float32) plus `coord_translation.npy`
    # (float64 Lambert offset). Raster origins in meta.json are absolute, so cell
    # assignment must reconstruct abs_xy the same way as Pointcept's ExtractAbsXY.
    pos_xy = pos[:, :2].astype(np.float64, copy=True)
    transl_path = osp.join(scene_dir, "coord_translation.npy")
    if osp.isfile(transl_path):
        transl = np.load(transl_path)
        if transl.shape[-1] < 2:
            raise ValueError(
                f"coord_translation.npy must have at least 2 values, got {transl.shape} "
                f"in {scene_dir}"
            )
        pos_xy = pos_xy + np.asarray(transl, dtype=np.float64).reshape(-1)[:2]
    scene_meta = None  # lazy-loaded, only needed once a raster asset is actually found
    for asset_name, meta_key, task_name, channel_name in PIXEL_SEMANTIC_TARGET_FILES:
        asset_path = osp.join(scene_dir, f"{asset_name}.npy")
        ignore_index = PIXEL_SEMANTIC_MISSING_FILLS[task_name]
        if osp.isfile(asset_path):
            if scene_meta is None:
                scene_meta = _load_scene_meta(scene_dir)
            if meta_key not in scene_meta:
                raise ValueError(
                    f"{asset_name}.npy exists in {scene_dir} but meta.json is missing "
                    f"its '{meta_key}' raster-geometry entry."
                )
            raster_meta = scene_meta[meta_key]
            raster = np.load(asset_path)
            raster_2d = (
                raster[raster_meta["channel_order"].index(channel_name)]
                if channel_name is not None
                else raster[0]
            )
            cell_id, label = _assign_raster_cells(pos_xy, raster_2d, raster_meta, ignore_index)
            raster_h, raster_w = int(raster_2d.shape[0]), int(raster_2d.shape[1])
        else:
            cell_id = np.full(num_points, -1, dtype=np.int64)
            label = np.full(num_points, ignore_index, dtype=np.int64)
            raster_h, raster_w = 0, 0
        kwargs[f"{task_name}_cell_id"] = torch.from_numpy(cell_id)
        kwargs[f"y_{task_name}"] = torch.from_numpy(label)
        # Graph-level (1,) so GridSampling / subsample_data leave them alone.
        kwargs[f"{task_name}_raster_h"] = torch.tensor([raster_h], dtype=torch.long)
        kwargs[f"{task_name}_raster_w"] = torch.tensor([raster_w], dtype=torch.long)

    # natural_habitat.npy stores raw (near-raw) CarHab ids; remapped here into 4
    # low-cardinality ecological axes (tile_distribution targets), never used directly.
    nh_path = osp.join(scene_dir, "natural_habitat.npy")
    if osp.isfile(nh_path):
        nh_raw = np.load(nh_path).reshape(-1).astype(np.int64, copy=False)
    else:
        nh_raw = np.full(num_points, NATURAL_HABITAT_MISSING_FILL_RAW_ID, dtype=np.int64)
    for task_name, definition_name in NATURAL_HABITAT_AXIS_DEFINITIONS.items():
        remapped = apply_remap(nh_raw, get_definition("natural_habitat", definition_name))
        kwargs[f"y_{task_name}"] = torch.from_numpy(remapped.astype(np.int64, copy=False))

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
        # Pointcept PreciseEvaluator file stem is the 100 m patch folder name.
        patch_id = osp.basename(osp.normpath(scene_dir))
        data.patch_id = patch_id

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

        # Re-attach in case a transform dropped the python string attribute.
        data.patch_id = patch_id
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
