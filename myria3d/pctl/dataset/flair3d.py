import argparse
import csv
import os
import os.path as osp
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.lib.recfunctions import append_fields
import h5py
import torch
from torch_geometric.data import Data
import pandas as pd
from myria3d.pctl.dataset.flair3d_label_remap import (
    apply_remap,
    build_preprocess_label_definitions,
)
from myria3d.pctl.dataset.raster_utils import (
    build_modality_patch_path,
    parse_ply_patch_metadata,
    sample_raster_to_points_float,
)
from myria3d.pctl.points_pre_transform.lidar_hd import lidar_hd_pre_transform
from myria3d.utils import utils

log = utils.get_logger(__name__)

from myria3d.pctl.dataset.hdf5 import HDF5Dataset

MULTITASK_TARGET_KEYS = (
    "y",
    "y_elevation",
)


def _parse_csv_bool(raw: str) -> bool:
    token = (raw or "").strip().lower()
    return token in ("true", "1", "yes")


def build_ply_path(labels_root: str, dept_year: str, roi: str, scene_i_j: str) -> str:
    """Resolve a Flair3D-build output PLY path (same layout for label=v12, v14, v20, …)."""
    ply_filename = f"{dept_year}_LIDARHD_{roi}_{scene_i_j}.ply"
    return osp.abspath(
        osp.join(labels_root, "LIDARHD", f"{dept_year}_LIDARHD", roi, ply_filename)
    )


def load_excluded_tiles_from_details_csv(
    csv_path: Optional[str],
) -> Set[Tuple[str, str]]:
    """Load (split, patch_id) tiles to exclude (reason=missing_coord_file)."""
    if not csv_path:
        return set()
    if not osp.isfile(csv_path):
        log.warning(
            "Excluded tiles details CSV not found: %s. Continuing with empty excluded set.",
            csv_path,
        )
        return set()

    excluded: Set[Tuple[str, str]] = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("reason") != "missing_coord_file":
                continue
            split = (row.get("split") or "").strip().lower()
            patch_id = (row.get("patch_id") or "").strip()
            if split and patch_id:
                excluded.add((split, patch_id))
    return excluded


@dataclass(frozen=True)
class ManifestPatch:
    split: str
    dept_year: str
    roi: str
    scene_i_j: str
    patch_id: str


def load_manifest_patches(
    split_manifest_csv: str,
    splits: Sequence[str] = ("train", "val", "test"),
) -> List[ManifestPatch]:
    """Load scene_split_manifest.csv rows with LIDARHD=True (one row per patch_id)."""
    if not osp.isfile(split_manifest_csv):
        raise FileNotFoundError(f"Split manifest CSV not found: {split_manifest_csv}")

    splits_set = set(splits)
    patches_by_id: dict[str, ManifestPatch] = {}

    with open(split_manifest_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get("split") or "").strip().lower()
            dept_year = (row.get("dept_year") or "").strip()
            roi = (row.get("roi") or "").strip()
            scene_i_j = (row.get("scene_i_j") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()

            if not split or not dept_year or not roi or not scene_i_j or not patch_id:
                continue
            if split not in splits_set:
                continue
            if not _parse_csv_bool(row.get("LIDARHD", "")):
                continue

            if patch_id not in patches_by_id:
                patches_by_id[patch_id] = ManifestPatch(
                    split=split,
                    dept_year=dept_year,
                    roi=roi,
                    scene_i_j=scene_i_j,
                    patch_id=patch_id,
                )

    return list(patches_by_id.values())


def manifest_to_split_csv(
    split_manifest_csv: str,
    labels_root: str,
    output_csv: str,
    excluded_tiles_details_csv: Optional[str] = None,
    splits: Sequence[str] = ("train", "val", "test"),
    require_existing_ply: bool = False,
) -> int:
    """Build basename,split CSV with absolute paths to Flair3D-build PLY files."""
    patches = load_manifest_patches(split_manifest_csv, splits=splits)
    excluded = load_excluded_tiles_from_details_csv(excluded_tiles_details_csv)

    rows = []
    skipped_excluded = 0
    skipped_missing = 0
    for patch in patches:
        if (patch.split, patch.patch_id) in excluded:
            skipped_excluded += 1
            continue
        ply_path = build_ply_path(
            labels_root, patch.dept_year, patch.roi, patch.scene_i_j
        )
        if require_existing_ply and not osp.isfile(ply_path):
            skipped_missing += 1
            continue
        rows.append({"basename": ply_path, "split": patch.split})

    os.makedirs(osp.dirname(osp.abspath(output_csv)) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    summary = (
        f"Split CSV saved to {output_csv} "
        f"({len(rows)} patches, {skipped_excluded} excluded, {skipped_missing} missing PLY)"
    )
    log.info(summary)
    print(summary)
    return len(rows)


class FLAIR3DDataset(HDF5Dataset):
    """Dataset for FLAIR3D dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> Optional[Data]:
        return super().__getitem__(idx)

    def _get_data(self, sample_hdf5_path: str) -> Data:
        if self.dataset is None:
            self.dataset = h5py.File(self.hdf5_file_path, "r")

        grp = self.dataset[sample_hdf5_path]
        kwargs = dict(
            x=torch.from_numpy(grp["x"][...]),
            pos=torch.from_numpy(grp["pos"][...]),
            idx_in_original_cloud=grp["idx_in_original_cloud"][...],
            x_features_names=grp["x"].attrs["x_features_names"].tolist(),
        )
        if "y" in grp:
            y = torch.from_numpy(grp["y"][...]).long()
            # Flair3D-build (v12/v14/v20): invalid semantic ids (-1) → Void (15, ignore_index).
            y[y < 0] = 15
            kwargs["y"] = y
        if "y_cosia" in grp:
            kwargs["y_cosia"] = torch.from_numpy(grp["y_cosia"][...])
        if "y_lidarhd" in grp:
            kwargs["y_lidarhd"] = torch.from_numpy(grp["y_lidarhd"][...])
        if "y_elevation" in grp:
            kwargs["y_elevation"] = torch.from_numpy(grp["y_elevation"][...]).float()
        return Data(**kwargs)


def _get_xyz(points) -> np.ndarray:
    """Read point coordinates from LAS (X,Y,Z) or Flair3D-build PLY (x,y,z)."""
    if all(c in points.dtype.names for c in ["X", "Y", "Z"]):
        axes = ["X", "Y", "Z"]
    elif all(c in points.dtype.names for c in ["x", "y", "z"]):
        axes = ["x", "y", "z"]
    else:
        raise KeyError(
            "Point cloud must contain X/Y/Z or x/y/z fields. "
            f"Available: {points.dtype.names}"
        )
    return np.asarray([points[a] for a in axes], dtype=np.float32).transpose()


def _flair3d_feature_tensors(points) -> Tuple[np.ndarray, List[str]]:
    """Build feature matrix and names (shared by flair3d and flair3d_plus pre_transforms)."""
    if "Intensity" in points.dtype.names:
        intensity = np.array(points["Intensity"], dtype=np.float32).clip(
            min=0, max=60000
        ) / 60000
    else:
        intensity = np.zeros(points.shape[0], dtype=np.float32)

    rgb_avg = np.zeros(points.shape[0], dtype=np.float32)
    if all(c in points.dtype.names for c in ["Red", "Green", "Blue"]):
        rgb_avg = (
            np.asarray([points["Red"], points["Green"], points["Blue"]], dtype=np.float32)
            .transpose()
            .mean(axis=1)
        )

    x_list = [intensity]
    x_features_names = ["Intensity"]

    for color in ["Red", "Green", "Blue", "Infrared"]:
        if color in points.dtype.names:
            x_list.append(np.asarray(points[color], dtype=np.float32))
            x_features_names.append(color)

    x_list.append(rgb_avg)
    x_features_names.append("rgb_avg")

    x = np.stack(x_list, axis=0).transpose()
    return x, x_features_names


def oldflair3d_pre_transform(points):
    if "ReturnNumber" not in points.dtype.names:
        return_number = np.ones(points.shape[0], dtype=np.float32)
        points = append_fields(points, "ReturnNumber", return_number, dtypes=np.float32, usemask=False)
    else:
        points["ReturnNumber"] = points["ReturnNumber"].astype(np.float32)

    if "NumberOfReturns" not in points.dtype.names:
        number_of_returns = np.ones(points.shape[0], dtype=np.float32)
        points = append_fields(
            points, "NumberOfReturns", number_of_returns, dtypes=np.float32, usemask=False
        )
    else:
        points["NumberOfReturns"] = points["NumberOfReturns"].astype(np.float32)

    if "Classification" not in points.dtype.names:
        if "lidarhd_class" in points.dtype.names:
            classification = points["lidarhd_class"].astype(np.int32)
        else:
            print("No lidarhd_class found, using 0")
            classification = np.zeros(points.shape[0], dtype=np.int32)
        points = append_fields(
            points, "Classification", classification, dtypes=np.int32, usemask=False
        )
    else:
        points["Classification"] = points["Classification"].astype(np.int32)

    if "Infrared" not in points.dtype.names:
        infrared = points["Intensity"]
        points = append_fields(points, "Infrared", infrared, dtypes=np.float32, usemask=False)
    else:
        points["Infrared"] = points["Infrared"].astype(np.float32)

    return lidar_hd_pre_transform(points)


def flair3d_pre_transform(points):
    pos = _get_xyz(points)

    x, x_features_names = _flair3d_feature_tensors(points)

    y_cosia = points["cosia_class"].astype(np.float32)
    y_lidarhd = points["lidarhd_class"].astype(np.float32)

    return Data(
        pos=pos,
        x=x,
        y_cosia=y_cosia,
        y_lidarhd=y_lidarhd,
        x_features_names=x_features_names,
    )


def flair3d_plus_pre_transform(points):
    """Load Flair3D-build output PLY with precomputed ``semantic`` labels (v20; v12/v14 also supported)."""
    if "semantic" not in points.dtype.names:
        raise KeyError(
            "Field 'semantic' not found in PLY. Run Flair3D-build (e.g. label=v20) on this patch first. "
            f"Available fields: {points.dtype.names}"
        )

    pos = _get_xyz(points)
    x, x_features_names = _flair3d_feature_tensors(points)
    y = np.asarray(points["semantic"], dtype=np.float32)

    return Data(pos=pos, x=x, y=y, x_features_names=x_features_names)


def enrich_points_with_raster_labels(points, cloud_path: str, raster_root: str):
    """Sample the DEM_ELEV GeoTIFF onto points once per patch (before subtiling)."""
    dept_year, roi, lidar_patch_stem = parse_ply_patch_metadata(cloud_path)
    pos = _get_xyz(points)
    xy = pos[:, :2]
    z = pos[:, 2].astype(np.float32, copy=False)

    dem_path = build_modality_patch_path(
        raster_root, "DEM_ELEV", dept_year, roi, lidar_patch_stem
    )
    if osp.isfile(dem_path):
        dtm_values, _ = sample_raster_to_points_float(
            dem_path, xy, fill_value=np.nan, band_index=2
        )
        elevation = z - dtm_values
    else:
        elevation = np.full(points.shape[0], np.nan, dtype=np.float32)
    points = append_fields(points, "elevation", elevation, dtypes=np.float32, usemask=False)

    return points


def flair3d_plus_multitask_pre_transform(points):
    """Flair3D+ pre-transform with segment + raster-derived elevation labels."""
    if "semantic" not in points.dtype.names:
        raise KeyError(
            "Field 'semantic' not found in PLY. Run Flair3D-build (e.g. label=v20) on this patch first. "
            f"Available fields: {points.dtype.names}"
        )

    label_definitions = build_preprocess_label_definitions()
    pos = _get_xyz(points)
    x, x_features_names = _flair3d_feature_tensors(points)

    y = apply_remap(
        np.asarray(points["semantic"], dtype=np.int64), label_definitions.segment
    ).astype(np.float32)

    kwargs = dict(pos=pos, x=x, y=y, x_features_names=x_features_names)

    if "elevation" in points.dtype.names:
        kwargs["y_elevation"] = np.asarray(points["elevation"], dtype=np.float32)

    return Data(**kwargs)


def _cli_manifest_to_split():
    parser = argparse.ArgumentParser(
        description="Build myria3d split CSV from Flair3D+ scene_split_manifest.csv"
    )
    parser.add_argument(
        "--split-manifest-csv",
        required=True,
        help="Path to scene_split_manifest.csv",
    )
    parser.add_argument(
        "--labels-root",
        required=True,
        help="Flair3D-build output_root (directory containing LIDARHD/)",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output path for basename,split CSV",
    )
    parser.add_argument(
        "--excluded-tiles-details-csv",
        default=None,
        help="Optional missing_coord_tiles.details.csv",
    )
    parser.add_argument(
        "--require-existing-ply",
        action="store_true",
        help="Skip patches whose built PLY file is not on disk",
    )
    args = parser.parse_args()
    manifest_to_split_csv(
        args.split_manifest_csv,
        args.labels_root,
        args.output_csv,
        excluded_tiles_details_csv=args.excluded_tiles_details_csv,
        require_existing_ply=args.require_existing_ply,
    )


if __name__ == "__main__":
    _cli_manifest_to_split()
