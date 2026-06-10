"""GeoTIFF sampling utilities for Flair3D+ multitask preprocessing."""

from __future__ import annotations

import os
import os.path as osp
from typing import Any, Dict, Tuple

import numpy as np


def parse_ply_patch_metadata(cloud_path: str) -> Tuple[str, str, str]:
    """Parse dept_year, roi, and lidar patch stem from a Flair3D-build PLY path."""
    stem = osp.splitext(osp.basename(cloud_path))[0]
    if "_LIDARHD_" not in stem:
        raise ValueError(
            f"Expected Flair3D+ PLY stem containing '_LIDARHD_', got: {stem}"
        )
    dept_year, rest = stem.split("_LIDARHD_", 1)
    roi, _scene_i_j = rest.rsplit("_", 1)
    return dept_year, roi, stem


def build_modality_patch_path(
    dataset_root: str,
    modality: str,
    dept_year: str,
    roi: str,
    lidar_patch_stem: str,
) -> str:
    """Build one modality patch path from split metadata and LiDAR patch stem."""
    modality_stem = lidar_patch_stem.replace("_LIDARHD_", f"_{modality}_")
    if modality == "DEM_ELEV":
        default_path = osp.join(
            dataset_root,
            modality,
            f"{dept_year}_{modality}",
            roi,
            f"{modality_stem}.tif",
        )
        if osp.isfile(default_path):
            return default_path
        return osp.join(dataset_root, modality, roi, f"{modality_stem}.tif")
    return osp.join(
        dataset_root,
        modality,
        f"{dept_year}_{modality}",
        roi,
        f"{modality_stem}.tif",
    )


def sample_raster_to_points(
    raster_path: str,
    xy: np.ndarray,
    fill_value: int = -1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample one raster value per XY point (nearest pixel, no interpolation)."""
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be of shape (N, 2), got {xy.shape}")

    import rasterio  # type: ignore[import-not-found]

    values = np.full(xy.shape[0], fill_value=fill_value, dtype=np.int16)
    with rasterio.open(raster_path) as src:
        raster_nodata = src.nodata

        inv = ~src.transform
        x = xy[:, 0].astype(np.float64, copy=False)
        y = xy[:, 1].astype(np.float64, copy=False)
        cols = np.floor(inv.a * x + inv.b * y + inv.c).astype(np.int64, copy=False)
        rows = np.floor(inv.d * x + inv.e * y + inv.f).astype(np.int64, copy=False)
        inside_mask = (
            (rows >= 0)
            & (rows < src.height)
            & (cols >= 0)
            & (cols < src.width)
        )
        outside_count = int((~inside_mask).sum())

        if inside_mask.any():
            band1 = src.read(1)
            sampled = band1[rows[inside_mask], cols[inside_mask]]

            if raster_nodata is not None:
                nodata_mask = sampled == raster_nodata
                sampled = sampled.astype(np.int16, copy=True)
                sampled[nodata_mask] = fill_value
                nodata_points = int(nodata_mask.sum())
            else:
                nodata_points = 0

            values[inside_mask] = sampled.astype(np.int16, copy=False)
        else:
            nodata_points = 0

    stats = {
        "raster_nodata": raster_nodata,
        "num_points": int(xy.shape[0]),
        "num_points_outside_raster": outside_count,
        "num_points_nodata": nodata_points,
    }
    return values, stats


def sample_raster_to_points_float(
    raster_path: str,
    xy: np.ndarray,
    fill_value: float = np.nan,
    band_index: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample one float raster value per XY point (nearest pixel, no interpolation)."""
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be of shape (N, 2), got {xy.shape}")

    import rasterio  # type: ignore[import-not-found]

    values = np.full(xy.shape[0], fill_value=fill_value, dtype=np.float32)
    with rasterio.open(raster_path) as src:
        if band_index < 1 or band_index > src.count:
            raise ValueError(
                f"Invalid band_index={band_index}; raster has {src.count} band(s) for {raster_path}."
            )
        raster_nodata = src.nodata

        inv = ~src.transform
        x = xy[:, 0].astype(np.float64, copy=False)
        y = xy[:, 1].astype(np.float64, copy=False)
        cols = np.floor(inv.a * x + inv.b * y + inv.c).astype(np.int64, copy=False)
        rows = np.floor(inv.d * x + inv.e * y + inv.f).astype(np.int64, copy=False)
        inside_mask = (
            (rows >= 0)
            & (rows < src.height)
            & (cols >= 0)
            & (cols < src.width)
        )
        outside_count = int((~inside_mask).sum())

        if inside_mask.any():
            band = src.read(band_index).astype(np.float32, copy=False)
            sampled = band[rows[inside_mask], cols[inside_mask]]

            if raster_nodata is not None:
                nodata_mask = sampled == raster_nodata
                sampled = sampled.astype(np.float32, copy=True)
                sampled[nodata_mask] = fill_value
                nodata_points = int(nodata_mask.sum())
            else:
                nodata_points = 0

            values[inside_mask] = sampled.astype(np.float32, copy=False)
        else:
            nodata_points = 0

    stats = {
        "raster_nodata": raster_nodata,
        "num_points": int(xy.shape[0]),
        "num_points_outside_raster": outside_count,
        "num_points_nodata": nodata_points,
    }
    return values, stats
