import glob
import json
from pathlib import Path
import subprocess as sp
from numbers import Number
from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd
import pdal
import torch

SPLIT_TYPE = Union[Literal["train"], Literal["val"], Literal["test"]]
LAS_PATHS_BY_SPLIT_DICT_TYPE = Dict[SPLIT_TYPE, List[str]]


def find_file_in_dir(data_dir: str, basename: str) -> str:
    """Query files matching a basename in input_data_dir and its subdirectories.
    Args:
        input_data_dir (str): data directory
    Returns:
        [str]: first file path matching the query.
    """
    query = f"{data_dir}/**/{basename}"
    files = glob.glob(query, recursive=True)
    return files[0]


def get_mosaic_of_centers(tile_width: Number, subtile_width: Number, subtile_overlap: Number = 0):
    if subtile_overlap < 0:
        raise ValueError("datamodule.subtile_overlap must be positive.")

    xy_range = np.arange(
        subtile_width / 2,
        tile_width + (subtile_width / 2) - subtile_overlap,
        step=subtile_width - subtile_overlap,
    )
    return [np.array([x, y]) for x in xy_range for y in xy_range]


def get_num_subtiles(
    tile_width: Number,
    subtile_width: Number,
    subtile_overlap: Number = 0,
) -> int:
    """Number of mosaic subtiles for a tile (e.g. 4 for 100 m / 50 m, overlap 0)."""
    return len(get_mosaic_of_centers(tile_width, subtile_width, subtile_overlap))


def pdal_read_point_cloud_array(cloud_path: str, epsg: str):
    """Read point cloud (LAS or PLY) as a named array.

    Args:
        cloud_path (str): input point cloud file path (LAS or PLY)
        epsg (str): epsg to force the reading with

    Returns:
        np.ndarray: named array with all point cloud dimensions, including extra ones, with dict-like access.

    """
    p1 = pdal.Pipeline() | get_pdal_reader(cloud_path, epsg)
    p1.execute()
    return p1.arrays[0]


def pdal_read_point_cloud_array_as_float32(cloud_path: str, epsg: str):
    """Read point cloud (LAS or PLY) as a named array, casted to floats.

    Args:
        cloud_path (str): input point cloud file path (LAS or PLY)
        epsg (str): epsg to force the reading with

    Returns:
        np.ndarray: named array with all point cloud dimensions casted to float32.

    """
    arr = pdal_read_point_cloud_array(cloud_path, epsg)
    all_floats = np.dtype({"names": arr.dtype.names, "formats": ["f4"] * len(arr.dtype.names)})
    return arr.astype(all_floats)


def pdal_read_las_array(las_path: str, epsg: str):
    """Read LAS as a named array.

    Args:
        las_path (str): input LAS path
        epsg (str): epsg to force the reading with

    Returns:
        np.ndarray: named array with all LAS dimensions, including extra ones, with dict-like access.

    """
    return pdal_read_point_cloud_array(las_path, epsg)


def pdal_read_las_array_as_float32(las_path: str, epsg: str):
    """Read LAS as a a named array, casted to floats."""
    return pdal_read_point_cloud_array_as_float32(las_path, epsg)


def get_metadata(cloud_path: str) -> dict:
    """Returns metadata contained in a point cloud file (LAS or PLY).

    Args:
        cloud_path (str): input point cloud file path (LAS or PLY) to get metadata from.
    Returns:
        dict : the metadata.
    """
    file_ext = Path(cloud_path).suffix.lower()
    if file_ext == ".las" or file_ext == ".laz":
        pipeline = pdal.Reader.las(filename=cloud_path).pipeline()
    elif file_ext == ".ply":
        pipeline = pdal.Reader.ply(filename=cloud_path).pipeline()
    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. Supported formats are .las, .laz, and .ply"
        )
    pipeline.execute()
    return pipeline.metadata


def get_pdal_reader(cloud_path: str, epsg: str):
    """Get the appropriate PDAL reader based on file extension.

    Args:
        cloud_path (str): input point cloud file path (LAS or PLY).
        epsg (str): epsg to force the reading with
    Returns:
        pdal.Reader: reader to use in a pipeline (Reader.las or Reader.ply).

    """
    file_ext = Path(cloud_path).suffix.lower()

    if file_ext == ".las" or file_ext == ".laz":
        if epsg:
            # if an epsg in provided, force pdal to read the lidar file with it
            # epsg can be added as a number like "2154" or as a string like "EPSG:2154"
            return pdal.Reader.las(
                filename=cloud_path,
                nosrs=True,
                override_srs=f"EPSG:{epsg}" if str(epsg).isdigit() else epsg,
            )

        try:
            if get_metadata(cloud_path)["metadata"]["readers.las"]["srs"]["compoundwkt"]:
                # read the lidar file with pdal default
                return pdal.Reader.las(filename=cloud_path)
        except Exception:
            pass  # we will go to the "raise exception" anyway

        raise Exception("No EPSG provided, neither in the lidar file or as parameter")

    elif file_ext == ".ply":
        if epsg:
            # if an epsg in provided, force pdal to read the point cloud file with it
            # epsg can be added as a number like "2154" or as a string like "EPSG:2154"
            return pdal.Reader.ply(
                filename=cloud_path,
                override_srs=f"EPSG:{epsg}" if str(epsg).isdigit() else epsg,
            )
        else:
            # For PLY files, if no EPSG is provided, we still create a reader
            # but the user should be aware that coordinates might not be in the correct CRS
            return pdal.Reader.ply(filename=cloud_path)

    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. Supported formats are .las, .laz, and .ply"
        )


def get_pdal_info_metadata(las_path: str) -> Dict:
    """Read las metadata using pdal info
    Args:
        las_path (str): input LAS path to read.
    Returns:
        (dict): dictionary containing metadata from the las file
    """
    r = sp.run(["pdal", "info", "--metadata", las_path], capture_output=True)
    if r.returncode == 1:
        msg = r.stderr.decode()
        raise RuntimeError(msg)

    output = r.stdout.decode()
    json_info = json.loads(output)

    return json_info["metadata"]


# hdf5, iterable


def get_subtile_choice(
    pos: torch.Tensor,
    tile_width: Number,
    subtile_width: Number,
    subtile_index: int,
    subtile_overlap: Number = 0,
) -> torch.Tensor:
    """Return a boolean mask selecting one square subtile; stays on pos.device."""
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError(f"pos must be (N, >=2), got {tuple(pos.shape)}")

    centers = get_mosaic_of_centers(
        tile_width, subtile_width, subtile_overlap=subtile_overlap
    )
    if subtile_index < 0 or subtile_index >= len(centers):
        raise ValueError(
            f"subtile_index must be in [0, {len(centers)}), got {subtile_index}"
        )

    center = torch.as_tensor(
        centers[subtile_index], device=pos.device, dtype=pos.dtype
    )
    xy_local = pos[:, :2] - pos[:, :2].min(dim=0).values
    radius = subtile_width // 2
    return (torch.abs(xy_local - center) <= radius).all(dim=1)


def get_subtile_mask(
    pos: np.ndarray,
    tile_width: Number,
    subtile_width: Number,
    subtile_index: int,
    subtile_overlap: Number = 0,
) -> np.ndarray:
    """Return a boolean mask selecting one square subtile from a tile-local point cloud."""
    pos = np.asarray(pos, dtype=np.float32)
    choice = get_subtile_choice(
        torch.from_numpy(pos),
        tile_width,
        subtile_width,
        subtile_index,
        subtile_overlap=subtile_overlap,
    )
    return choice.numpy()


def split_points_into_samples(
    points: np.ndarray,
    tile_width: Number,
    subtile_width: Number,
    subtile_overlap: Number = 0,
):
    """Split an in-memory point cloud array into subtiles.

    Yields:
        idx_in_original_cloud, and points of sample in pdal input format casted as floats.
    """
    pos = np.asarray([points["X"], points["Y"], points["Z"]], dtype=np.float32).transpose()
    centers = get_mosaic_of_centers(
        tile_width, subtile_width, subtile_overlap=subtile_overlap
    )
    for subtile_index in range(len(centers)):
        mask = get_subtile_mask(
            pos, tile_width, subtile_width, subtile_index, subtile_overlap
        )
        sample_idx = np.where(mask)[0]
        if not len(sample_idx):
            continue
        yield sample_idx, points[sample_idx]


def split_cloud_into_samples(
    cloud_path: str,
    tile_width: Number,
    subtile_width: Number,
    epsg: str,
    subtile_overlap: Number = 0,
):
    """Split point cloud (LAS or PLY) into samples.

    Args:
        cloud_path (str): path to raw point cloud file (LAS or PLY)
        tile_width (Number): width of input point cloud file
        subtile_width (Number): width of receptive field.
        epsg (str): epsg to force the reading with
        subtile_overlap (Number, optional): overlap between adjacent tiles. Defaults to 0.

    Yields:
        _type_: idx_in_original_cloud, and points of sample in pdal input format casted as floats.

    """
    points = pdal_read_point_cloud_array_as_float32(cloud_path, epsg)
    yield from split_points_into_samples(
        points, tile_width, subtile_width, subtile_overlap=subtile_overlap
    )


def pre_filter_below_n_points(data, min_num_nodes=1):
    return data.pos.shape[0] < min_num_nodes


def get_las_paths_by_split_dict(
    data_dir: str, split_csv_path: str
) -> LAS_PATHS_BY_SPLIT_DICT_TYPE:
    las_paths_by_split_dict: LAS_PATHS_BY_SPLIT_DICT_TYPE = {}
    split_df = pd.read_csv(split_csv_path)
    for phase in ["train", "val", "test"]:
        basenames = split_df[split_df.split == phase].basename.tolist()
        # Reminder: an explicit data structure with ./val, ./train, ./test subfolder is required.
        las_paths_by_split_dict[phase] = [str(Path(data_dir) / phase / b) for b in basenames]

    if not las_paths_by_split_dict:
        raise FileNotFoundError(
            (
                f"No basename found while parsing directory {data_dir}"
                f"using {split_csv_path} as split CSV."
            )
        )

    return las_paths_by_split_dict
