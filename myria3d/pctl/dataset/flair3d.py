import copy
import os
import os.path as osp
from numbers import Number
from typing import Callable, List, Optional

import numpy as np
from numpy.lib.recfunctions import append_fields
import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
import pandas as pd

from myria3d.pctl.dataset.utils import (
    LAS_PATHS_BY_SPLIT_DICT_TYPE,
    SPLIT_TYPE,
    pre_filter_below_n_points,
    split_cloud_into_samples,
)
from myria3d.pctl.points_pre_transform.lidar_hd import lidar_hd_pre_transform
from myria3d.utils import utils

log = utils.get_logger(__name__)

from myria3d.pctl.dataset.hdf5 import HDF5Dataset

class FLAIR3DDataset(HDF5Dataset):
    """Dataset for FLAIR3D dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> Optional[Data]:
        return super().__getitem__(idx)
    
RAW_DIR = "/data/geist/superpixel_transformer_dev/data/flair3d/raw"
    
RAW_FOLDER_NAMES = {
    'train': [
        # 'D005-2018_LIDARHD/AA-S1-14',
        'D006-2020_LIDARHD/AU-S2-13',
        'D009-2019_LIDARHD/AA-S1-14',
        'D013-2020_LIDARHD/AA-S1-14',
        'D015-2020_LIDARHD/AA-S1-11',
        'D023-2020_LIDARHD/AA-S1-23',
        # 'D033-2021_LIDARHD/AA-S1-1',
        'D034-2021_LIDARHD/AA-S1-27',
        'D035-2020_LIDARHD/AA-S1-37',
        'D036-2020_LIDARHD/AA-S1-14',
        'D038-2021_LIDARHD/FA-S1-18',
        'D041-2021_LIDARHD/AA-S1-24',
        'D052-2019_LIDARHD/AA-S1-28',
        'D058-2020_LIDARHD/AA-S1-17',
        # 'D060-2021_LIDARHD/AA-S1-40',
        'D064-2021_LIDARHD/AA-S1-26',
        'D069-2020_LIDARHD/AA-S1-28',
        'D070-2020_LIDARHD/FA-S1-18',
        # 'D072-2019_LIDARHD/AA-S1-15',
        'D074-2020_LIDARHD/AA-S1-47',
        'D080-2017_LIDARHD/UU-S1-25',
        'D081-2020_LIDARHD/AA-S1-13',
        # 'D084-2021_LIDARHD/AA-S1-21',
        ],
    'val': [
        'D060-2021_LIDARHD/AA-S1-40',
        'D072-2019_LIDARHD/AA-S1-15',
        ],
    'test': [
        'D005-2018_LIDARHD/AA-S1-14',
        'D033-2021_LIDARHD/AA-S1-1',
        'D084-2021_LIDARHD/AA-S1-21',
        ]
} 
    
def construct_las_paths_by_split_dict(raw_dir, raw_folder_names):
    las_paths_by_split_dict = {}
    
    for split, folder_names in raw_folder_names.items():
        # Retrieve the available files.
        paths = []
        
        for folder_name in folder_names:
            folder_path = os.path.join(raw_dir, folder_name)
            filenames = os.listdir(folder_path)
            # Construct full paths by joining folder path with each filename
            full_paths = [os.path.join(folder_path, filename) for filename in filenames]
            paths.extend(full_paths)
        
        # Add to the output dictionary
        las_paths_by_split_dict[split] = paths
        
        
    return las_paths_by_split_dict


from myria3d.pctl.points_pre_transform.lidar_hd import lidar_hd_pre_transform

def oldflair3d_pre_transform(points):
    # Add ReturnNumber field if it doesn't exist (for PLY files that don't have this LAS-specific field)
    if "ReturnNumber" not in points.dtype.names:
        return_number = np.ones(points.shape[0], dtype=np.float32)
        points = append_fields(points, "ReturnNumber", return_number, dtypes=np.float32, usemask=False)
    else:
        # If it exists, ensure it's float32
        points["ReturnNumber"] = points["ReturnNumber"].astype(np.float32)
    
    
    # Add NumberOfReturns field if it doesn't exist
    if "NumberOfReturns" not in points.dtype.names:
        number_of_returns = np.ones(points.shape[0], dtype=np.float32)
        points = append_fields(points, "NumberOfReturns", number_of_returns, dtypes=np.float32, usemask=False)
    else:
        # If it exists, ensure it's float32
        points["NumberOfReturns"] = points["NumberOfReturns"].astype(np.float32)
    
    # # Add Classification field if it doesn't exist (use lidarhd_class if available, otherwise 0)
    if "Classification" not in points.dtype.names:
        if "lidarhd_class" in points.dtype.names:
            classification = points["lidarhd_class"].astype(np.int32)
        else:
            print("No lidarhd_class found, using 0")
            classification = np.zeros(points.shape[0], dtype=np.int32)
        points = append_fields(points, "Classification", classification, dtypes=np.int32, usemask=False)
    else:
        # If it exists, ensure it's int32
        points["Classification"] = points["Classification"].astype(np.int32)
        
    # Add infrared
    if "Infrared" not in points.dtype.names:
        infrared = points['Intensity']
        points = append_fields(points, "Infrared", infrared, dtypes=np.float32, usemask=False)
    else:
        # If it exists, ensure it's float32
        points["Infrared"] = points["Infrared"].astype(np.float32)
    
    return lidar_hd_pre_transform(points)

def flair3d_pre_transform(points):
    pos = np.asarray([points["X"], points["Y"], points["Z"]], dtype=np.float32).transpose()
    
    # Intensity
    intensity = np.array(points['Intensity'], 
                         dtype=np.float32
                        ).clip(min=0, max=60000) / 60000
    points["Intensity"] = intensity
    
    # Additional features
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
            x_list.append(points[color])
            x_features_names.append(color)

    x_list += [rgb_avg,]
    x_features_names += ["rgb_avg",]

    x = np.stack(x_list, axis=0).transpose()
    
    # Semantic
    y = points["cosia_class"].astype(np.float32)
    
    data = Data(pos=pos,
                x=x,
                y=y,
                x_features_names=x_features_names)
    
    return data

def save_split_csv(las_paths_by_split_dict: dict, output_path: str):
    """Save a split CSV file from a las_paths_by_split_dict.
    
    Args:
        las_paths_by_split_dict: Dictionary with keys 'train', 'val', 'test' and values being lists of file paths
        output_path: Path where to save the CSV file
    """
    rows = []
    for split, paths in las_paths_by_split_dict.items():
        for path in paths:
            rows.append({"basename": path, "split": split})
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    log.info(f"Split CSV saved to {output_path}")
    print(f"Split CSV saved to {output_path}")

if __name__ == "__main__":
    
    las_paths_by_split_dict = construct_las_paths_by_split_dict(RAW_DIR, RAW_FOLDER_NAMES)
    save_split_csv(las_paths_by_split_dict, "tests/data/flair3d_split.csv")
    
    mini_las_paths_by_split_dict = {k: v[:1] for k, v in las_paths_by_split_dict.items()}
    
    # Save the split CSV
    split_csv_path = "tests/data/mini_flair3d_split.csv"
    save_split_csv(mini_las_paths_by_split_dict, split_csv_path)
    
    # dataset = FLAIR3DDataset(
    #     las_paths_by_split_dict=mini_las_paths_by_split_dict,
    #     hdf5_file_path="tests/data/mini_flair3d.hdf5",
    #     points_pre_transform=flair3d_pre_transform,
    #     epsg="2154",
    # )
    # print(dataset[0])