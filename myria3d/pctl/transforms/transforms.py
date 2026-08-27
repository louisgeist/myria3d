import math
import random
import re
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from myria3d.pctl.dataset.utils import get_num_subtiles, get_subtile_choice
from myria3d.utils import utils

log = utils.get_logger(__name__)

COMMON_CODE_FOR_ALL_ARTEFACTS = 65

# Feature groups dropped by RandomDropColor / RandomDropStrength (Pointcept names).
# Layout matches Flair3D+ `x`: Intensity, Red, Green, Blue, rgb_avg.
COLOR_FEATURE_NAMES = ("Red", "Green", "Blue", "rgb_avg")
STRENGTH_FEATURE_NAMES = ("Intensity",)


def resolve_x_feature_names(data: Data) -> List[str]:
    """Return per-point feature names, even when PyG collate nests them per graph."""
    names = getattr(data, "x_features_names", None)
    if not names:
        return []
    first = names[0]
    if isinstance(first, (list, tuple)):
        return list(first)
    if isinstance(first, str):
        return list(names)
    return []


class ToTensor(BaseTransform):
    """Turn np.arrays specified by their keys into Tensor."""

    def __init__(self, keys: List[str] = ["pos", "x", "y"]):
        self.keys = keys

    def __call__(self, data: Data):
        for key in data.keys:
            if key in self.keys:
                data[key] = torch.from_numpy(data[key])
        return data


def subsample_data(data, num_nodes, choice: torch.Tensor):
    # TODO: get num_nodes from data.num_nodes instead to simplify signature
    out_nodes = torch.sum(choice) if choice.dtype == torch.bool else choice.size(0)
    for key, item in data:
        if key == "num_nodes":
            data.num_nodes = out_nodes
        elif key in ["copies", "idx_in_original_cloud"]:
            # Do not subsample copies of the original point cloud or indices of the original points
            # contained in the patch
            continue
        elif bool(re.search("edge", key)):
            continue
        elif torch.is_tensor(item) and item.size(0) == num_nodes:
            data[key] = item[choice]

    return data


class SubtileCrop(BaseTransform):
    """Crop a point cloud to one square subtile (HDF5 mosaic logic).

    When data.subtile_index is set, that subtile is used. Otherwise, if random=True,
    one subtile is drawn uniformly. Returns None when the crop is empty.
    """

    def __init__(
        self,
        tile_width: float = 100,
        subtile_width: float = 50,
        subtile_overlap: float = 0,
        random: bool = False,
        min_points: int = 1,
    ):
        self.tile_width = tile_width
        self.subtile_width = subtile_width
        self.subtile_overlap = subtile_overlap
        self.random = random
        # Skip a crop with fewer than this many points (default 1 == only reject empty).
        # A near-empty quadrant of an otherwise-fine tile is a useless training sample and,
        # once GridSampling collapses it toward num_nodes==1, breaks batch collate.
        self.min_points = min_points

    def __call__(self, data: Data):
        num_subtiles = get_num_subtiles(
            self.tile_width, self.subtile_width, subtile_overlap=self.subtile_overlap
        )
        if num_subtiles == 0:
            raise ValueError("SubtileCrop requires at least one subtile.")

        subtile_index = getattr(data, "subtile_index", None)
        if subtile_index is not None:
            # Fixed index (val/test): that quadrant only -- None if it is (near-)empty.
            candidate_indices = [int(subtile_index)]
        elif self.random:
            # Try quadrants in random order and take the first with enough points, so a
            # tile whose points sit in only 1-2 quadrants still yields a usable sample
            # instead of a wasted None. Uniform over the quadrants that pass min_points.
            candidate_indices = random.sample(range(num_subtiles), num_subtiles)
        else:
            raise ValueError("SubtileCrop requires data.subtile_index or random=True.")

        if hasattr(data, "subtile_index"):
            del data.subtile_index

        choice = None
        for idx in candidate_indices:
            candidate = get_subtile_choice(
                data.pos,
                self.tile_width,
                self.subtile_width,
                idx,
                subtile_overlap=self.subtile_overlap,
            )
            if int(candidate.sum()) >= self.min_points:
                choice = candidate
                break
        if choice is None:
            return None

        num_nodes = data.num_nodes
        if hasattr(data, "idx_in_original_cloud") and data.idx_in_original_cloud is not None:
            original_idx = np.asarray(data.idx_in_original_cloud)
        else:
            original_idx = np.arange(num_nodes, dtype=np.int32)
        data = subsample_data(data, num_nodes, choice)
        data.idx_in_original_cloud = original_idx[choice.numpy()]
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(tile_width={self.tile_width}, "
            f"subtile_width={self.subtile_width}, random={self.random}, "
            f"min_points={self.min_points})"
        )


class MaximumNumNodes(BaseTransform):
    def __init__(self, num: int):
        self.num = num

    def __call__(self, data):
        num_nodes = data.num_nodes

        if num_nodes <= self.num:
            return data

        choice = torch.randperm(data.num_nodes)[: self.num]
        data = subsample_data(data, num_nodes, choice)

        return data


class MinimumNumNodes(BaseTransform):
    def __init__(self, num: int):
        self.num = num

    def __call__(self, data):
        num_nodes = data.num_nodes

        if num_nodes >= self.num:
            return data

        choice = torch.cat(
            [torch.randperm(num_nodes) for _ in range(math.ceil(self.num / num_nodes))],
            dim=0,
        )[: self.num]

        data = subsample_data(data, num_nodes, choice)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.num}"


class CopyFullPos:
    """Make a copy of the original positions - to be used for test and inference."""

    def __call__(self, data: Data):
        if "copies" not in data:
            data.copies = dict()
        data.copies["pos_copy"] = data["pos"].clone()
        return data


class CopyFullPreparedTargets:
    """Make a copy of all prepared targets - to be used for test.

    Only tasks that get KNN-interpolated to the full point cloud need a copy here.
    pixel_semantic (forest_2d, roads) and tile_distribution (nathab_*) tasks are always
    evaluated at the model's native resolution — see MultiTaskModel — so they are
    deliberately absent from this list.
    """

    MULTITASK_TARGET_KEYS = (
        "y",
        "y_elevation",
    )

    def __call__(self, data: Data):
        if "copies" not in data:
            data.copies = dict()
        for key in self.MULTITASK_TARGET_KEYS:
            if hasattr(data, key) and getattr(data, key) is not None:
                data.copies[f"transformed_{key}_copy"] = getattr(data, key).clone()
        return data


class CopySampledPos(BaseTransform):
    """Make a copy of the unormalized positions of subsampled points - to be used for test and inference."""

    def __call__(self, data: Data):
        if "copies" not in data:
            data.copies = dict()
        data.copies["pos_sampled_copy"] = data["pos"].clone()
        return data


class StandardizeRGBAndIntensity(BaseTransform):
    """Standardize RGB and log(Intensity) features."""

    def __call__(self, data: Data):
        idx = data.x_features_names.index("Intensity")
        # Log transform to be less sensitive to large outliers - info is in lower values
        data.x[:, idx] = torch.log(data.x[:, idx] + 1)
        data.x[:, idx] = self.standardize_channel(data.x[:, idx])
        idx = data.x_features_names.index("rgb_avg")
        data.x[:, idx] = self.standardize_channel(data.x[:, idx])
        return data

    def standardize_channel(self, channel_data: torch.Tensor, clamp_sigma: int = 3):
        """Sample-wise standardization y* = (y-y_mean)/y_std. clamping to ignore large values."""
        mean = channel_data.mean()
        std = channel_data.std() + 10**-6
        if torch.isnan(std):
            std = 1.0
        standard = (channel_data - mean) / std
        clamp = clamp_sigma * std
        clamped = torch.clamp(input=standard, min=-clamp, max=clamp)
        return clamped


class NullifyLowestZ(BaseTransform):
    """Set lowest z to 0 (Z-only; XY centering is handled separately, e.g. by `Center`)."""

    def __call__(self, data):
        data.pos[:, 2] = data.pos[:, 2] - data.pos[:, 2].min()
        return data


class NormalizePos(BaseTransform):
    """
    Normalizes xy in [-1;1] range by scaling the whole point cloud (including z dim).
    XY are expected to be centered on zéro.

    """

    def __init__(self, subtile_width=50):
        half_subtile_width = subtile_width / 2
        self.scaling_factor = 1 / half_subtile_width

    def __call__(self, data):
        data.pos = data.pos * self.scaling_factor
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class ZRandomOffset(BaseTransform):
    """Random vertical translation applied uniformly to all points (Z registration noise)."""

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, data: Data) -> Data:
        z_offset = torch.randn(1, device=data.pos.device, dtype=data.pos.dtype).item() * self.std
        data.pos[:, 2] = data.pos[:, 2] + z_offset
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


class RandomDropFeatureGroup(BaseTransform):
    """Zero a group of per-point features on a random subset of points.

    Port of Pointcept ``RandomDropColor`` / ``RandomDropStrength``. When
    ``keep_mask`` is set, a boolean mask (True = dropped) is stored so the model
    can replace zeros with a learned fill-in value. Stacked calls OR into the
    existing mask. Train-only: put this in augmentations (after
    ``StandardizeRGBAndIntensity``).
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        mask_key: str,
        drop_ratio: float = 0.2,
        drop_application_ratio: float = 0.5,
        keep_mask: bool = False,
        drop_value: float = 0.0,
    ):
        self.feature_names = tuple(feature_names)
        self.mask_key = mask_key
        self.drop_ratio = drop_ratio
        self.drop_application_ratio = drop_application_ratio
        self.keep_mask = keep_mask
        self.drop_value = drop_value

    def __call__(self, data: Data) -> Data:
        names = resolve_x_feature_names(data)
        indices = [names.index(name) for name in self.feature_names if name in names]
        if not indices or data.x is None or data.x.size(0) == 0:
            return data

        n = data.x.size(0)
        existing = getattr(data, self.mask_key, None) if self.keep_mask else None
        if existing is not None:
            drop_mask = existing.bool().reshape(-1).clone()
        else:
            drop_mask = torch.zeros(n, dtype=torch.bool, device=data.x.device)

        if random.random() < self.drop_application_ratio:
            num_to_drop = int(n * self.drop_ratio)
            if num_to_drop > 0:
                idx = torch.randperm(n, device=data.x.device)[:num_to_drop]
                drop_mask[idx] = True
                for col in indices:
                    data.x[idx, col] = self.drop_value

        if self.keep_mask:
            data[self.mask_key] = drop_mask
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(drop_ratio={self.drop_ratio}, "
            f"drop_application_ratio={self.drop_application_ratio}, "
            f"keep_mask={self.keep_mask})"
        )


class RandomDropColor(RandomDropFeatureGroup):
    """Drop RGB (+ ``rgb_avg``) features; see Pointcept ``RandomDropColor``."""

    def __init__(
        self,
        drop_ratio: float = 0.2,
        drop_application_ratio: float = 0.5,
        keep_mask: bool = False,
        drop_value: float = 0.0,
    ):
        super().__init__(
            COLOR_FEATURE_NAMES,
            "color_mask",
            drop_ratio=drop_ratio,
            drop_application_ratio=drop_application_ratio,
            keep_mask=keep_mask,
            drop_value=drop_value,
        )


class RandomDropStrength(RandomDropFeatureGroup):
    """Drop LiDAR intensity; see Pointcept ``RandomDropStrength``."""

    def __init__(
        self,
        drop_ratio: float = 0.2,
        drop_application_ratio: float = 0.5,
        keep_mask: bool = False,
        drop_value: float = 0.0,
    ):
        super().__init__(
            STRENGTH_FEATURE_NAMES,
            "strength_mask",
            drop_ratio=drop_ratio,
            drop_application_ratio=drop_application_ratio,
            keep_mask=keep_mask,
            drop_value=drop_value,
        )


class TargetTransform(BaseTransform):
    """
    Make target vector based on input classification dictionnary.

    Example:
    Source : y = [6,6,17,9,1]
    Pre-processed:
    - classification_preprocessing_dict = {17:1, 9:1}
    - y' = [6,6,1,1,1]
    Mapped to consecutive integers:
    - classification_dict = {1:"unclassified", 6:"building"}
    - y'' = [1,1,0,0,0]

    """

    def __init__(
        self,
        classification_preprocessing_dict: Dict[int, int],
        classification_dict: Dict[int, str],
    ):
        self._set_preprocessing_mapper(classification_preprocessing_dict)
        self._set_mapper(classification_dict)

        # Set to attribute to log potential type errors
        self.classification_dict = classification_dict
        self.classification_preprocessing_dict = classification_preprocessing_dict

    def __call__(self, data: Data):
        data.y = self.transform(data.y)
        return data

    def transform(self, y):
        y = self.preprocessing_mapper(y)
        try:
            y = self.mapper(y)
        except TypeError as e:
            log.error(
                "A TypeError occured when mapping target from arbitrary integers "
                "to consecutive integers (0-(n-1)) using the provided classification_dict "
                "This usually happens when an unknown classification code was encounterd. "
                "Check that all classification codes in your data are either "
                "specified via the classification_dict "
                "or transformed into a specified code via the preprocessing_mapper. \n"
                f"Current classification_dict: \n{self.classification_dict}\n"
                f"Current preprocessing_mapper: \n{self.classification_preprocessing_dict}\n"
                f"Current unique values in preprocessed target array: \n{np.unique(y)}\n"
            )
            raise e
        return torch.LongTensor(y)

    def _set_preprocessing_mapper(self, classification_preprocessing_dict):
        """Set mapper from source classification code to another code."""
        d = {key: value for key, value in classification_preprocessing_dict.items()}
        self.preprocessing_mapper = np.vectorize(lambda class_code: d.get(class_code, class_code))

    def _set_mapper(self, classification_dict):
        """Set mapper from source classification code to consecutive integers."""
        d = {
            class_code: class_index
            for class_index, class_code in enumerate(classification_dict.keys())
        }
        # Here we update the dict so that code 65 remains unchanged.
        # Indeed, 65 is reserved for noise/artefacts points, that will be deleted by transform "DropPointsByClass".
        d.update({65: 65})
        self.mapper = np.vectorize(lambda class_code: d.get(class_code))

class TargetToOneHot(BaseTransform):
    """Convert target to one-hot encoding."""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, data):
        return data
    
        # To one hot 
        data.y = torch.nn.functional.one_hot(data.y, num_classes=self.num_classes+1)
        
        # Remove the void class
        data.y = data.y[:,:-1]
        
        # Convert to float, for the loss function
        data.y = data.y.float()
        return data

class DropPointsByClass(BaseTransform):
    """Drop points with class -1 (i.e. artefacts that would have been mapped to code -1)"""

    def __call__(self, data):
        points_to_drop = torch.isin(data.y, COMMON_CODE_FOR_ALL_ARTEFACTS)
        if points_to_drop.sum() > 0:
            points_to_keep = torch.logical_not(points_to_drop)
            data = subsample_data(data, num_nodes=data.num_nodes, choice=points_to_keep)
            # Here we also subsample these idx since we do not need to interpolate these points back
            # It supposes that DropPointsByClass is run before copying the original point cloud
            if "idx_in_original_cloud" in data:
                data.idx_in_original_cloud = data.idx_in_original_cloud[points_to_keep.numpy()]

        return data
