from numbers import Number
from typing import Callable, Dict, List, Optional

from pytorch_lightning import LightningDataModule

from myria3d.pctl.dataloader.dataloader import GeometricNoneProofDataloader
from myria3d.pctl.dataloader.iter_limited_sampler import (
    IterLimitedDistributedSampler,
    IterLimitedSampler,
)
from myria3d.pctl.dataset.pointcept_npy import PointceptNpyDataset
from myria3d.pctl.dataset.utils import pre_filter_below_n_points
from myria3d.pctl.transforms.compose import CustomCompose
from myria3d.utils import utils

log = utils.get_logger(__name__)

TRANSFORMS_LIST = List[Callable]


class PointceptNpyDatamodule(LightningDataModule):
    """Datamodule reading Pointcept-preprocessed Flair3D+ .npy scene folders."""

    def __init__(
        self,
        data_root: str,
        csv_manifest: str,
        excluded_tiles_details_csv: Optional[str] = None,
        too_small_tiles_manifest: Optional[str] = None,
        tile_width: Number = 100,
        subtile_width: Number = 50,
        subtile_overlap: Number = 0,
        pre_filter: Optional[Callable] = pre_filter_below_n_points,
        batch_size: int = 12,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        iter_per_epoch: Optional[int] = None,
        seed: int = 0,
        transforms: Optional[Dict[str, TRANSFORMS_LIST]] = None,
        **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.csv_manifest = csv_manifest
        self.excluded_tiles_details_csv = excluded_tiles_details_csv
        self.too_small_tiles_manifest = too_small_tiles_manifest
        self.tile_width = tile_width
        self.subtile_width = subtile_width
        self.subtile_overlap = subtile_overlap
        self.pre_filter = pre_filter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.iter_per_epoch = iter_per_epoch
        self.seed = seed
        self._dataset = None

        t = transforms or {}
        self.preparation_train_transform: TRANSFORMS_LIST = t.get("preparations_train_list", [])
        self.preparation_eval_transform: TRANSFORMS_LIST = t.get("preparations_eval_list", [])
        self.augmentation_transform: TRANSFORMS_LIST = t.get("augmentations_list", [])
        self.normalization_transform: TRANSFORMS_LIST = t.get("normalizations_list", [])

    @property
    def train_transform(self) -> CustomCompose:
        return CustomCompose(
            self.preparation_train_transform
            + self.normalization_transform
            + self.augmentation_transform
        )

    @property
    def eval_transform(self) -> CustomCompose:
        return CustomCompose(self.preparation_eval_transform + self.normalization_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset

    @property
    def dataset(self) -> PointceptNpyDataset:
        if self._dataset:
            return self._dataset

        self._dataset = PointceptNpyDataset(
            data_root=self.data_root,
            csv_manifest=self.csv_manifest,
            excluded_tiles_details_csv=self.excluded_tiles_details_csv,
            too_small_tiles_manifest=self.too_small_tiles_manifest,
            tile_width=self.tile_width,
            subtile_width=self.subtile_width,
            subtile_overlap=self.subtile_overlap,
            pre_filter=self.pre_filter,
            train_transform=self.train_transform,
            eval_transform=self.eval_transform,
        )
        return self._dataset

    def _build_train_sampler(self):
        if self.iter_per_epoch is None:
            return None

        seed = getattr(self.trainer, "global_seed", self.seed) if self.trainer else self.seed
        dataset = self.dataset.traindata
        world_size = getattr(self.trainer, "world_size", 1) if self.trainer else 1
        rank = getattr(self.trainer, "global_rank", 0) if self.trainer else 0

        if world_size > 1:
            return IterLimitedDistributedSampler(
                dataset,
                iter_per_epoch=self.iter_per_epoch,
                batch_size_per_gpu=self.batch_size,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
            )
        return IterLimitedSampler(
            dataset,
            iter_per_epoch=self.iter_per_epoch,
            batch_size=self.batch_size,
            shuffle=True,
            seed=seed,
        )

    def train_dataloader(self):
        sampler = self._build_train_sampler()
        return GeometricNoneProofDataloader(
            dataset=self.dataset.traindata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=self.iter_per_epoch is not None,
        )

    def val_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.dataset.valdata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.dataset.testdata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
