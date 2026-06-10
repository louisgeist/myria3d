from typing import Dict, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import knn
from torch_geometric.utils import scatter

from myria3d.models.model import MODEL_ZOO, get_neural_net_class
from myria3d.models.modules.pyg_randla_net_multitask import PyGRandLANetMultiTask
from myria3d.utils import utils

log = utils.get_logger(__name__)

MODEL_ZOO.append(PyGRandLANetMultiTask)

SEMANTIC_BATCH_TARGETS = {
    "segment": "y",
    "forest": "y_forest",
    "land_use": "y_land_use",
    "natural_habitat": "y_natural_habitat",
}
REGRESSION_BATCH_TARGETS = {
    "elevation": "y_elevation",
}


def _compute_shared_knn_interpolation_weights(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    k: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute KNN neighbor indices and weights once for a fixed geometry pair."""
    with torch.no_grad():
        print(f"pos_x: {pos_x.shape}")
        print(f"pos_y: {pos_y.shape}")
        assign_index = knn(
            pos_x,
            pos_y,
            k,
            batch_x=batch_x,
            batch_y=batch_y,
            num_workers=num_workers,
        )
        y_idx, x_idx = assign_index[0], assign_index[1]
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
    return y_idx, x_idx, weights


def _apply_knn_interpolation(
    x: torch.Tensor,
    y_idx: torch.Tensor,
    x_idx: torch.Tensor,
    weights: torch.Tensor,
    num_target_points: int,
) -> torch.Tensor:
    """Interpolate features using precomputed KNN indices and weights."""
    y = scatter(x[x_idx] * weights, y_idx, 0, num_target_points, reduce="sum")
    return y / scatter(weights, y_idx, 0, num_target_points, reduce="sum")


class MultiTaskModel(LightningModule):
    """Lightning module for Flair3D+ multitask point cloud learning."""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["criteria"])

        neural_net_class = get_neural_net_class(kwargs.get("neural_net_class_name"))
        self.model = neural_net_class(**kwargs.get("neural_net_hparams"))

        self.task_configs = kwargs.get("task_configs", {})
        self.task_weights = kwargs.get("task_weights", {})
        self.main_task = kwargs.get("main_task", "segment")
        self.elevation_target_scale = float(kwargs.get("elevation_target_scale", 0.01))

        criteria_cfg = kwargs.get("criteria", {})
        self.criteria = nn.ModuleDict()
        for task_name in self.task_configs:
            if task_name not in criteria_cfg:
                continue
            criterion_spec = criteria_cfg[task_name]
            if isinstance(criterion_spec, (dict, DictConfig)):
                self.criteria[task_name] = hydra.utils.instantiate(criterion_spec)
            else:
                self.criteria[task_name] = criterion_spec

    def _task_type(self, task_name: str) -> str:
        return self.task_configs[task_name].get("task_type", "semantic")

    def _get_batch_target(self, batch: Batch, task_name: str) -> Optional[torch.Tensor]:
        if task_name in SEMANTIC_BATCH_TARGETS:
            key = SEMANTIC_BATCH_TARGETS[task_name]
            return getattr(batch, key, None)
        if task_name in REGRESSION_BATCH_TARGETS:
            key = REGRESSION_BATCH_TARGETS[task_name]
            return getattr(batch, key, None)
        return None

    def _get_copy_target(self, batch: Batch, task_name: str) -> Optional[torch.Tensor]:
        if "copies" not in batch:
            return None
        if task_name in SEMANTIC_BATCH_TARGETS:
            key = SEMANTIC_BATCH_TARGETS[task_name]
            copy_key = f"transformed_{key}_copy"
            if copy_key in batch.copies:
                return batch.copies[copy_key]
            if key == "y":
                return batch.copies.get("transformed_y_copy")
        if task_name in REGRESSION_BATCH_TARGETS:
            key = REGRESSION_BATCH_TARGETS[task_name]
            return batch.copies.get(f"transformed_{key}_copy")
        return None

    def _interpolate_outputs(
        self, outputs: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        batch_y = self._get_batch_tensor_by_enumeration(batch.idx_in_original_cloud)
        pos_x = batch.copies["pos_sampled_copy"].cpu()
        pos_y = batch.copies["pos_copy"].cpu()
        y_idx, x_idx, weights = _compute_shared_knn_interpolation_weights(
            pos_x,
            pos_y,
            batch.batch.cpu(),
            batch_y.cpu(),
            k=self.hparams.interpolation_k,
            num_workers=self.hparams.num_workers,
        )
        num_target_points = pos_y.size(0)

        interpolated = {}
        for task_name, preds in outputs.items():
            preds_cpu = preds.cpu()
            if self._task_type(task_name) == "semantic":
                features = preds_cpu
            else:
                features = preds_cpu.unsqueeze(-1)
            result = _apply_knn_interpolation(
                features, y_idx, x_idx, weights, num_target_points
            )
            if self._task_type(task_name) != "semantic":
                result = result.squeeze(-1)
            interpolated[task_name] = result
        return interpolated

    def forward(
        self, batch: Batch
    ) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor]]:
        outputs = self.model(batch.x, batch.pos, batch.batch, batch.ptr)

        interpolate_at_eval = bool(getattr(self.hparams, "interpolate_at_eval", True))
        if self.training or "copies" not in batch or not interpolate_at_eval:
            targets = {
                task_name: self._get_batch_target(batch, task_name)
                for task_name in self.task_configs
            }
            return targets, outputs

        outputs = self._interpolate_outputs(outputs, batch)
        # knn_interpolate runs on CPU (see model.py); copy targets must match output device.
        output_device = next(iter(outputs.values())).device
        targets = {}
        for task_name in self.task_configs:
            target = self._get_copy_target(batch, task_name)
            targets[task_name] = target.to(output_device) if target is not None else None
        return targets, outputs

    def _compute_task_loss(
        self,
        task_name: str,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        task_config = self.task_configs[task_name]
        criterion = self.criteria[task_name].to(preds.device)
        targets = targets.to(preds.device)

        if self._task_type(task_name) == "semantic":
            return criterion(preds, targets.long())

        valid_mask = torch.isfinite(targets)
        if not valid_mask.any():
            return preds.new_zeros(())
        scaled_preds = preds[valid_mask] / self.elevation_target_scale
        scaled_targets = targets[valid_mask] / self.elevation_target_scale
        return criterion(scaled_preds, scaled_targets)

    def _compute_loss(
        self,
        targets: Dict[str, Optional[torch.Tensor]],
        outputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        total_loss = None
        for task_name, preds in outputs.items():
            task_targets = targets.get(task_name)
            if task_targets is None:
                continue
            task_loss = self._compute_task_loss(task_name, preds, task_targets)
            weight = float(self.task_weights.get(task_name, 1.0))
            weighted_loss = task_loss * weight
            losses[task_name] = task_loss
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        if total_loss is None:
            total_loss = next(iter(outputs.values())).new_zeros(())
        return total_loss, losses

    def training_step(self, batch: Batch, batch_idx: int) -> dict:
        targets, outputs = self.forward(batch)
        loss, losses = self._compute_loss(targets, outputs)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        for task_name, task_loss in losses.items():
            self.log(f"train/loss_{task_name}", task_loss, on_step=False, on_epoch=True)
        return {
            "loss": loss,
            "outputs": outputs,
            "targets": targets,
        }

    def validation_step(self, batch: Batch, batch_idx: int) -> dict:
        targets, outputs = self.forward(batch)
        loss, losses = self._compute_loss(targets, outputs)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        for task_name, task_loss in losses.items():
            self.log(f"val/loss_{task_name}", task_loss, on_step=False, on_epoch=True)
        return {
            "loss": loss,
            "outputs": outputs,
            "targets": targets,
        }

    def test_step(self, batch: Batch, batch_idx: int):
        targets, outputs = self.forward(batch)
        loss, losses = self._compute_loss(targets, outputs)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        for task_name, task_loss in losses.items():
            self.log(f"test/loss_{task_name}", task_loss, on_step=False, on_epoch=True)
        return {
            "loss": loss,
            "outputs": outputs,
            "targets": targets,
        }

    def predict_step(self, batch: Batch) -> dict:
        _, outputs = self.forward(batch)
        return {"outputs": {k: v.detach().cpu() for k, v in outputs.items()}}

    def configure_optimizers(self):
        self.lr = self.hparams.lr
        optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )
        if self.hparams.lr_scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": self.hparams.lr_scheduler(optimizer),
            "monitor": self.hparams.monitor,
        }

    def _get_batch_tensor_by_enumeration(self, pos_x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.full((len(sample_pos),), i) for i, sample_pos in enumerate(pos_x)])
