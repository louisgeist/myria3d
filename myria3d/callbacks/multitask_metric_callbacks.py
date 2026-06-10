from typing import Dict, List, Optional

import torch
from pytorch_lightning import Callback
from torchmetrics import JaccardIndex, MeanAbsoluteError, MeanSquaredError

from myria3d.pctl.dataset.flair3d_label_remap import (
    DEFAULT_LABEL_DEFINITION_NAMES,
    get_definition,
)


def _class_names_for_task(
    task_name: str,
    task_config: dict,
    classification_dict: Optional[Dict[int, str]] = None,
) -> List[str]:
    if task_name == "segment" and classification_dict:
        return [
            classification_dict[int(class_id)]
            for class_id in sorted(classification_dict.keys(), key=int)
        ]
    if task_name in DEFAULT_LABEL_DEFINITION_NAMES:
        defn_name = DEFAULT_LABEL_DEFINITION_NAMES[task_name]
        return list(get_definition(task_name, defn_name).names)
    return [str(class_id) for class_id in range(int(task_config["num_classes"]))]


class MultiTaskMetrics(Callback):
    """Compute per-task metrics for multitask segmentation and elevation regression."""

    def __init__(
        self,
        task_configs: dict,
        main_task: str = "segment",
        elevation_target_scale: float = 0.01,
        classification_dict: Optional[Dict[int, str]] = None,
    ):
        self.task_configs = task_configs
        self.main_task = main_task
        self.elevation_target_scale = elevation_target_scale
        self.classification_dict = classification_dict or {}
        self.best_val_iou = 0.0

        self.semantic_iou: Dict[str, Dict[str, JaccardIndex]] = {
            "train": {},
            "val": {},
            "test": {},
        }
        self.semantic_iou_by_class: Dict[str, JaccardIndex] = {}
        self.class_names_by_task: Dict[str, List[str]] = {}
        for task_name, task_config in task_configs.items():
            if task_config.get("task_type", "semantic") != "semantic":
                continue
            num_classes = int(task_config["num_classes"])
            ignore_index = int(task_config.get("ignore_index", num_classes - 1))
            for phase in self.semantic_iou:
                self.semantic_iou[phase][task_name] = JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                    ignore_index=ignore_index,
                )
            self.class_names_by_task[task_name] = _class_names_for_task(
                task_name,
                task_config,
                self.classification_dict,
            )
            self.semantic_iou_by_class[task_name] = JaccardIndex(
                task="multiclass",
                num_classes=num_classes,
                average=None,
                ignore_index=ignore_index,
            )

        self.elevation_mae = MeanAbsoluteError()
        self.elevation_rmse = MeanSquaredError(squared=False)

    def _end_of_batch(self, phase: str, outputs: dict):
        preds_by_task = outputs["outputs"]
        targets_by_task = outputs["targets"]

        for task_name, metric in self.semantic_iou[phase].items():
            preds = preds_by_task[task_name]
            targets = targets_by_task.get(task_name)
            if targets is None:
                continue
            pred_labels = torch.argmax(preds.detach(), dim=1)
            targets = targets.long().to(pred_labels.device)
            metric.to(pred_labels.device)(pred_labels, targets)
            if phase == "test" and task_name in self.semantic_iou_by_class:
                self.semantic_iou_by_class[task_name].to(pred_labels.device)(
                    pred_labels,
                    targets,
                )

        if "elevation" in preds_by_task and targets_by_task.get("elevation") is not None:
            preds = preds_by_task["elevation"].detach()
            targets = targets_by_task["elevation"].to(preds.device)
            valid_mask = torch.isfinite(targets)
            if valid_mask.any():
                scaled_preds = preds[valid_mask] / self.elevation_target_scale
                scaled_targets = targets[valid_mask] / self.elevation_target_scale
                self.elevation_mae.to(preds.device)(scaled_preds, scaled_targets)
                self.elevation_rmse.to(preds.device)(scaled_preds, scaled_targets)

    def _end_of_epoch(self, phase: str, pl_module):
        for task_name, metric in self.semantic_iou[phase].items():
            value = metric.to(pl_module.device).compute()
            metric_name = (
                f"{phase}/iou"
                if task_name == self.main_task
                else f"{phase}/iou_{task_name}"
            )
            pl_module.log(metric_name, value, on_epoch=True, on_step=False)
            if phase == "val" and task_name == self.main_task:
                self.best_val_iou = max(self.best_val_iou, float(value))
                pl_module.log(
                    "val/best_iou",
                    self.best_val_iou,
                    on_epoch=True,
                    on_step=False,
                    metric_attribute="val/best_iou",
                )
            metric.reset()

        if phase == "test":
            for task_name, metric in self.semantic_iou_by_class.items():
                values = metric.to(pl_module.device).compute()
                metric_prefix = (
                    f"{phase}/iou"
                    if task_name == self.main_task
                    else f"{phase}/iou_{task_name}"
                )
                for value, class_name in zip(values, self.class_names_by_task[task_name]):
                    pl_module.log(
                        f"{metric_prefix}/{class_name}",
                        value,
                        on_epoch=True,
                        on_step=False,
                    )
                metric.reset()

        if getattr(self.elevation_mae, "_update_called", False):
            pl_module.log(
                f"{phase}/elevation_mae",
                self.elevation_mae.compute(),
                on_epoch=True,
                on_step=False,
            )
            pl_module.log(
                f"{phase}/elevation_rmse",
                self.elevation_rmse.compute(),
                on_epoch=True,
                on_step=False,
            )
            self.elevation_mae.reset()
            self.elevation_rmse.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end_of_batch("train", outputs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end_of_batch("val", outputs)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end_of_batch("test", outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        self._end_of_epoch("train", pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._end_of_epoch("val", pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self._end_of_epoch("test", pl_module)
