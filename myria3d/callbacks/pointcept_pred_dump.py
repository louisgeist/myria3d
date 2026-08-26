"""Dump pixel-semantic predictions in Pointcept PreciseEvaluator format.

Pointcept's ``eval_network_apls.py`` consumes ``{patch_id}_logits_network.npy``:
a ``(C, H, W)`` float32 raster of **soft foreground probabilities** in ``[0, 1]``,
with NaN on 1 m Lambert cells that had no LiDAR point (unobserved). ``C=1`` for
the roads-only head used here (Pointcept ``num_networks=1``).

myria3d evaluates 50 m quadrants of each 100 m Pointcept patch. This callback
scatters cell-pooled predictions back onto the full-patch raster (same ``H, W``
as ``meta.json['network']``) and nanmean-merges overlapping quadrants / DDP ranks
before writing one file per patch, matching Pointcept's ``save_path/result``
layout.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.core.module import LightningModule

from myria3d.models.modules.pixel_pooling import (
    pool_points_by_cell,
    scatter_cell_values_to_raster,
)
from myria3d.utils import utils

log = utils.get_logger(__name__)

# myria3d task name -> Pointcept dump stem (``{patch_id}_logits_{stem}.npy``).
DEFAULT_PIXEL_SEMANTIC_DUMP_NAMES = {
    "roads": "network",
    "forest_2d": "forest_2d",
}


def patch_ids_from_batch(batch, num_graphs: int) -> List[str]:
    """Recover per-graph Pointcept patch ids after PyG collate (list of strings)."""
    value = getattr(batch, "patch_id", None)
    if value is None:
        raise AttributeError(
            "Batch is missing `patch_id`. PointceptNpyDataset must attach it."
        )
    if isinstance(value, str):
        if num_graphs != 1:
            raise ValueError(
                f"Got a single patch_id string but batch has {num_graphs} graphs."
            )
        return [value]
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                if not item:
                    raise ValueError("Empty nested patch_id entry in batch.")
                out.append(str(item[0]))
            else:
                out.append(str(item))
        if len(out) != num_graphs:
            raise ValueError(
                f"Expected {num_graphs} patch ids after collate, got {len(out)}."
            )
        return out
    raise TypeError(f"Unsupported batch.patch_id type: {type(value)!r}")


def merge_dense_probability_rasters(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Nanmean-merge ``(C, H, W)`` probability rasters (unobserved = NaN)."""
    if not arrays:
        raise ValueError("merge_dense_probability_rasters requires at least one array.")
    first = np.asarray(arrays[0])
    if first.ndim != 3:
        raise ValueError(f"Expected (C, H, W) rasters, got shape {first.shape}")
    sum_ = np.zeros(first.shape, dtype=np.float64)
    count = np.zeros(first.shape, dtype=np.int32)
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.shape != first.shape:
            raise ValueError(
                f"Raster shape mismatch: {arr.shape} vs {first.shape}"
            )
        finite = np.isfinite(arr)
        sum_[finite] += arr[finite]
        count[finite] += 1
    out = np.full(first.shape, np.nan, dtype=np.float32)
    observed = count > 0
    out[observed] = (sum_[observed] / count[observed]).astype(np.float32)
    return out


class _DenseRasterAccum:
    """Running nanmean of per-cell foreground probs onto a ``(C, H, W)`` grid."""

    def __init__(self, channels: int, height: int, width: int):
        self.sum = np.zeros((channels, height, width), dtype=np.float64)
        self.count = np.zeros((channels, height, width), dtype=np.int32)

    @property
    def shape(self):
        return self.sum.shape

    def add_hw(self, channel: int, raster_hw: np.ndarray) -> None:
        expected = self.sum.shape[1:]
        if raster_hw.shape != expected:
            raise ValueError(
                f"Raster HW {raster_hw.shape} does not match accum {expected}"
            )
        finite = np.isfinite(raster_hw)
        self.sum[channel][finite] += raster_hw[finite]
        self.count[channel][finite] += 1

    def finalize(self) -> np.ndarray:
        out = np.full(self.sum.shape, np.nan, dtype=np.float32)
        observed = self.count > 0
        out[observed] = (self.sum[observed] / self.count[observed]).astype(np.float32)
        return out


class PointceptPredictionDump(Callback):
    """Write Pointcept-format dense pixel-semantic logits during test (opt-in val)."""

    def __init__(
        self,
        task_configs: dict,
        output_dir: str = "result",
        phases: Sequence[str] = ("test",),
        dump_names: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.task_configs = dict(task_configs)
        self.output_dir = str(output_dir)
        self.phases = tuple(phases)
        names = dump_names if dump_names is not None else DEFAULT_PIXEL_SEMANTIC_DUMP_NAMES
        self.dump_names = {str(k): str(v) for k, v in dict(names).items()}
        self.enabled = bool(enabled)
        self._tasks = tuple(
            task_name
            for task_name, dump_name in self.dump_names.items()
            if self.task_configs.get(task_name, {}).get("task_type") == "pixel_semantic"
            and dump_name
        )
        self._acc: Dict[str, Dict[str, Dict[str, _DenseRasterAccum]]] = {
            phase: defaultdict(dict) for phase in self.phases
        }

    def _phase_enabled(self, trainer: Trainer, phase: str) -> bool:
        return (
            self.enabled
            and phase in self.phases
            and not bool(getattr(trainer, "sanity_checking", False))
        )

    def _output_path(self) -> Path:
        path = Path(self.output_dir)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _update_batch(self, phase: str, outputs: dict, batch) -> None:
        if batch is None or not isinstance(outputs, dict):
            return
        preds_by_task = outputs.get("outputs") or {}
        targets_by_task = outputs.get("targets") or {}
        num_graphs = int(getattr(batch, "num_graphs", 0) or 0)
        if num_graphs <= 0 and getattr(batch, "batch", None) is not None:
            num_graphs = int(batch.batch.max().item()) + 1
        if num_graphs <= 0:
            return
        patch_ids = patch_ids_from_batch(batch, num_graphs)

        for task_name in self._tasks:
            preds = preds_by_task.get(task_name)
            targets = targets_by_task.get(task_name)
            cell_id = getattr(batch, f"{task_name}_cell_id", None)
            if preds is None or targets is None or cell_id is None:
                continue
            task_config = self.task_configs[task_name]
            pooling = task_config.get("pooling", "mean")
            fg_index = int(task_config.get("fg_index", 1))
            pooled_preds, _, pooled_cell_id, pooled_batch = pool_points_by_cell(
                preds.detach(),
                cell_id.to(preds.device),
                targets.to(preds.device),
                batch.batch.to(preds.device),
                pooling=pooling,
            )
            fg_probs = None
            if pooled_preds.size(0) > 0:
                fg_probs = torch.softmax(pooled_preds.float(), dim=-1)[:, fg_index]

            acc_by_patch = self._acc[phase][task_name]
            for graph_idx, patch_id in enumerate(patch_ids):
                height_t = getattr(batch, f"{task_name}_raster_h", None)
                width_t = getattr(batch, f"{task_name}_raster_w", None)
                if height_t is None or width_t is None:
                    continue
                height = int(height_t.reshape(-1)[graph_idx].item())
                width = int(width_t.reshape(-1)[graph_idx].item())
                if height <= 0 or width <= 0 or not patch_id:
                    continue
                accum = acc_by_patch.get(patch_id)
                if accum is None:
                    accum = _DenseRasterAccum(1, height, width)
                    acc_by_patch[patch_id] = accum
                elif accum.shape != (1, height, width):
                    raise ValueError(
                        f"{patch_id}/{task_name}: raster shape mismatch "
                        f"{accum.shape} vs (1, {height}, {width})"
                    )
                if fg_probs is None:
                    continue
                sel = pooled_batch == graph_idx
                if not bool(sel.any()):
                    continue
                raster_hw = scatter_cell_values_to_raster(
                    fg_probs[sel], pooled_cell_id[sel], height, width
                )
                accum.add_hw(0, raster_hw.detach().cpu().numpy())

    def _flush_phase(self, trainer: Trainer, phase: str) -> None:
        acc_by_task = self._acc.get(phase)
        if not acc_by_task:
            return
        output_dir = self._output_path()
        rank = int(getattr(trainer, "global_rank", 0) or 0)
        world_size = int(getattr(trainer, "world_size", 1) or 1)
        is_zero = bool(getattr(trainer, "is_global_zero", rank == 0))

        if world_size == 1:
            write_dir = output_dir
        else:
            write_dir = output_dir / f".rank{rank}"
        write_dir.mkdir(parents=True, exist_ok=True)

        n_written = 0
        for task_name, patches in acc_by_task.items():
            dump_name = self.dump_names[task_name]
            for patch_id, accum in patches.items():
                np.save(write_dir / f"{patch_id}_logits_{dump_name}.npy", accum.finalize())
                n_written += 1
        self._acc[phase] = defaultdict(dict)

        if world_size > 1:
            trainer.strategy.barrier()
            if not is_zero:
                return
            n_written = _merge_rank_dump_dirs(output_dir, world_size)
            for rank_idx in range(world_size):
                shutil.rmtree(output_dir / f".rank{rank_idx}", ignore_errors=True)

        log.info(
            "PointceptPredictionDump: wrote %d raster(s) to %s (%s)",
            n_written,
            output_dir,
            phase,
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._phase_enabled(trainer, "val"):
            self._update_batch("val", outputs, batch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._phase_enabled(trainer, "test"):
            self._update_batch("test", outputs, batch)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self._phase_enabled(trainer, "val"):
            self._flush_phase(trainer, "val")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self._phase_enabled(trainer, "test"):
            self._flush_phase(trainer, "test")


def _merge_rank_dump_dirs(output_dir: Path, world_size: int) -> int:
    """Nanmean-merge per-rank rasters into ``output_dir/{patch_id}_logits_*.npy``."""
    grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
    for rank_idx in range(world_size):
        rank_dir = output_dir / f".rank{rank_idx}"
        if not rank_dir.is_dir():
            continue
        for path in rank_dir.glob("*_logits_*.npy"):
            grouped[path.name].append(np.load(path))
    n_written = 0
    for filename, arrays in grouped.items():
        merged = arrays[0] if len(arrays) == 1 else merge_dense_probability_rasters(arrays)
        np.save(output_dir / filename, merged)
        n_written += 1
    return n_written
