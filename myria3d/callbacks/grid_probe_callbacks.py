"""Metric computation and winner/seed-ensemble bookkeeping for GridProbeModel.

Kept out of the LightningModule per this repo's convention (see CLAUDE.md /
metric_callbacks.py / multitask_metric_callbacks.py): `GridProbeMetrics` computes
per-probe metrics, `GridProbeWinnerSelector` closes the LR-sweep phase by picking a
winner, `GridProbeSeedEnsembleTester` closes the seed phase by aggregating the 10
seed-head test results.
"""

from __future__ import annotations

import csv
import json
import os.path as osp
import re
from typing import Dict, Optional, Sequence

import torch
from pytorch_lightning import Callback
from torchmetrics import Accuracy, F1Score, JaccardIndex

from myria3d.utils import utils

log = utils.get_logger(__name__)

_PHASES = ("train", "val", "test")
# metric key -> logged tag suffix (val/probe_{name}/{tag}).
_METRIC_TAGS = {"iou": "mIoU", "f1": "macro_f1", "acc": "acc"}


def _class_iou_tag(class_name: str) -> str:
    """``"Non-ground"`` -> ``"iou_non_ground"``."""
    return "iou_" + re.sub(r"[^0-9a-z]+", "_", class_name.lower()).strip("_")


class GridProbeMetrics(Callback):
    """Per-probe mIoU / macro-F1 / accuracy, logged as ``{phase}/probe_{name}/{tag}``.

    Probe names aren't known until `GridProbeModel` is instantiated, so per-probe
    `torchmetrics` objects are created lazily on first use rather than upfront.

    When ``class_names`` (one name per class id) is given, the per-class IoU is also
    logged as ``{phase}/probe_{name}/iou_{class}`` (e.g. ``iou_ground``) and averaged
    over probes as ``{phase}/probes_mean/iou_{class}``; the ``ignore_index`` class, if
    any, is skipped. Off by default (``None``): existing datasets log the three
    aggregate metrics only.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError(
                f"class_names has {len(class_names)} entries but num_classes={num_classes}"
            )
        self.class_iou_tags: Dict[int, str] = {
            class_id: _class_iou_tag(name)
            for class_id, name in enumerate(class_names or [])
            if class_id != ignore_index
        }
        self._metrics: Dict[str, Dict[str, dict]] = {phase: {} for phase in _PHASES}

    def _metrics_for(self, phase: str, probe_name: str) -> dict:
        probes = self._metrics[phase]
        if probe_name not in probes:
            probes[probe_name] = {
                "iou": JaccardIndex(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="macro",
                    ignore_index=self.ignore_index,
                ),
                "f1": F1Score(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="macro",
                    ignore_index=self.ignore_index,
                ),
                "acc": Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="micro",
                    ignore_index=self.ignore_index,
                ),
            }
            if self.class_iou_tags:
                probes[probe_name]["class_iou"] = JaccardIndex(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="none",
                    ignore_index=self.ignore_index,
                )
        return probes[probe_name]

    def _end_of_batch(self, phase: str, outputs: dict) -> None:
        logits_by_probe = outputs["logits"]
        targets = outputs["targets"].long()
        for probe_name, logits in logits_by_probe.items():
            preds = torch.argmax(logits.detach(), dim=1)
            metrics = self._metrics_for(phase, probe_name)
            for metric in metrics.values():
                metric.to(preds.device)(preds, targets.to(preds.device))

    def _end_of_epoch(self, phase: str, pl_module) -> None:
        per_tag_values: Dict[str, list] = {tag: [] for tag in _METRIC_TAGS.values()}
        per_tag_values.update({tag: [] for tag in self.class_iou_tags.values()})
        for probe_name, metrics in self._metrics[phase].items():
            for metric_key, metric in metrics.items():
                value = metric.to(pl_module.device).compute()
                if metric_key == "class_iou":
                    named_values = {
                        tag: value[class_id] for class_id, tag in self.class_iou_tags.items()
                    }
                else:
                    named_values = {_METRIC_TAGS[metric_key]: value}
                for tag, tag_value in named_values.items():
                    pl_module.log(
                        f"{phase}/probe_{probe_name}/{tag}",
                        tag_value,
                        on_epoch=True,
                        on_step=False,
                    )
                    per_tag_values[tag].append(tag_value)
                metric.reset()
        # Mean across all active probes -- gives ModelCheckpoint/EarlyStopping a
        # single well-defined validation signal (the actual select_metric, not
        # val/loss) even though each probe otherwise only logs its own key.
        for tag, values in per_tag_values.items():
            if values:
                pl_module.log(
                    f"{phase}/probes_mean/{tag}",
                    torch.stack(values).mean(),
                    on_epoch=True,
                    on_step=False,
                )

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


class GridProbeWinnerSelector(Callback):
    """End of the grid-phase fit: pick the probe (LR) with the best running-best
    ``select_metric`` on val, write ``grid_search_results.json``, and set
    ``pl_module.active_probe`` to the winner so the automatic post-fit
    ``trainer.test()`` (see ``myria3d/train.py``) only scores it.

    ``select_metric``: ``"mIoU"`` for DALES/ECLAIR, ``"macro_f1"`` for H3D (dataset
    convention from the Pointcept protocol -- see ``dataset_description.select_metric``).
    """

    def __init__(self, select_metric: str = "mIoU", output_path: str = "grid_search_results.json"):
        if select_metric not in ("mIoU", "macro_f1"):
            raise ValueError(f"select_metric must be 'mIoU' or 'macro_f1', got {select_metric!r}")
        self.select_metric = select_metric
        self.output_path = output_path
        self._best_per_probe: Dict[str, float] = {}

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        for probe_name in pl_module.probe_names:
            key = f"val/probe_{probe_name}/{self.select_metric}"
            value = trainer.callback_metrics.get(key)
            if value is None:
                continue
            value = float(value)
            self._best_per_probe[probe_name] = max(
                value, self._best_per_probe.get(probe_name, float("-inf"))
            )

    def on_fit_end(self, trainer, pl_module) -> None:
        if not self._best_per_probe:
            log.warning(
                "GridProbeWinnerSelector: no val/probe_*/%s metrics observed; skipping.",
                self.select_metric,
            )
            return

        winner_name = max(self._best_per_probe, key=self._best_per_probe.get)
        winner_lr = pl_module.probe_lrs[winner_name]
        winner_metric = self._best_per_probe[winner_name]

        leaderboard = [
            {"probe": name, "lr": pl_module.probe_lrs[name], self.select_metric: value}
            for name, value in sorted(
                self._best_per_probe.items(), key=lambda kv: kv[1], reverse=True
            )
        ]
        results = {
            "select_metric": self.select_metric,
            "winner": {"probe": winner_name, "lr": winner_lr, self.select_metric: winner_metric},
            "leaderboard": leaderboard,
        }
        if trainer.is_global_zero:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            log.info(
                "GridProbeWinnerSelector: winner=%s (lr=%s, %s=%.4f) -> %s",
                winner_name,
                winner_lr,
                self.select_metric,
                winner_metric,
                self.output_path,
            )

        # Narrows the automatic post-fit `trainer.test()` call (train.py) to the
        # winner only -- harmless no-op for the seed phase, which never sets this.
        pl_module.active_probe = winner_name


class GridProbeSeedEnsembleTester(Callback):
    """End of the seed-phase test (all 10 seed heads active): aggregate
    mean/std/min/max test mIoU/macro-F1/accuracy per metric across seed heads into
    ``seed_ensemble_results.json`` and append one row to
    ``grid_then_seeds_summary.csv``.
    """

    def __init__(
        self,
        results_path: str = "seed_ensemble_results.json",
        summary_csv_path: str = "grid_then_seeds_summary.csv",
        dataset_name: Optional[str] = None,
    ):
        self.results_path = results_path
        self.summary_csv_path = summary_csv_path
        self.dataset_name = dataset_name

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics
        per_seed: Dict[str, Dict[str, float]] = {}
        for probe_name in pl_module.probe_names:
            entry = {}
            prefix = f"test/probe_{probe_name}/"
            # Aggregate tags plus any per-class ``iou_*`` GridProbeMetrics logged.
            tags = list(_METRIC_TAGS.values()) + sorted(
                key[len(prefix) :] for key in metrics if key.startswith(prefix + "iou_")
            )
            for tag in tags:
                key = f"{prefix}{tag}"
                if key in metrics:
                    entry[tag] = float(metrics[key])
            if entry:
                per_seed[probe_name] = entry
        if not per_seed:
            log.warning(
                "GridProbeSeedEnsembleTester: no test/probe_*/* metrics observed; skipping."
            )
            return

        aggregate: Dict[str, Dict[str, float]] = {}
        all_tags = list(dict.fromkeys(tag for entry in per_seed.values() for tag in entry))
        for tag in all_tags:
            values = [entry[tag] for entry in per_seed.values() if tag in entry]
            if not values:
                continue
            values_t = torch.tensor(values)
            aggregate[tag] = {
                "mean": float(values_t.mean()),
                "std": float(values_t.std(unbiased=True)) if len(values) > 1 else 0.0,
                "min": float(values_t.min()),
                "max": float(values_t.max()),
                "n": len(values),
            }

        results = {"per_seed": per_seed, "aggregate": aggregate}
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        row = {"dataset": self.dataset_name or ""}
        for tag, stats in aggregate.items():
            for stat_name, stat_value in stats.items():
                row[f"{tag}_{stat_name}"] = stat_value
        write_header = not osp.isfile(self.summary_csv_path)
        with open(self.summary_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        log.info(
            "GridProbeSeedEnsembleTester: wrote %s and appended a row to %s",
            self.results_path,
            self.summary_csv_path,
        )
