import pytest
import torch
from torch_geometric.data import Batch, Data

from myria3d.callbacks.multitask_metric_callbacks import MultiTaskMetrics


class _FakeModule:
    def __init__(self):
        self.device = torch.device("cpu")
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = float(value)


def _pixel_task_configs():
    return {
        "forest_2d": {
            "task_type": "pixel_semantic",
            "num_classes": 2,
            "ignore_index": 2,
            "pooling": "mean",
            "names": ["Not Forest", "Forest"],
            "enable_dilated_prf": False,
        },
        "roads": {
            "task_type": "pixel_semantic",
            "num_classes": 2,
            "ignore_index": 2,
            "pooling": "max",
            "names": ["Background", "Road"],
            "buffer_radius_px": 3,
            "enable_dilated_prf": True,
        },
    }


def _one_point_per_cell_batch(height, width, gt_rc, pred_rc, task_name):
    n = height * width
    cell_id = torch.arange(n, dtype=torch.long)
    target = torch.zeros(n, dtype=torch.long)
    logits = torch.zeros(n, 2, dtype=torch.float32)
    logits[:, 0] = 1.0
    for row, col in gt_rc:
        target[row * width + col] = 1
    for row, col in pred_rc:
        idx = row * width + col
        logits[idx, 0] = 0.0
        logits[idx, 1] = 1.0
    data = Data(
        pos=torch.zeros(n, 3),
        **{
            f"y_{task_name}": target,
            f"{task_name}_cell_id": cell_id,
            f"{task_name}_raster_h": torch.tensor([height], dtype=torch.long),
            f"{task_name}_raster_w": torch.tensor([width], dtype=torch.long),
        },
    )
    return Batch.from_data_list([data]), logits, target


def test_logs_cell_level_prf_and_dilated_roads_only():
    callback = MultiTaskMetrics(_pixel_task_configs(), main_task="segment")
    # 11x11 grid: GT at center, prediction 3 px away on the diagonal (Chebyshev 3).
    batch, logits, target = _one_point_per_cell_batch(
        11, 11, gt_rc=[(5, 5)], pred_rc=[(8, 8)], task_name="roads"
    )
    outputs = {
        "outputs": {"roads": logits},
        "targets": {"roads": target},
    }
    callback._end_of_batch("val", outputs, batch)
    module = _FakeModule()
    callback._end_of_epoch("val", module)

    assert module.logged["val/roads/precision"] == pytest.approx(0.0, abs=1e-8)
    assert module.logged["val/roads/recall"] == pytest.approx(0.0, abs=1e-8)
    assert module.logged["val/roads/f1"] == pytest.approx(0.0, abs=1e-8)
    assert module.logged["val/roads/dilated_precision"] == pytest.approx(1.0)
    assert module.logged["val/roads/dilated_recall"] == pytest.approx(1.0)
    assert module.logged["val/roads/dilated_f1"] == pytest.approx(1.0)
    assert "val/forest_2d/dilated_f1" not in module.logged


def test_forest_logs_exact_prf_without_dilation():
    callback = MultiTaskMetrics(_pixel_task_configs(), main_task="segment")
    batch, logits, target = _one_point_per_cell_batch(
        3, 3, gt_rc=[(1, 0)], pred_rc=[(1, 1)], task_name="forest_2d"
    )
    outputs = {
        "outputs": {"forest_2d": logits},
        "targets": {"forest_2d": target},
    }
    callback._end_of_batch("val", outputs, batch)
    module = _FakeModule()
    callback._end_of_epoch("val", module)

    assert module.logged["val/forest_2d/precision"] == pytest.approx(0.0, abs=1e-8)
    assert module.logged["val/forest_2d/recall"] == pytest.approx(0.0, abs=1e-8)
    assert module.logged["val/forest_2d/f1"] == pytest.approx(0.0, abs=1e-8)
    assert "val/forest_2d/dilated_precision" not in module.logged
    assert "val/forest_2d/dilated_recall" not in module.logged
    assert "val/forest_2d/dilated_f1" not in module.logged
