import torch
from torch_geometric.data import Batch, Data

from myria3d.callbacks.multitask_metric_callbacks import MultiTaskMetrics
from myria3d.models.losses import abs_freq_error_rows, tv_from_abs_errors


class _LogCapture:
    def __init__(self):
        self.logged = {}

    @property
    def device(self):
        return torch.device("cpu")

    def log(self, name, value, **kwargs):
        if torch.is_tensor(value):
            value = value.detach().cpu()
            value = float(value.item()) if value.numel() == 1 else value
        self.logged[name] = value


def test_tv_is_point_weighted_mean_of_per_tile_l1():
    """Pointcept: abs_weighted += n_t |pi - q|; TV = (abs_weighted / N).sum()."""
    pi = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    n_t = torch.tensor([2.0, 8.0])
    abs_err = abs_freq_error_rows(pi, q)
    abs_weighted = (n_t.unsqueeze(-1) * abs_err).sum(0)
    tv = float((abs_weighted / n_t.sum()).sum().item())
    expected = float((n_t * tv_from_abs_errors(abs_err)).sum().item() / n_t.sum().item())
    assert abs_err.tolist() == [[0.0, 0.0], [1.0, 1.0]]
    assert abs(tv - 1.6) < 1e-6
    assert abs(tv - expected) < 1e-6


def test_multitask_metrics_log_tv_miou_and_mae_keys():
    task_configs = {
        "segment": {"task_type": "semantic", "num_classes": 3, "ignore_index": 2},
        "nathab_habitat_type": {
            "task_type": "tile_distribution",
            "num_classes": 2,
            "ignore_index": 2,
        },
        "nathab_moisture_regime": {
            "task_type": "tile_distribution",
            "num_classes": 2,
            "ignore_index": 2,
        },
        "elevation": {"task_type": "regression"},
    }
    callback = MultiTaskMetrics(
        task_configs,
        main_task="segment",
        classification_dict={0: "Building", 1: "Water", 2: "Void"},
    )
    logits_match = torch.tensor([[20.0, 0.0], [20.0, 0.0]])
    logits_miss = torch.tensor([[20.0, 0.0]] * 8)
    nathab = torch.cat([logits_match, logits_miss], dim=0)
    batch = Batch.from_data_list(
        [
            Data(
                x=torch.rand(2, 5),
                pos=torch.rand(2, 3),
                y=torch.tensor([0, 0]),
                y_nathab_habitat_type=torch.tensor([0, 0]),
                y_nathab_moisture_regime=torch.tensor([0, 0]),
                y_elevation=torch.tensor([1.0, 3.0]),
            ),
            Data(
                x=torch.rand(8, 5),
                pos=torch.rand(8, 3),
                y=torch.tensor([1] * 8),
                y_nathab_habitat_type=torch.tensor([1] * 8),
                y_nathab_moisture_regime=torch.tensor([1] * 8),
                y_elevation=torch.tensor([0.0] * 8),
            ),
        ]
    )
    segment_logits = torch.zeros(10, 3)
    segment_logits[0, 0] = 10.0
    segment_logits[1, 0] = 10.0
    segment_logits[2:, 1] = 10.0
    outputs = {
        "outputs": {
            "segment": segment_logits,
            "nathab_habitat_type": nathab,
            "nathab_moisture_regime": nathab,
            "elevation": torch.tensor([1.0, 3.0] + [1.0] * 8),
        },
        "targets": {
            "segment": batch.y,
            "nathab_habitat_type": batch.y_nathab_habitat_type,
            "nathab_moisture_regime": batch.y_nathab_moisture_regime,
            "elevation": batch.y_elevation,
        },
    }
    callback._end_of_batch("test", outputs, batch)
    module = _LogCapture()
    callback._end_of_epoch("test", module)

    assert abs(module.logged["test/tv/nathab_habitat_type"] - 1.6) < 1e-5
    assert abs(module.logged["test/tv/nathab_total"] - 3.2) < 1e-5
    assert "test/segment/mIoU" in module.logged
    assert "test/iou" in module.logged
    assert "test/segment/iou/Building" in module.logged
    assert abs(module.logged["test/reg/elevation/mae"] - 0.8) < 1e-5
    assert abs(module.logged["test/elevation_mae"] - 0.8) < 1e-5
