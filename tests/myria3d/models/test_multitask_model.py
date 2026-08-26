import functools

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from myria3d.models.multitask_model import MultiTaskModel

TASK_CONFIGS = {
    "segment": {"task_type": "semantic", "num_classes": 4},
    "elevation": {"task_type": "regression"},
}
NUM_FEATURES = 5
GRAD_NORM_LITE_INTERVAL = 3
NUM_BATCHES = GRAD_NORM_LITE_INTERVAL * 2 + 1


class _PrebatchedDataset(Dataset):
    """Wraps already-batched PyG `Batch` objects so DataLoader(batch_size=None) yields them as-is."""

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


def _make_batches(num_batches, n_per_graph=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        data_list = []
        for graph_idx in range(2):
            n = n_per_graph
            data_list.append(
                Data(
                    x=torch.rand((n, NUM_FEATURES), generator=generator),
                    pos=torch.rand((n, 3), generator=generator),
                    y=torch.randint(
                        0, TASK_CONFIGS["segment"]["num_classes"], (n,), generator=generator
                    ),
                    y_elevation=torch.rand((n,), generator=generator) * 10.0,
                    batch=torch.full((n,), graph_idx),
                )
            )
        batches.append(Batch.from_data_list(data_list))
    return batches


def _make_dataloader(batches):
    return DataLoader(_PrebatchedDataset(batches), batch_size=None)


def _make_model(**overrides):
    kwargs = dict(
        neural_net_class_name="PyGRandLANetMultiTask",
        neural_net_hparams=dict(
            num_features=NUM_FEATURES,
            task_configs=TASK_CONFIGS,
            num_neighbors=4,
            decimation=4,
            return_logits=True,
        ),
        task_configs=TASK_CONFIGS,
        task_weights={"segment": 1.0, "elevation": 1.0},
        main_task="segment",
        elevation_target_scale=0.01,
        criteria={
            "segment": torch.nn.CrossEntropyLoss(),
            "elevation": torch.nn.SmoothL1Loss(beta=0.01),
        },
        optimizer=functools.partial(torch.optim.Adam, lr=1e-3),
        lr=1e-3,
        lr_scheduler=None,
        monitor="val/iou",
        grad_norm_lite=False,
        grad_norm_lite_interval=GRAD_NORM_LITE_INTERVAL,
        grad_norm_lite_ema_alpha=0.1,
        grad_norm_lite_eps=1e-3,
        grad_norm_lite_task_groups={},
        log_task_gradient_norms=False,
    )
    kwargs.update(overrides)
    return MultiTaskModel(**kwargs)


def _make_trainer(max_batches):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=max_batches,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def test_grad_norm_lite_ema_updates_only_on_interval_steps():
    model = _make_model(grad_norm_lite=True)
    trainer = _make_trainer(NUM_BATCHES)
    batches = _make_batches(NUM_BATCHES)

    trainer.fit(model, train_dataloaders=_make_dataloader(batches))

    assert model._grad_norm_lite_ema.ema.keys() == {"segment", "elevation"}
    for norm in model._grad_norm_lite_ema.ema.values():
        assert norm > 0.0


def test_grad_norm_lite_scale_is_consistent_with_ema():
    model = _make_model(grad_norm_lite=True)
    trainer = _make_trainer(NUM_BATCHES)
    batches = _make_batches(NUM_BATCHES)

    trainer.fit(model, train_dataloaders=_make_dataloader(batches))

    logged_scale = trainer.callback_metrics["train/grad_norm_lite_scale_segment"].item()
    expected_scale = model._grad_norm_lite_ema.scale("segment")
    assert logged_scale == pytest.approx(expected_scale, rel=1e-4)
    # Once observed, the EMA-derived scale should differ from the pre-observation
    # default of 1.0 (i.e. GradNorm-lite is actually rescaling, not a no-op).
    assert logged_scale != pytest.approx(1.0)


def test_grad_norm_lite_disabled_keeps_scale_effectively_unused():
    model = _make_model(grad_norm_lite=False)
    trainer = _make_trainer(NUM_BATCHES)
    batches = _make_batches(NUM_BATCHES)

    trainer.fit(model, train_dataloaders=_make_dataloader(batches))

    assert not hasattr(model, "_grad_norm_lite_ema")
    assert "train/grad_norm_lite_scale_segment" not in trainer.callback_metrics


def test_log_task_gradient_norms_logs_diagnostic_keys_without_affecting_loss():
    model = _make_model(grad_norm_lite=False, log_task_gradient_norms=True)
    trainer = _make_trainer(2)
    batches = _make_batches(2)

    trainer.fit(model, train_dataloaders=_make_dataloader(batches))

    metrics = trainer.callback_metrics
    assert "train/task_grad_norm_backbone_segment" in metrics
    assert "train/task_grad_norm_head_segment" in metrics
    assert "train/task_grad_cos_segment__elevation" in metrics
    assert -1.0 <= metrics["train/task_grad_cos_segment__elevation"].item() <= 1.0
