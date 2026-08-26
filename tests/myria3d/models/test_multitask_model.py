import functools

import pytest
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
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


def test_configure_optimizers_uses_lr_scheduler_frequency():
    model = _make_model(
        lr_scheduler=functools.partial(torch.optim.lr_scheduler.ReduceLROnPlateau),
        lr_scheduler_frequency=5,
    )
    opt_cfg = model.configure_optimizers()
    assert opt_cfg["lr_scheduler"]["frequency"] == 5
    assert opt_cfg["lr_scheduler"]["monitor"] == "val/iou"
    assert opt_cfg["lr_scheduler"]["interval"] == "epoch"


def _make_plateau_trainer(max_epochs, check_val_every_n_epoch=5):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )


def test_reduce_lr_on_plateau_does_not_step_before_first_validation():
    """Reproduce the Jean Zay crash: ReduceLROnPlateau must not look up val/iou
    on epochs where validation did not run (eval_every > 1)."""
    model = _make_model(
        lr_scheduler=functools.partial(torch.optim.lr_scheduler.ReduceLROnPlateau),
        lr_scheduler_frequency=5,
        monitor="val/iou",
    )
    trainer = _make_plateau_trainer(max_epochs=2, check_val_every_n_epoch=5)
    trainer.fit(model, train_dataloaders=_make_dataloader(_make_batches(2)))


def test_reduce_lr_on_plateau_raises_when_frequency_not_aligned():
    """Lock in the original bug: stepping every epoch without val/iou crashes."""
    model = _make_model(
        lr_scheduler=functools.partial(torch.optim.lr_scheduler.ReduceLROnPlateau),
        lr_scheduler_frequency=1,
        monitor="val/iou",
    )
    trainer = _make_plateau_trainer(max_epochs=1, check_val_every_n_epoch=5)
    with pytest.raises(MisconfigurationException, match="val/iou"):
        trainer.fit(model, train_dataloaders=_make_dataloader(_make_batches(1)))


def test_pixel_semantic_and_tile_distribution_training_step_smoke():
    from myria3d.models.losses import WeightedKLDivLoss

    task_configs = {
        "segment": {"task_type": "semantic", "num_classes": 4, "ignore_index": 3},
        "forest_2d": {
            "task_type": "pixel_semantic",
            "num_classes": 2,
            "ignore_index": 2,
            "pooling": "mean",
        },
        "roads": {
            "task_type": "pixel_semantic",
            "num_classes": 2,
            "ignore_index": 2,
            "pooling": "max",
        },
        "nathab_habitat_type": {
            "task_type": "tile_distribution",
            "num_classes": 4,
            "ignore_index": 4,
        },
        "elevation": {"task_type": "regression"},
    }
    model = _make_model(
        neural_net_hparams=dict(
            num_features=NUM_FEATURES,
            task_configs=task_configs,
            num_neighbors=4,
            decimation=4,
            return_logits=True,
        ),
        task_configs=task_configs,
        task_weights={name: 1.0 for name in task_configs},
        criteria={
            "segment": torch.nn.CrossEntropyLoss(ignore_index=3),
            "forest_2d": torch.nn.CrossEntropyLoss(ignore_index=2),
            "roads": torch.nn.CrossEntropyLoss(ignore_index=2),
            "nathab_habitat_type": WeightedKLDivLoss(),
            "elevation": torch.nn.SmoothL1Loss(beta=0.01),
        },
    )
    model.train()
    n = 64
    data = Data(
        x=torch.rand(n, NUM_FEATURES),
        pos=torch.rand(n, 3),
        y=torch.randint(0, 4, (n,)),
        y_forest_2d=torch.randint(0, 3, (n,)),
        forest_2d_cell_id=torch.arange(n) % 8,
        y_roads=torch.randint(0, 3, (n,)),
        roads_cell_id=torch.arange(n) % 8,
        y_nathab_habitat_type=torch.randint(0, 5, (n,)),
        y_elevation=torch.rand(n) * 10.0,
    )
    batch = Batch.from_data_list([data, data.clone()])
    out = model.training_step(batch, 0)
    assert torch.isfinite(out["loss"])
    assert set(out["outputs"]) == set(task_configs)

