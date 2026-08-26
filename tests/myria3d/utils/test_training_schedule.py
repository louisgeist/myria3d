import hydra
import pytest
from omegaconf import OmegaConf

from myria3d.utils.training_schedule import resolve_training_schedule
from tests.conftest import make_default_hydra_cfg


def test_resolve_training_schedule_classic_mode_unchanged():
    config = OmegaConf.create(
        {
            "total_iters": None,
            "trainer": {"max_epochs": 150, "min_epochs": 100},
        }
    )
    resolve_training_schedule(config)
    assert config.trainer.max_epochs == 150
    assert config.trainer.min_epochs == 100
    assert "limit_train_batches" not in config.trainer


def test_resolve_training_schedule_derives_pl_params():
    config = OmegaConf.create(
        {
            "total_iters": 10000,
            "trainer": {},
            "datamodule": {},
            "model": {},
        }
    )
    resolve_training_schedule(config)
    assert config.iter_per_epoch == 1000
    assert config.eval_every == 5
    assert config.num_epochs == 10
    assert config.trainer.max_epochs == 10
    assert config.trainer.min_epochs == 10
    assert config.trainer.limit_train_batches == 1000
    assert config.trainer.check_val_every_n_epoch == 5
    assert config.datamodule.iter_per_epoch == 1000
    assert config.model.lr_scheduler_frequency == 5


def test_resolve_training_schedule_custom_params():
    config = OmegaConf.create(
        {
            "total_iters": 15,
            "iter_per_epoch": 5,
            "eval_every": 3,
            "trainer": {},
            "datamodule": {},
            "model": {},
        }
    )
    resolve_training_schedule(config)
    assert config.num_epochs == 3
    assert config.trainer.max_epochs == 3
    assert config.trainer.limit_train_batches == 5
    assert config.trainer.check_val_every_n_epoch == 3
    assert config.model.lr_scheduler_frequency == 3


def test_resolve_training_schedule_rejects_non_divisible():
    config = OmegaConf.create(
        {
            "total_iters": 10001,
            "iter_per_epoch": 1000,
            "trainer": {},
        }
    )
    with pytest.raises(ValueError, match="divisible"):
        resolve_training_schedule(config)


def test_resolve_training_schedule_injects_frequency_on_hydra_struct_config():
    """lr_scheduler_frequency is not a YAML key: it must be added on a struct Hydra config."""
    config = make_default_hydra_cfg(
        overrides=[
            "experiment=flair3d_plus/multitask_v12_pointcept_jz",
            "logger=csv",
            "iter_per_epoch=2",
        ]
    )
    assert "lr_scheduler_frequency" not in config.model
    resolve_training_schedule(config)
    assert config.model.lr_scheduler_frequency == config.eval_every == 5
    assert config.trainer.check_val_every_n_epoch == 5

    model = hydra.utils.instantiate(config.model)
    opt_cfg = model.configure_optimizers()
    assert opt_cfg["lr_scheduler"]["frequency"] == 5
    assert opt_cfg["lr_scheduler"]["monitor"] == "val/iou"
    assert opt_cfg["lr_scheduler"]["interval"] == "epoch"
