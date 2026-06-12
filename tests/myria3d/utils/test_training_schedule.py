import pytest
from omegaconf import OmegaConf

from myria3d.utils.training_schedule import resolve_training_schedule


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


def test_resolve_training_schedule_custom_params():
    config = OmegaConf.create(
        {
            "total_iters": 15,
            "iter_per_epoch": 5,
            "eval_every": 3,
            "trainer": {},
            "datamodule": {},
        }
    )
    resolve_training_schedule(config)
    assert config.num_epochs == 3
    assert config.trainer.max_epochs == 3
    assert config.trainer.limit_train_batches == 5
    assert config.trainer.check_val_every_n_epoch == 3


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
