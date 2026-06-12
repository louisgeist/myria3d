"""Resolve iter-limited training schedule (Pointcept-compatible) into PyTorch Lightning config."""

from omegaconf import DictConfig, OmegaConf

from myria3d.utils import utils

log = utils.get_logger(__name__)

DEFAULT_ITER_PER_EPOCH = 1000
DEFAULT_EVAL_EVERY = 5


def resolve_training_schedule(config: DictConfig) -> None:
    """Derive PyTorch Lightning trainer/datamodule params from iter-limited schedule.

    When ``total_iters`` is set, one training epoch means ``iter_per_epoch`` optimizer
    steps (batch steps), not a full dataset pass. Classic ``trainer.max_epochs`` mode
    is unchanged when ``total_iters`` is null or absent.

    Modifies ``config`` in place.
    """
    total_iters = config.get("total_iters")
    if total_iters is None:
        return

    OmegaConf.set_struct(config, False)

    iter_per_epoch = config.get("iter_per_epoch")
    if iter_per_epoch is None:
        iter_per_epoch = DEFAULT_ITER_PER_EPOCH
        config.iter_per_epoch = iter_per_epoch

    eval_every = config.get("eval_every")
    if eval_every is None:
        eval_every = DEFAULT_EVAL_EVERY
        config.eval_every = eval_every

    if total_iters <= 0:
        raise ValueError("total_iters must be > 0")
    if iter_per_epoch <= 0:
        raise ValueError("iter_per_epoch must be > 0 when total_iters is set")
    if eval_every <= 0:
        raise ValueError("eval_every must be > 0 when total_iters is set")
    if total_iters % iter_per_epoch != 0:
        raise ValueError(
            f"total_iters ({total_iters}) must be divisible by iter_per_epoch "
            f"({iter_per_epoch})"
        )

    num_epochs = total_iters // iter_per_epoch
    config.num_epochs = num_epochs

    if "trainer" not in config:
        config.trainer = OmegaConf.create({})
    config.trainer.max_epochs = num_epochs
    config.trainer.min_epochs = num_epochs
    config.trainer.limit_train_batches = iter_per_epoch
    config.trainer.check_val_every_n_epoch = eval_every

    if "datamodule" in config:
        config.datamodule.iter_per_epoch = iter_per_epoch
        if config.get("seed") is not None and config.datamodule.get("seed") is None:
            config.datamodule.seed = config.seed

    log.info(
        "Iter-limited training enabled: total_iters=%d optimizer steps, "
        "iter_per_epoch=%d, num_epochs=%d, eval_every=%d. "
        "Validation runs every eval_every epochs. "
        "Classic trainer.max_epochs is overridden.",
        total_iters,
        iter_per_epoch,
        num_epochs,
        eval_every,
    )

    OmegaConf.set_struct(config, True)
