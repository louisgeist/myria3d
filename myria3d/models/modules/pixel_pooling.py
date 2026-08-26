"""Pool per-point predictions/labels up to raster-cell level for pixel_semantic tasks."""

from typing import Tuple

import torch
from torch import Tensor
from torch_scatter import scatter, scatter_min

# Must exceed the largest realistic `row * width + col` raster cell id, so combining it
# with a per-point scene/batch index never lets two scenes' cell ids collide.
CELL_ID_SCENE_STRIDE = 1_000_000_000


def pool_points_by_cell(
    preds: Tensor,
    cell_id: Tensor,
    target: Tensor,
    batch_index: Tensor,
    pooling: str = "mean",
) -> Tuple[Tensor, Tensor]:
    """Pool per-point logits/labels to (scene, raster-cell) groups.

    Points with `cell_id < 0` (outside the raster grid) are dropped. Pooling never mixes
    points from different scenes in a batch, even if their raw cell ids collide.
    """
    valid = cell_id >= 0
    if not bool(valid.any()):
        return preds.new_zeros((0, preds.size(-1))), target.new_zeros((0,))

    valid_cell_id = cell_id[valid].to(torch.int64)
    if int(valid_cell_id.max()) >= CELL_ID_SCENE_STRIDE:
        raise ValueError(
            "cell_id exceeds CELL_ID_SCENE_STRIDE; raster grid too large to pool safely."
        )

    group_key = batch_index[valid].to(torch.int64) * CELL_ID_SCENE_STRIDE + valid_cell_id
    _, inverse = torch.unique(group_key, return_inverse=True)
    num_groups = int(inverse.max().item()) + 1

    reduce = "mean" if pooling == "mean" else "max"
    pooled_preds = scatter(preds[valid], inverse, dim=0, dim_size=num_groups, reduce=reduce)

    # All points sharing a cell were sampled from the same raster pixel and therefore
    # carry an identical label; pick one representative point per group deterministically.
    point_index = torch.arange(preds.size(0), device=preds.device)[valid]
    _, representative = scatter_min(point_index, inverse, dim=0, dim_size=num_groups)
    pooled_target = target[valid][representative]

    return pooled_preds, pooled_target
