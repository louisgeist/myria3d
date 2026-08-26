import torch

from myria3d.models.modules.pixel_pooling import CELL_ID_SCENE_STRIDE, pool_points_by_cell


def test_pool_points_by_cell_mean_and_drops_negative_ids():
    preds = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [10.0, 0.0],  # cell_id = -1, dropped
        ]
    )
    cell_id = torch.tensor([0, 0, -1])
    target = torch.tensor([1, 1, 2])
    batch_index = torch.zeros(3, dtype=torch.long)

    pooled_preds, pooled_target = pool_points_by_cell(
        preds, cell_id, target, batch_index, pooling="mean"
    )

    assert pooled_preds.shape == (1, 2)
    torch.testing.assert_close(pooled_preds[0], torch.tensor([2.0, 0.0]))
    assert pooled_target.tolist() == [1]


def test_pool_points_by_cell_max_and_does_not_mix_scenes():
    preds = torch.tensor(
        [
            [1.0, 0.0],
            [4.0, 0.0],
            [9.0, 0.0],
            [2.0, 0.0],
        ]
    )
    # Same raw cell id 0 in two different scenes.
    cell_id = torch.tensor([0, 0, 0, 0])
    target = torch.tensor([1, 1, 0, 0])
    batch_index = torch.tensor([0, 0, 1, 1])

    pooled_preds, pooled_target = pool_points_by_cell(
        preds, cell_id, target, batch_index, pooling="max"
    )

    assert pooled_preds.shape == (2, 2)
    # Scene 0 max of 1 and 4; scene 1 max of 9 and 2.
    torch.testing.assert_close(pooled_preds[0], torch.tensor([4.0, 0.0]))
    torch.testing.assert_close(pooled_preds[1], torch.tensor([9.0, 0.0]))
    assert pooled_target.tolist() == [1, 0]
    assert CELL_ID_SCENE_STRIDE > 0
