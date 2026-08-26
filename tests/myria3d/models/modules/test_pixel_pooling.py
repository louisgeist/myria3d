import torch

from myria3d.models.modules.pixel_pooling import (
    pool_points_by_cell,
    scatter_cell_values_to_raster,
)


def test_scatter_cell_values_to_raster_places_values_and_leaves_nan():
    values = torch.tensor([0.2, 0.8], dtype=torch.float32)
    cell_id = torch.tensor([0, 3], dtype=torch.int64)
    raster = scatter_cell_values_to_raster(values, cell_id, height=2, width=2)

    assert raster.shape == (2, 2)
    assert torch.isclose(raster[0, 0], torch.tensor(0.2))
    assert torch.isnan(raster[0, 1])
    assert torch.isnan(raster[1, 0])
    assert torch.isclose(raster[1, 1], torch.tensor(0.8))


def test_scatter_cell_values_to_raster_empty_and_oob():
    empty = scatter_cell_values_to_raster(
        torch.zeros((0,), dtype=torch.float32),
        torch.zeros((0,), dtype=torch.int64),
        height=2,
        width=2,
    )
    assert empty.shape == (2, 2)
    assert torch.isnan(empty).all()

    oob = scatter_cell_values_to_raster(
        torch.tensor([1.0]),
        torch.tensor([99]),
        height=2,
        width=2,
    )
    assert torch.isnan(oob).all()


def test_pool_then_scatter_matches_unique_cells():
    preds = torch.tensor(
        [[0.0, 4.0], [0.0, 4.0], [4.0, 0.0]],
        dtype=torch.float32,
    )
    cell_id = torch.tensor([1, 1, 2], dtype=torch.int64)
    target = torch.tensor([1, 1, 0], dtype=torch.int64)
    batch_index = torch.zeros(3, dtype=torch.int64)
    pooled_preds, _, pooled_cell_id, _ = pool_points_by_cell(
        preds, cell_id, target, batch_index, pooling="max"
    )
    fg = torch.softmax(pooled_preds, dim=-1)[:, 1]
    raster = scatter_cell_values_to_raster(fg, pooled_cell_id, height=2, width=2)
    assert torch.isnan(raster[0, 0])
    assert raster[0, 1] > 0.9
    assert raster[1, 0] < 0.1
    assert torch.isnan(raster[1, 1])
