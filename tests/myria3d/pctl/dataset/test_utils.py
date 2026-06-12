import numpy as np
import pytest
import torch

from myria3d.pctl.dataset.utils import (
    get_mosaic_of_centers,
    get_num_subtiles,
    get_subtile_choice,
    get_subtile_mask,
)


@pytest.mark.parametrize(
    "tile_width, subtile_width, subtile_overlap",
    zip([1000], [50], [25]),
)
def test_get_mosaic_of_centers(tile_width, subtile_width, subtile_overlap):
    mosaic = get_mosaic_of_centers(tile_width, subtile_width, subtile_overlap=subtile_overlap)
    for s in np.stack(mosaic).transpose():
        assert min(s - subtile_width / 2) <= 0
        assert max(s + subtile_width / 2) <= 1000


def test_get_num_subtiles_matches_mosaic_size():
    assert get_num_subtiles(100, 50, subtile_overlap=0) == 4
    assert get_num_subtiles(100, 50) == len(get_mosaic_of_centers(100, 50))


def test_get_subtile_choice_matches_mask():
    xs = np.linspace(0.0, 100.0, 11)
    ys = np.linspace(0.0, 100.0, 11)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pos = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)], axis=1).astype(
        np.float32
    )
    for subtile_index in range(4):
        mask = get_subtile_mask(pos, 100, 50, subtile_index)
        choice = get_subtile_choice(torch.from_numpy(pos), 100, 50, subtile_index)
        assert np.array_equal(mask, choice.numpy())


def test_get_subtile_mask_four_quadrants_on_synthetic_tile():
    xs = np.linspace(0.0, 100.0, 101)
    ys = np.linspace(0.0, 100.0, 101)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pos = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)], axis=1)

    masks = [
        get_subtile_mask(pos, tile_width=100, subtile_width=50, subtile_index=i)
        for i in range(4)
    ]
    for mask in masks:
        assert mask.any()

    union = np.logical_or.reduce(masks)
    assert union.sum() == pos.shape[0]

    with pytest.raises(ValueError):
        get_subtile_mask(pos, tile_width=100, subtile_width=50, subtile_index=4)
