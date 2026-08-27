import numpy as np
import pytest
import torch
import torch_geometric

from myria3d.pctl.transforms.transforms import (
    DropPointsByClass,
    MinimumNumNodes,
    RandomDropColor,
    RandomDropStrength,
    SubtileCrop,
    TargetTransform,
    subsample_data,
)


@pytest.mark.parametrize(
    "x,idx,choice,nb_out_nodes",
    [
        # Standard use case with choice contiaining indices
        (
            torch.Tensor([10, 11, 12, 13, 14]),
            np.array([20, 21, 22, 23, 24]),
            torch.IntTensor([0, 1, 4]),
            3,
        ),
        # Edge case with choice containing indices: select no point
        (
            torch.Tensor([10, 11, 12, 13, 14]),
            np.array([20, 21, 22, 23, 24]),
            torch.IntTensor([]),
            0,
        ),
        # Edge case with choice containing indices: select one point
        (
            torch.Tensor([10, 11, 12, 13, 14]),
            np.array([20, 21, 22, 23, 24]),
            torch.IntTensor([1]),
            1,
        ),
        # Edge case with choice containing indices: input array with one point
        (
            torch.Tensor([10]),
            np.array([20]),
            torch.IntTensor([0]),
            1,
        ),
        # Edge case with choice containing indices: input array with one point
        (
            torch.Tensor([10]),
            np.array([20]),
            torch.IntTensor([]),
            0,
        ),
        # Standard use case with choice as boolean array
        (
            torch.Tensor([10, 11, 12, 13, 14]),
            np.array([20, 21, 22, 23, 24]),
            torch.BoolTensor([True, True, False, True, False]),
            3,
        ),
        # Edge case with choice as boolean array: select no point
        (
            torch.Tensor([10, 11, 12, 13, 14]),
            np.array([20, 21, 22, 23, 24]),
            torch.BoolTensor([False, False, False, False, False]),
            0,
        ),
        # Edge case with choice as boolean array: select one point
        (
            torch.Tensor([10, 11, 12, 13, 14]),
            np.array([20, 21, 22, 23, 24]),
            torch.BoolTensor([False, True, False, False, False]),
            1,
        ),
        # Edge case with choice as boolean array: input array with one point
        (
            torch.Tensor([10]),
            np.array([20]),
            torch.BoolTensor([True]),
            1,
        ),
        # Edge case with choice as boolean array: input array with one point
        (
            torch.Tensor([10]),
            np.array([20]),
            torch.BoolTensor([False]),
            0,
        ),
    ],
)
def test_subsample_data(x, idx, choice, nb_out_nodes):
    num_nodes = x.size(0)
    data = torch_geometric.data.Data(x=x, idx_in_original_cloud=idx, num_nodes=num_nodes)
    transformed_data = subsample_data(data, num_nodes, choice)
    assert transformed_data.num_nodes == nb_out_nodes
    assert isinstance(transformed_data.x, torch.Tensor)
    assert transformed_data.x.size(0) == nb_out_nodes
    assert isinstance(transformed_data.idx_in_original_cloud, np.ndarray)
    # Check that "idx_in_original_cloud" key is not modified
    assert transformed_data.idx_in_original_cloud.shape[0] == num_nodes


def _synthetic_tile_data(num_per_side: int = 51) -> torch_geometric.data.Data:
    xs = np.linspace(0.0, 100.0, num_per_side)
    ys = np.linspace(0.0, 100.0, num_per_side)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pos = torch.from_numpy(
        np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)], axis=1).astype(
            np.float32
        )
    )
    x = torch.rand((pos.size(0), 3))
    idx = np.arange(pos.size(0))
    return torch_geometric.data.Data(pos=pos, x=x, idx_in_original_cloud=idx)


def test_SubtileCrop_fixed_index():
    data = _synthetic_tile_data()
    data.subtile_index = 0
    original_nodes = data.num_nodes
    crop = SubtileCrop(tile_width=100, subtile_width=50, random=False)
    cropped = crop(data.clone())
    assert cropped is not None
    assert cropped.num_nodes < original_nodes
    assert cropped.num_nodes > 0
    assert cropped.idx_in_original_cloud.shape[0] == cropped.num_nodes


def test_SubtileCrop_random():
    data = _synthetic_tile_data()
    crop = SubtileCrop(tile_width=100, subtile_width=50, random=True)
    cropped = crop(data.clone())
    assert cropped is not None
    assert cropped.num_nodes < data.num_nodes


def test_SubtileCrop_returns_none_below_min_points():
    # A tile whose points all sit in quadrant 0 -> quadrant 3 crop is (near-)empty.
    pos = torch.tensor(
        [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]], dtype=torch.float32
    )
    data = torch_geometric.data.Data(pos=pos, x=torch.rand(3, 3), idx_in_original_cloud=np.arange(3))
    data.subtile_index = 3  # far quadrant, no points there
    assert SubtileCrop(tile_width=100, subtile_width=50, min_points=1)(data.clone()) is None

    data.subtile_index = 0  # 3 points land here
    assert SubtileCrop(tile_width=100, subtile_width=50, min_points=1)(data.clone()) is not None
    assert SubtileCrop(tile_width=100, subtile_width=50, min_points=10)(data.clone()) is None


def test_SubtileCrop_uses_data_subtile_index():
    data = _synthetic_tile_data()
    crop = SubtileCrop(tile_width=100, subtile_width=50, random=False)
    first_data = data.clone()
    first_data.subtile_index = 0
    second_data = data.clone()
    second_data.subtile_index = 1
    first = crop(first_data)
    second = crop(second_data)
    assert first.num_nodes > 0
    assert second.num_nodes > 0
    assert first.num_nodes != second.num_nodes or not torch.equal(first.pos, second.pos)


def test_TargetTransform_with_valid_config():
    # 2 are turned into 1.
    classification_preprocessing_dict = {2: 1}
    # 1 becomes 0, and 6 becomes 1.
    classification_dict = {1: "unclassified", 6: "building"}
    tt = TargetTransform(classification_preprocessing_dict, classification_dict)
    y = np.array([1, 1, 2, 2, 6, 6])
    idx = np.arange(6)
    data = torch_geometric.data.Data(x=None, y=y, idx_in_original_cloud=idx)
    out_data = tt(data)
    assert np.array_equal(out_data.y, np.array([0, 0, 0, 0, 1, 1]))
    assert np.array_equal(out_data.idx_in_original_cloud, idx)


def test_TargetTransform_throws_type_error_if_invalid_classification_dict():
    classification_preprocessing_dict = {2: 1}
    classification_dict = {1: "unclassified", 2: "ground", 6: "building"}
    tt = TargetTransform(classification_preprocessing_dict, classification_dict)

    invalid_input_data = torch_geometric.data.Data(x=None, y=np.array([1, 1, 1, 2, 99999, 1]))
    with pytest.raises(TypeError):
        # error content:
        # int() argument must be a string, a bytes-like object or a number, not 'NoneType'
        _ = tt(invalid_input_data)


def test_DropPointsByClass():
    # points with class 65 are droped.
    y = torch.Tensor([1, 65, 65, 2, 65])
    x = torch.rand((5, 3))
    idx = np.arange(5)  # Not a tensor
    data = torch_geometric.data.Data(x=x, y=y, idx_in_original_cloud=idx)
    drop_transforms = DropPointsByClass()
    transformed_data = drop_transforms(data)
    assert torch.equal(transformed_data.y, torch.Tensor([1, 2]))
    assert transformed_data.x.size(0) == 2
    assert isinstance(transformed_data.idx_in_original_cloud, np.ndarray)
    assert transformed_data.idx_in_original_cloud.size == 2
    assert np.all(transformed_data.idx_in_original_cloud == np.array([0, 3]))

    # No modification
    x = torch.rand((3, 3))
    y = torch.Tensor([1, 2, 3])
    data = torch_geometric.data.Data(x=x, y=y)
    transformed_data = drop_transforms(data)
    assert torch.equal(data.x, transformed_data.x)
    assert torch.equal(data.y, transformed_data.y)

    # Keep one point only
    y = torch.Tensor([1, 65, 65, 65, 65])
    x = torch.rand((5, 3))
    idx = np.arange(5)  # Not a tensor
    data = torch_geometric.data.Data(x=x, y=y, idx_in_original_cloud=idx)
    transformed_data = drop_transforms(data)
    assert torch.equal(transformed_data.y, torch.Tensor([1]))
    assert transformed_data.x.size(0) == 1
    assert isinstance(transformed_data.idx_in_original_cloud, np.ndarray)
    assert transformed_data.idx_in_original_cloud.shape[0] == 1
    assert np.all(transformed_data.idx_in_original_cloud == np.array([0]))


@pytest.mark.parametrize("input_nodes,min_nodes", [(5, 10), (1, 10), (15, 10)])
def test_MinimumNumNodes(input_nodes, min_nodes):
    x = torch.rand((input_nodes, 3))
    idx = np.arange(input_nodes)  # Not a tensor
    data = torch_geometric.data.Data(x=x, idx_in_original_cloud=idx)
    transform = MinimumNumNodes(min_nodes)

    transformed_data = transform(data)
    expected_nodes = max(input_nodes, min_nodes)
    assert transformed_data.num_nodes == expected_nodes
    assert isinstance(transformed_data.x, torch.Tensor)
    assert transformed_data.x.size(0) == expected_nodes
    # Check that "idx_in_original_cloud" key is not modified
    assert isinstance(transformed_data.idx_in_original_cloud, np.ndarray)
    assert transformed_data.idx_in_original_cloud.shape[0] == input_nodes


FEATURE_NAMES = ["Intensity", "Red", "Green", "Blue", "rgb_avg"]


def _radiometry_data(num_points: int = 20) -> torch_geometric.data.Data:
    return torch_geometric.data.Data(
        x=torch.arange(num_points * 5, dtype=torch.float32).reshape(num_points, 5) + 1.0,
        x_features_names=FEATURE_NAMES,
        num_nodes=num_points,
    )


def test_RandomDropColor_drops_all_rgb_channels_and_stores_mask():
    torch.manual_seed(0)
    data = _radiometry_data(10)
    original_intensity = data.x[:, 0].clone()
    dropped = RandomDropColor(
        drop_ratio=1.0, drop_application_ratio=1.0, keep_mask=True
    )(data)
    assert torch.equal(dropped.x[:, 0], original_intensity)
    assert torch.all(dropped.x[:, 1:] == 0)
    assert torch.all(dropped.color_mask)


def test_RandomDropStrength_drops_intensity_only():
    torch.manual_seed(0)
    data = _radiometry_data(10)
    original_color = data.x[:, 1:].clone()
    dropped = RandomDropStrength(
        drop_ratio=1.0, drop_application_ratio=1.0, keep_mask=True
    )(data)
    assert torch.all(dropped.x[:, 0] == 0)
    assert torch.equal(dropped.x[:, 1:], original_color)
    assert torch.all(dropped.strength_mask)


def test_RandomDropColor_stacked_calls_or_the_mask():
    torch.manual_seed(0)
    data = _radiometry_data(20)
    first = RandomDropColor(drop_ratio=0.5, drop_application_ratio=1.0, keep_mask=True)
    second = RandomDropColor(drop_ratio=0.5, drop_application_ratio=1.0, keep_mask=True)
    dropped = second(first(data))
    assert int(dropped.color_mask.sum()) >= 10
    assert torch.all(dropped.x[dropped.color_mask, 1:] == 0)
