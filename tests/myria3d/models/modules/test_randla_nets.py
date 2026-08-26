import pytest
import torch
from torch_geometric.data import Batch, Data

from myria3d.models.modules.pyg_randla_net import PyGRandLANet
from myria3d.models.modules.pyg_randla_net_multitask import PyGRandLANetMultiTask


@pytest.mark.parametrize("num_nodes", [[12500, 12500], [50, 50], [12500, 10000]])
def test_fake_run_pyg_randlanet(num_nodes):
    """Documents expected data format and make a forward pass with PyG RandLa-Net.

    Accepts small clouds even though decimation should lead to empty cloud.
    Accepts point clouds of various sizes.

    """
    num_euclidian_dimensions = 3
    num_features = 9
    num_classes = 6
    decimation = 4
    num_neighbors = 16

    data = Batch.from_data_list(
        [
            Data(
                x=torch.rand((n, num_features)),
                pos=torch.rand((n, num_euclidian_dimensions)),
                batch=torch.full((n,), idx),
            )
            for idx, n in enumerate(num_nodes)
        ]
    )

    model = PyGRandLANet(
        num_features,
        num_classes,
        decimation=decimation,
        num_neighbors=num_neighbors,
    )
    output = model(data.x, data.pos, data.batch, data.ptr)
    assert output.shape == torch.Size([sum(num_nodes), num_classes])


def test_pyg_randlanet_multitask_head_shapes():
    num_nodes = [128, 128]
    num_features = 5
    data = Batch.from_data_list(
        [
            Data(
                x=torch.rand((n, num_features)),
                pos=torch.rand((n, 3)),
                batch=torch.full((n,), idx),
            )
            for idx, n in enumerate(num_nodes)
        ]
    )
    task_configs = {
        "segment": {"task_type": "semantic", "num_classes": 16, "ignore_index": 15},
        "forest_2d": {"task_type": "pixel_semantic", "num_classes": 2, "ignore_index": 2},
        "nathab_habitat_type": {
            "task_type": "tile_distribution",
            "num_classes": 4,
            "ignore_index": 4,
        },
        "elevation": {"task_type": "regression"},
    }
    model = PyGRandLANetMultiTask(
        num_features,
        task_configs,
        decimation=4,
        num_neighbors=16,
    )
    outputs = model(data.x, data.pos, data.batch, data.ptr)
    total = sum(num_nodes)
    assert outputs["segment"].shape == torch.Size([total, 16])
    assert outputs["forest_2d"].shape == torch.Size([total, 2])
    assert outputs["nathab_habitat_type"].shape == torch.Size([total, 4])
    assert outputs["elevation"].shape == torch.Size([total])
    # d_bottleneck is sized from max(32, max num_classes, num_features).
    assert model.fc0.out_features == 32
