import torch
from torch_geometric.data import Batch, Data

from myria3d.models.modules.pyg_randla_net_multitask import PyGRandLANetMultiTask


def _make_model_and_batch():
    num_features = 5
    task_configs = {
        "segment": {"task_type": "semantic", "num_classes": 6},
        "elevation": {"task_type": "regression"},
    }
    model = PyGRandLANetMultiTask(num_features, task_configs, decimation=4, num_neighbors=4)

    data = Batch.from_data_list(
        [
            Data(
                x=torch.rand((n, num_features)),
                pos=torch.rand((n, 3)),
                batch=torch.full((n,), idx),
            )
            for idx, n in enumerate([64, 64])
        ]
    )
    return model, data


def test_last_backbone_layer_parameters_are_fp1_and_receive_gradients():
    model, data = _make_model_and_batch()
    outputs = model(data.x, data.pos, data.batch, data.ptr)
    loss = sum(o.sum() for o in outputs.values())
    loss.backward()

    last_layer_params = model.last_backbone_layer_parameters()
    fp1_params = list(model.fp1.parameters())
    assert last_layer_params == fp1_params
    assert len(last_layer_params) > 0
    for p in last_layer_params:
        assert p.grad is not None


def test_backbone_parameters_excludes_task_heads():
    model, _ = _make_model_and_batch()
    backbone_param_ids = {id(p) for p in model.backbone_parameters()}
    for task_name in model.task_configs:
        for p in model.task_head_parameters(task_name):
            assert id(p) not in backbone_param_ids


def test_task_head_parameters_are_disjoint_across_tasks():
    model, _ = _make_model_and_batch()
    segment_ids = {id(p) for p in model.task_head_parameters("segment")}
    elevation_ids = {id(p) for p in model.task_head_parameters("elevation")}
    assert segment_ids.isdisjoint(elevation_ids)
    assert len(segment_ids) > 0
    assert len(elevation_ids) > 0
