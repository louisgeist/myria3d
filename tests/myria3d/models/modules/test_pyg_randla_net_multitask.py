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


def test_pool_before_head_produces_cell_uniform_outputs_and_leaves_other_tasks_unpooled():
    """`pool_before_head` pools the backbone feature per (scene, cell) group before the
    head runs, then broadcasts it back — so every point sharing a cell must get an
    identical output row for that task, matching Pointcept's pool-then-classify
    pixel_semantic recipe. Tasks without the flag stay per-point (no such guarantee)."""
    num_features = 5
    task_configs = {
        "segment": {"task_type": "semantic", "num_classes": 6},
        "forest_2d": {
            "task_type": "pixel_semantic",
            "num_classes": 2,
            "pooling": "mean",
            "pool_before_head": True,
        },
    }
    model = PyGRandLANetMultiTask(num_features, task_configs, decimation=4, num_neighbors=4)
    model.eval()

    n_per_graph = 64
    data = Batch.from_data_list(
        [
            Data(
                x=torch.rand((n_per_graph, num_features)),
                pos=torch.rand((n_per_graph, 3)),
                batch=torch.full((n_per_graph,), idx),
            )
            for idx in range(2)
        ]
    )
    # Two cells per graph (first/second half of points), out-of-grid points (-1) excluded.
    cell_id = torch.cat(
        [
            torch.zeros(n_per_graph // 2, dtype=torch.long),
            torch.ones(n_per_graph // 2, dtype=torch.long),
        ]
    ).repeat(2)

    with torch.no_grad():
        outputs = model(
            data.x, data.pos, data.batch, data.ptr, pooled_head_cell_ids={"forest_2d": cell_id}
        )

    for graph_idx in range(2):
        graph_mask = data.batch == graph_idx
        for cid in (0, 1):
            group_mask = graph_mask & (cell_id == cid)
            group_logits = outputs["forest_2d"][group_mask]
            assert torch.allclose(group_logits, group_logits[0].expand_as(group_logits))

    # segment has no pool_before_head: no such uniformity guarantee (points differ).
    assert not torch.allclose(
        outputs["segment"][data.batch == 0],
        outputs["segment"][data.batch == 0][0].expand(n_per_graph, -1),
    )
