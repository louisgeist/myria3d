from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from myria3d.callbacks.pointcept_pred_dump import (
    PointceptPredictionDump,
    merge_dense_probability_rasters,
    patch_ids_from_batch,
)


def test_patch_ids_from_batch_handles_collated_strings():
    assert patch_ids_from_batch(SimpleNamespace(patch_id="tile-a"), 1) == ["tile-a"]
    assert patch_ids_from_batch(
        SimpleNamespace(patch_id=["tile-a", "tile-b"]), 2
    ) == ["tile-a", "tile-b"]
    assert patch_ids_from_batch(
        SimpleNamespace(patch_id=[["tile-a"], ["tile-b"]]), 2
    ) == ["tile-a", "tile-b"]


def test_merge_dense_probability_rasters_nanmean():
    a = np.array([[[1.0, np.nan], [np.nan, 0.0]]], dtype=np.float32)
    b = np.array([[[0.0, 0.5], [np.nan, np.nan]]], dtype=np.float32)
    merged = merge_dense_probability_rasters([a, b])
    assert merged.shape == (1, 2, 2)
    np.testing.assert_allclose(merged[0, 0, 0], 0.5)
    np.testing.assert_allclose(merged[0, 0, 1], 0.5)
    assert np.isnan(merged[0, 1, 0])
    np.testing.assert_allclose(merged[0, 1, 1], 0.0)


def _roads_graph(patch_id: str, cell_ids, logits):
    n = len(cell_ids)
    data = Data(
        pos=torch.zeros((n, 3), dtype=torch.float32),
        x=torch.zeros((n, 5), dtype=torch.float32),
        y_roads=torch.zeros(n, dtype=torch.long),
        roads_cell_id=torch.tensor(cell_ids, dtype=torch.int64),
        roads_raster_h=torch.tensor([2], dtype=torch.long),
        roads_raster_w=torch.tensor([2], dtype=torch.long),
    )
    data.patch_id = patch_id
    return data, torch.tensor(logits, dtype=torch.float32)


def test_pointcept_prediction_dump_writes_network_logits(tmp_path):
    graph_a, logits_a = _roads_graph(
        "D067-2021_UU-S1-31_1-1",
        [0, 1],
        [[0.0, 8.0], [8.0, 0.0]],
    )
    graph_b, logits_b = _roads_graph(
        "D067-2021_UU-S1-31_1-1",
        [2, 3],
        [[0.0, 8.0], [8.0, 0.0]],
    )
    batch = Batch.from_data_list([graph_a, graph_b])
    outputs = {
        "outputs": {"roads": torch.cat([logits_a, logits_b], dim=0)},
        "targets": {"roads": batch.y_roads},
    }
    callback = PointceptPredictionDump(
        task_configs={
            "roads": {
                "task_type": "pixel_semantic",
                "num_classes": 2,
                "pooling": "max",
                "fg_index": 1,
            }
        },
        output_dir=str(tmp_path / "result"),
        phases=("test",),
    )
    trainer = SimpleNamespace(
        sanity_checking=False,
        global_rank=0,
        world_size=1,
        is_global_zero=True,
    )
    callback.on_test_batch_end(trainer, None, outputs, batch, 0)
    callback.on_test_epoch_end(trainer, None)

    path = tmp_path / "result" / "D067-2021_UU-S1-31_1-1_logits_network.npy"
    assert path.is_file()
    raster = np.load(path)
    assert raster.shape == (1, 2, 2)
    assert raster.dtype == np.float32
    assert raster[0, 0, 0] > 0.9
    assert raster[0, 0, 1] < 0.1
    assert raster[0, 1, 0] > 0.9
    assert raster[0, 1, 1] < 0.1


def test_pointcept_prediction_dump_merges_quadrants_across_batches(tmp_path):
    callback = PointceptPredictionDump(
        task_configs={
            "roads": {
                "task_type": "pixel_semantic",
                "num_classes": 2,
                "pooling": "max",
                "fg_index": 1,
            }
        },
        output_dir=str(tmp_path / "result"),
        phases=("test",),
    )
    trainer = SimpleNamespace(
        sanity_checking=False,
        global_rank=0,
        world_size=1,
        is_global_zero=True,
    )
    patch = "D067-2021_UU-S1-31_1-2"
    graph_a, logits_a = _roads_graph(patch, [0], [[0.0, 8.0]])
    graph_b, logits_b = _roads_graph(patch, [3], [[0.0, 8.0]])
    for graph, logits in ((graph_a, logits_a), (graph_b, logits_b)):
        batch = Batch.from_data_list([graph])
        outputs = {
            "outputs": {"roads": logits},
            "targets": {"roads": batch.y_roads},
        }
        callback.on_test_batch_end(trainer, None, outputs, batch, 0)
    callback.on_test_epoch_end(trainer, None)

    raster = np.load(tmp_path / "result" / f"{patch}_logits_network.npy")
    assert raster[0, 0, 0] > 0.9
    assert np.isnan(raster[0, 0, 1])
    assert np.isnan(raster[0, 1, 0])
    assert raster[0, 1, 1] > 0.9


def test_merge_rank_dump_dirs_nanmean(tmp_path):
    from myria3d.callbacks.pointcept_pred_dump import _merge_rank_dump_dirs

    output_dir = tmp_path / "result"
    name = "tile_logits_network.npy"
    a = np.array([[[1.0, np.nan]]], dtype=np.float32)
    b = np.array([[[0.0, 0.4]]], dtype=np.float32)
    for rank, arr in enumerate((a, b)):
        rank_dir = output_dir / f".rank{rank}"
        rank_dir.mkdir(parents=True)
        np.save(rank_dir / name, arr)

    n_written = _merge_rank_dump_dirs(output_dir, world_size=2)
    assert n_written == 1
    merged = np.load(output_dir / name)
    np.testing.assert_allclose(merged[0, 0, 0], 0.5)
    np.testing.assert_allclose(merged[0, 0, 1], 0.4)
