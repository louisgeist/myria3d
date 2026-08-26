import pytest
import torch
from torch import nn

from myria3d.models.gradnorm_lite import (
    GradNormLiteEMA,
    combine_weighted_task_losses,
    compute_task_gradient_norms,
    compute_task_last_layer_grad_norms,
    cosine_similarity_flat,
    resolve_grad_norm_lite_scales,
)


class TestGradNormLiteEMA:
    def test_scale_defaults_to_one_before_first_observation(self):
        ema = GradNormLiteEMA()
        assert ema.scale("segment") == 1.0

    def test_update_sets_first_observation_directly(self):
        ema = GradNormLiteEMA(alpha=0.1)
        ema.update({"segment": 4.0})
        assert ema.ema["segment"] == pytest.approx(4.0)

    def test_update_applies_ema_formula_on_second_observation(self):
        ema = GradNormLiteEMA(alpha=0.1)
        ema.update({"segment": 4.0})
        ema.update({"segment": 8.0})
        expected = 0.9 * 4.0 + 0.1 * 8.0
        assert ema.ema["segment"] == pytest.approx(expected)

    def test_update_skips_non_finite_and_non_positive_values(self):
        ema = GradNormLiteEMA()
        ema.update({"segment": float("nan"), "forest": 0.0, "elevation": -1.0})
        assert ema.ema == {}

    def test_scale_is_inverse_of_ema(self):
        ema = GradNormLiteEMA(eps=1e-3)
        ema.update({"segment": 2.0})
        assert ema.scale("segment") == pytest.approx(0.5)

    def test_scale_clamps_to_eps(self):
        ema = GradNormLiteEMA(eps=1e-3)
        ema.update({"segment": 1e-6})
        assert ema.scale("segment") == pytest.approx(1.0 / 1e-3)

    def test_scales_returns_all_requested_task_names(self):
        ema = GradNormLiteEMA()
        ema.update({"segment": 2.0})
        scales = ema.scales(["segment", "forest"])
        assert scales["segment"] == pytest.approx(0.5)
        assert scales["forest"] == 1.0


def _linear_task_losses(num_tasks=3, in_features=4):
    """A shared Linear layer with `num_tasks` losses depending on it at different scales."""
    layer = nn.Linear(in_features, in_features)
    x = torch.rand(5, in_features)
    losses = {}
    for i in range(num_tasks):
        scale = float(i + 1)
        losses[f"task_{i}"] = (layer(x) * scale).pow(2).sum()
    return layer, losses


class TestComputeTaskLastLayerGradNorms:
    def test_matches_manual_autograd_grad(self):
        layer, losses = _linear_task_losses(num_tasks=2)
        params = list(layer.parameters())
        norms = compute_task_last_layer_grad_norms(params, losses)

        for task_name, task_loss in losses.items():
            grads = torch.autograd.grad(task_loss, params, retain_graph=True)
            expected = float(torch.cat([g.reshape(-1) for g in grads]).norm().item())
            assert norms[task_name] == pytest.approx(expected, rel=1e-4)

    def test_task_groups_pool_losses_before_probing(self):
        layer, losses = _linear_task_losses(num_tasks=2)
        params = list(layer.parameters())
        grouped_norms = compute_task_last_layer_grad_norms(
            params, losses, task_groups={"task_0": "group_a", "task_1": "group_a"}
        )
        assert set(grouped_norms.keys()) == {"group_a"}

        summed_loss = sum(losses.values())
        grads = torch.autograd.grad(summed_loss, params, retain_graph=True)
        expected = float(torch.cat([g.reshape(-1) for g in grads]).norm().item())
        assert grouped_norms["group_a"] == pytest.approx(expected, rel=1e-4)

    def test_skips_loss_without_grad_fn(self):
        layer, losses = _linear_task_losses(num_tasks=1)
        losses["constant_task"] = torch.zeros(())
        params = list(layer.parameters())
        norms = compute_task_last_layer_grad_norms(params, losses)
        assert "constant_task" not in norms

    def test_empty_params_returns_empty_dict(self):
        assert compute_task_last_layer_grad_norms([], {"segment": torch.zeros(())}) == {}


class TestResolveGradNormLiteScales:
    def test_without_task_groups_delegates_to_ema(self):
        ema = GradNormLiteEMA()
        ema.update({"segment": 2.0})
        scales = resolve_grad_norm_lite_scales(ema, ["segment", "forest"])
        assert scales == {"segment": pytest.approx(0.5), "forest": 1.0}

    def test_with_task_groups_shares_scale_within_group(self):
        ema = GradNormLiteEMA()
        ema.update({"nathab": 4.0})
        scales = resolve_grad_norm_lite_scales(
            ema,
            ["forest", "axis_b"],
            task_groups={"forest": "nathab", "axis_b": "nathab"},
        )
        assert scales["forest"] == pytest.approx(0.25)
        assert scales["axis_b"] == pytest.approx(0.25)


class TestCombineWeightedTaskLosses:
    def test_combination_formula(self):
        losses = {"segment": torch.tensor(2.0), "forest": torch.tensor(4.0)}
        weights = {"segment": 1.0, "forest": 0.5}
        scales = {"segment": 1.0, "forest": 2.0}
        total, applied = combine_weighted_task_losses(losses, weights, scales)
        assert total.item() == pytest.approx(2.0 * 1.0 * 1.0 + 4.0 * 0.5 * 2.0)
        assert applied == {"segment": 1.0, "forest": 2.0}

    def test_missing_weight_and_scale_default_to_one(self):
        losses = {"segment": torch.tensor(3.0)}
        total, applied = combine_weighted_task_losses(losses, {}, None)
        assert total.item() == pytest.approx(3.0)
        assert applied == {"segment": 1.0}


class TestCosineSimilarityFlat:
    def test_identical_vectors_have_cosine_one(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_similarity_flat(a, a.clone()) == pytest.approx(1.0)

    def test_orthogonal_vectors_have_cosine_zero(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert cosine_similarity_flat(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        a = torch.zeros(3)
        b = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_similarity_flat(a, b) == 0.0


class TestComputeTaskGradientNorms:
    def test_backbone_head_split_and_cosine(self):
        backbone = nn.Linear(4, 4)
        head_a = nn.Linear(4, 1)
        head_b = nn.Linear(4, 1)
        x = torch.rand(3, 4)
        shared = backbone(x)
        losses = {
            "task_a": head_a(shared).sum(),
            "task_b": head_b(shared).sum(),
        }
        result = compute_task_gradient_norms(
            list(backbone.parameters()),
            {"task_a": list(head_a.parameters()), "task_b": list(head_b.parameters())},
            losses,
            task_weights={},
        )
        assert result["norms"]["task_a"]["backbone"] > 0.0
        assert result["norms"]["task_a"]["head"] > 0.0
        assert "task_a__task_b" in result["backbone_cos"]
        assert -1.0 <= result["backbone_cos"]["task_a__task_b"] <= 1.0
