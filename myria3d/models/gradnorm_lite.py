"""GradNorm-lite: EMA-based per-task loss rescaling, ported from Pointcept.

See /data/geist/Pointcept/pointcept/utils/gradient_norm.py for the reference
implementation this was ported from. Trimmed of AMP-underflow probing (myria3d
does not use mixed precision) and multi-backbone dispatch (myria3d has a
single multitask backbone, `PyGRandLANetMultiTask`).
"""

import math
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor


def l2_grad_norm(grads: Iterable[Optional[Tensor]]) -> float:
    total_sq = None
    for g in grads:
        if g is None:
            continue
        sq = g.detach().float().pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
    if total_sq is None:
        return 0.0
    value = float(total_sq.sqrt().item())
    if not math.isfinite(value):
        return 0.0
    return value


def _flatten_grads(grads: Iterable[Optional[Tensor]]) -> Optional[Tensor]:
    parts = [g.detach().reshape(-1) for g in grads if g is not None]
    if not parts:
        return None
    return torch.cat(parts)


def cosine_similarity_flat(a: Tensor, b: Tensor) -> float:
    norm_a = a.norm()
    norm_b = b.norm()
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float((a @ b / (norm_a * norm_b)).item())


def _pairwise_backbone_cosine_similarities(
    backbone_grads_by_task: Dict[str, Optional[Tensor]],
) -> Dict[str, float]:
    task_names = list(backbone_grads_by_task.keys())
    cos_pairs = {}
    for i, task_a in enumerate(task_names):
        grad_a = backbone_grads_by_task[task_a]
        if grad_a is None:
            continue
        for task_b in task_names[i + 1 :]:
            grad_b = backbone_grads_by_task[task_b]
            if grad_b is None:
                continue
            cos_pairs[f"{task_a}__{task_b}"] = cosine_similarity_flat(grad_a, grad_b)
    return cos_pairs


def compute_task_last_layer_grad_norms(
    last_layer_params: List[torch.nn.Parameter],
    loss_by_task: Dict[str, Tensor],
    task_groups: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Per-task (or per-group) L2 norm of grads w.r.t. the last shared backbone layer.

    Uses raw (unweighted) task losses. Does not write into ``.grad``. Losses
    without a grad_fn (e.g. an all-invalid-target batch producing `new_zeros`)
    are skipped. Non-finite / zero norms are omitted. ``task_groups`` optionally
    maps task_name -> group_name; tasks sharing a group have their losses summed
    before probing, so the returned dict is keyed by group_name (defaults to
    task_name for ungrouped tasks) and holds a single norm per group.
    """
    if not last_layer_params:
        return {}
    grouped_losses: Dict[str, List[Tensor]] = {}
    for task_name, task_loss in loss_by_task.items():
        if not isinstance(task_loss, Tensor) or not task_loss.requires_grad:
            continue
        group = task_groups.get(task_name, task_name) if task_groups else task_name
        grouped_losses.setdefault(group, []).append(task_loss)

    norms = {}
    for group, losses in grouped_losses.items():
        group_loss = losses[0] if len(losses) == 1 else sum(losses)
        grads = torch.autograd.grad(
            group_loss, last_layer_params, retain_graph=True, allow_unused=True
        )
        norm = l2_grad_norm(grads)
        if math.isfinite(norm) and norm > 0.0:
            norms[group] = norm
    return norms


def resolve_grad_norm_lite_scales(
    ema: "GradNormLiteEMA",
    task_names: Iterable[str],
    task_groups: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Per-task loss scales from a ``GradNormLiteEMA``, honoring ``task_groups``.

    Without ``task_groups`` this is just ``ema.scales(task_names)``. With it,
    every task in the same group shares the group's EMA scale (the EMA itself
    is keyed by group_name, matching ``compute_task_last_layer_grad_norms``).
    """
    task_names = list(task_names)
    if not task_groups:
        return ema.scales(task_names)
    groups = sorted(set(task_groups.get(t, t) for t in task_names))
    group_scales = ema.scales(groups)
    return {t: group_scales[task_groups.get(t, t)] for t in task_names}


def combine_weighted_task_losses(
    loss_by_task: Dict[str, Tensor],
    task_weights: Dict[str, float],
    scales: Optional[Dict[str, float]] = None,
) -> "tuple[Tensor, Dict[str, float]]":
    """Combine per-task losses as total = Sum L_t * w_t * s_t (s_t defaults to 1.0)."""
    task_weights = task_weights or {}
    scales = scales or {}
    total_loss = None
    applied_scales = {}
    for task_name, task_loss in loss_by_task.items():
        w = float(task_weights.get(task_name, 1.0))
        scale = float(scales.get(task_name, 1.0))
        applied_scales[task_name] = scale
        weighted = task_loss * w * scale
        total_loss = weighted if total_loss is None else total_loss + weighted
    return total_loss, applied_scales


class GradNormLiteEMA:
    """Per-task EMA of last-layer gradient norms for loss reweighting."""

    def __init__(self, alpha: float = 0.1, eps: float = 1e-3):
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.ema: Dict[str, float] = {}

    def update(self, norms: Dict[str, float]) -> None:
        for task_name, norm in norms.items():
            value = float(norm)
            if not math.isfinite(value) or value <= 0.0:
                continue
            if task_name not in self.ema:
                self.ema[task_name] = value
            else:
                self.ema[task_name] = (1.0 - self.alpha) * self.ema[task_name] + self.alpha * value

    def scale(self, task_name: str) -> float:
        """Return 1 / max(ema, eps), or 1.0 before the first valid observation."""
        ema = self.ema.get(task_name)
        if ema is None or not math.isfinite(ema) or ema <= 0.0:
            return 1.0
        return 1.0 / max(ema, self.eps)

    def scales(self, task_names: Optional[Iterable[str]] = None) -> Dict[str, float]:
        names = list(task_names) if task_names is not None else list(self.ema.keys())
        return {name: self.scale(name) for name in names}


def compute_task_gradient_norms(
    backbone_params: List[torch.nn.Parameter],
    head_params_by_task: Dict[str, List[torch.nn.Parameter]],
    loss_by_task: Dict[str, Tensor],
    task_weights: Dict[str, float],
) -> Dict[str, Dict]:
    """Diagnostic-only: per-task L2 grad norms (backbone/head split) and
    pairwise backbone-gradient cosine similarities between tasks. Never fed
    back into the loss; purely for logging task interference."""
    norms = {}
    backbone_grads_by_task: Dict[str, Optional[Tensor]] = {}
    for task_name, task_loss in loss_by_task.items():
        head_params = head_params_by_task.get(task_name, [])
        all_params = backbone_params + head_params
        if not isinstance(task_loss, Tensor) or not task_loss.requires_grad or not all_params:
            norms[task_name] = {"backbone": 0.0, "head": 0.0}
            backbone_grads_by_task[task_name] = None
            continue
        w = float(task_weights.get(task_name, 1.0))
        scaled_loss = task_loss * w
        grads = torch.autograd.grad(scaled_loss, all_params, retain_graph=True, allow_unused=True)
        n_bb = len(backbone_params)
        backbone_grads = grads[:n_bb]
        norms[task_name] = {
            "backbone": l2_grad_norm(backbone_grads),
            "head": l2_grad_norm(grads[n_bb:]),
        }
        backbone_grads_by_task[task_name] = _flatten_grads(backbone_grads)
    return {
        "norms": norms,
        "backbone_cos": _pairwise_backbone_cosine_similarities(backbone_grads_by_task),
    }
