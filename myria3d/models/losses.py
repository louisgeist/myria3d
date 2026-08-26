"""Losses and pooling for Flair3D+ tile_distribution tasks (ported from Pointcept)."""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch_scatter import segment_csr


def pool_axis_distribution_from_probs(
    probs: Tensor,
    target: Tensor,
    ptr: Tensor,
    ignore_index: int,
    num_classes: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pool per-point softmax probs and one-hot targets to per-tile distributions.

    `ptr` is a PyG `Batch.ptr` (padded cumulative point-count offsets, one tile per
    segment). Returns (pi_hat, q_t, n_t): the predicted per-tile distribution, the
    empirical per-tile target distribution, and the non-void point count per tile.
    """
    valid = target != ignore_index
    counted = valid.to(probs.dtype)

    one_hot = torch.zeros((target.size(0), num_classes), dtype=probs.dtype, device=probs.device)
    valid_idx = valid.nonzero(as_tuple=True)[0]
    one_hot[valid_idx, target[valid_idx]] = 1.0

    masked_probs = probs * counted.unsqueeze(-1)

    sum_probs = segment_csr(masked_probs, ptr, reduce="sum")
    sum_one_hot = segment_csr(one_hot, ptr, reduce="sum")
    n_t = segment_csr(counted, ptr, reduce="sum")

    denom = n_t.clamp(min=1.0).unsqueeze(-1)
    pi_hat = sum_probs / denom
    q_t = sum_one_hot / denom
    return pi_hat, q_t, n_t


def kl_divergence_rows(q: Tensor, p: Tensor, eps: float = 1e-8) -> Tensor:
    """Row-wise KL(q || p) for two batches of distributions, shape (T, C) -> (T,)."""
    return (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=-1)


def abs_freq_error_rows(pi_hat: Tensor, q_t: Tensor) -> Tensor:
    """Per-row |pi_hat - q_t|, shape (T, C)."""
    return (pi_hat.float() - q_t.float()).abs()


def tv_from_abs_errors(abs_err: Tensor) -> Tensor:
    """Per-row total variation ``sum_c |pi - q|``, shape (T, C) -> (T,)."""
    return abs_err.float().sum(dim=-1)


class WeightedKLDivLoss(nn.Module):
    """Point-count-weighted mean of KL(target_distribution || predicted_distribution)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        kl = kl_divergence_rows(target, pred, eps=self.eps)
        if weight is None:
            return kl.mean()
        weight = weight.to(kl.dtype)
        return (weight * kl).sum() / weight.sum().clamp(min=self.eps)
