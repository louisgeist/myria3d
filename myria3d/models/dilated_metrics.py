"""Buffer ("relaxed") precision/recall/F1 for thin binary pixel masks.

Port of Pointcept ``pointcept/utils/dilated_metrics.py``: a predicted foreground
pixel counts as correct if it lies within ``radius_px`` of any GT foreground pixel
(precision), and a GT foreground pixel counts as found if it lies within
``radius_px`` of any predicted foreground pixel (recall). Dilation uses 8-connectivity
Chebyshev distance (``iterations=radius_px``).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

_NEIGH8_OFFSETS = np.array(
    [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ],
    dtype=np.int64,
)


def _shift_with_false_pad(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    if dr >= 0:
        src_r0, src_r1, dst_r0, dst_r1 = 0, h - dr, dr, h
    else:
        src_r0, src_r1, dst_r0, dst_r1 = -dr, h, 0, h + dr
    if dc >= 0:
        src_c0, src_c1, dst_c0, dst_c1 = 0, w - dc, dc, w
    else:
        src_c0, src_c1, dst_c0, dst_c1 = -dc, w, 0, w + dc
    if src_r0 < src_r1 and src_c0 < src_c1:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


def morph_dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary dilation, 8-connectivity, ``iterations`` steps (Chebyshev radius)."""
    out = np.asarray(mask, dtype=bool).copy()
    it = int(iterations)
    if it < 0:
        raise ValueError(f"iterations must be >= 0, got {it}")
    for _ in range(it):
        expanded = out.copy()
        for dr, dc in _NEIGH8_OFFSETS:
            expanded |= _shift_with_false_pad(out, int(dr), int(dc))
        out = expanded
    return out


def precision_recall_f1(
    num_p: float,
    denom_p: float,
    num_r: float,
    denom_r: float,
    eps: float = 1e-10,
) -> Tuple[float, float, float]:
    precision = num_p / (denom_p + eps)
    recall = num_r / (denom_r + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return float(precision), float(recall), float(f1)


def dilated_precision_recall_counts(
    pred_fg: np.ndarray,
    gt_fg: np.ndarray,
    valid: np.ndarray,
    radius_px: int,
) -> Tuple[float, float, float, float]:
    """Return ``(precision_num, precision_denom, recall_num, recall_denom)``."""
    pred_fg = np.asarray(pred_fg, dtype=bool)
    gt_fg = np.asarray(gt_fg, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if pred_fg.shape != gt_fg.shape or pred_fg.shape != valid.shape:
        raise ValueError(
            "pred_fg/gt_fg/valid must share the same shape, got "
            f"{pred_fg.shape}, {gt_fg.shape}, {valid.shape}"
        )
    radius_px = int(radius_px)
    if radius_px < 0:
        raise ValueError(f"radius_px must be >= 0, got {radius_px}")

    gt_dilated = morph_dilate_mask(gt_fg, iterations=radius_px)
    pred_dilated = morph_dilate_mask(pred_fg, iterations=radius_px)
    precision_num = float(np.count_nonzero(pred_fg & gt_dilated & valid))
    precision_denom = float(np.count_nonzero(pred_fg & valid))
    recall_num = float(np.count_nonzero(gt_fg & pred_dilated & valid))
    recall_denom = float(np.count_nonzero(gt_fg & valid))
    return precision_num, precision_denom, recall_num, recall_denom


def dilated_prf_enabled(task_config: dict) -> bool:
    """Pointcept default is True; forest_2d opts out (area coverage, not thin lines)."""
    return bool(task_config.get("enable_dilated_prf", True))
