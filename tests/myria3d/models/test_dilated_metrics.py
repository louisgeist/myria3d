import numpy as np
import pytest

from myria3d.models.dilated_metrics import (
    dilated_prf_enabled,
    dilated_precision_recall_counts,
    precision_recall_f1,
)


def test_precision_recall_f1_from_counts():
    precision, recall, f1 = precision_recall_f1(1.0, 2.0, 1.0, 2.0)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_dilated_prf_opt_out_matches_pointcept():
    assert dilated_prf_enabled({"enable_dilated_prf": False}) is False
    assert dilated_prf_enabled({"enable_dilated_prf": True}) is True
    # Pointcept default: on unless a task (forest_2d) opts out.
    assert dilated_prf_enabled({}) is True


def test_chebyshev_radius_3_hits_distance_4_misses():
    """8-connect dilation of `radius_px` iterations = Chebyshev ball of that radius."""
    size = 11
    gt = np.zeros((size, size), dtype=bool)
    valid = np.ones((size, size), dtype=bool)
    gt[5, 5] = True

    pred_dist3 = np.zeros_like(gt)
    pred_dist3[5 + 3, 5 + 3] = True  # diagonal Chebyshev distance 3
    p_num, p_denom, r_num, r_denom = dilated_precision_recall_counts(
        pred_dist3, gt, valid, radius_px=3
    )
    assert (p_num, p_denom, r_num, r_denom) == (1.0, 1.0, 1.0, 1.0)

    pred_dist4 = np.zeros_like(gt)
    pred_dist4[5 + 4, 5 + 4] = True
    p_num, p_denom, r_num, r_denom = dilated_precision_recall_counts(
        pred_dist4, gt, valid, radius_px=3
    )
    assert (p_num, p_denom, r_num, r_denom) == (0.0, 1.0, 0.0, 1.0)


def test_one_pixel_offset_kills_exact_f1_but_dilated_hits():
    pred = np.zeros((3, 3), dtype=bool)
    gt = np.zeros((3, 3), dtype=bool)
    valid = np.ones((3, 3), dtype=bool)
    gt[1, 0] = True
    pred[1, 1] = True

    precision, recall, f1 = precision_recall_f1(0.0, 1.0, 0.0, 1.0)
    assert f1 == pytest.approx(0.0, abs=1e-8)
    assert precision == pytest.approx(0.0, abs=1e-8)
    assert recall == pytest.approx(0.0, abs=1e-8)

    p_num, p_denom, r_num, r_denom = dilated_precision_recall_counts(
        pred, gt, valid, radius_px=1
    )
    d_p, d_r, d_f1 = precision_recall_f1(p_num, p_denom, r_num, r_denom)
    assert (p_num, p_denom, r_num, r_denom) == (1.0, 1.0, 1.0, 1.0)
    assert d_p == pytest.approx(1.0)
    assert d_r == pytest.approx(1.0)
    assert d_f1 == pytest.approx(1.0)
