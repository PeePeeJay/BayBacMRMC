import numpy as np
import pandas as pd
import pytest

from mrmc_baybac.utils import (
    common_fpr_interval,
    get_or_reference_curves,
    partial_roc_auc_from_curve,
    compare_partial_auc_between_settings,
)


def test_common_fpr_interval_returns_overlap():
    fpr_min, fpr_max = common_fpr_interval(
        [0.1, 0.3, 0.7],
        [0.2, 0.4, 0.8],
        [0.15, 0.5, 0.75],
    )
    assert np.isclose(fpr_min, 0.2)
    assert np.isclose(fpr_max, 0.7)


def test_get_or_reference_curves_keeps_fpr_tpr_alignment():
    specs = pd.DataFrame(
        {
            "sens": [0.3, 0.1, 0.2],
            "mean_0": [0.9, 0.7, 0.8],
            "mean_1": [0.95, 0.75, 0.85],
        }
    )
    curves = get_or_reference_curves(specs)

    fpr0, tpr0 = curves["0"]
    fpr1, tpr1 = curves["1"]

    assert np.allclose(fpr0, [0.1, 0.2, 0.3])
    assert np.allclose(fpr1, [0.1, 0.2, 0.3])
    assert np.allclose(tpr0, [0.7, 0.8, 0.9])
    assert np.allclose(tpr1, [0.75, 0.85, 0.95])


def test_partial_roc_auc_from_curve_interpolates_bounds():
    fpr = np.array([0.1, 0.2, 0.4, 0.8])
    tpr = np.array([0.5, 0.6, 0.75, 0.95])

    auc_val, fpr_partial, tpr_partial = partial_roc_auc_from_curve(
        fpr,
        tpr,
        fpr_min=0.15,
        fpr_max=0.6,
    )

    assert np.isfinite(auc_val)
    assert np.isclose(fpr_partial[0], 0.15)
    assert np.isclose(fpr_partial[-1], 0.6)
    assert len(fpr_partial) == len(tpr_partial)


def test_compare_partial_auc_between_settings_returns_probability_and_hdis():
    # Shape: (n_points, n_samples)
    fpr0 = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.4, 0.4, 0.4],
            [0.7, 0.7, 0.7],
        ]
    )
    tpr0 = np.array(
        [
            [0.50, 0.52, 0.51],
            [0.62, 0.63, 0.61],
            [0.73, 0.74, 0.72],
            [0.86, 0.87, 0.85],
        ]
    )

    fpr1 = fpr0.copy()
    tpr1 = tpr0 + 0.05

    result = compare_partial_auc_between_settings(
        fpr0,
        tpr0,
        fpr1,
        tpr1,
        fpr_min=0.1,
        fpr_max=0.7,
        hdi_prob=0.95,
    )

    assert result["n_valid_samples"] == 3
    assert np.isclose(result["prob_auc1_gt_auc0"], 1.0)
    assert len(result["auc_hdi_0"]) == 2
    assert len(result["auc_hdi_1"]) == 2
    assert len(result["auc_diff_hdi"]) == 2
    assert result["auc_diff_mean"] > 0


def test_compare_partial_auc_between_settings_uses_default_bounds():
    fpr0 = np.array(
        [
            [0.05, 0.05, 0.05],
            [0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20],
            [0.40, 0.40, 0.40],
            [0.70, 0.70, 0.70],
            [0.90, 0.90, 0.90],
        ]
    )
    tpr0 = np.array(
        [
            [0.40, 0.41, 0.39],
            [0.50, 0.51, 0.49],
            [0.60, 0.61, 0.59],
            [0.72, 0.73, 0.71],
            [0.83, 0.84, 0.82],
            [0.92, 0.93, 0.91],
        ]
    )
    fpr1 = fpr0.copy()
    tpr1 = tpr0 + 0.03

    result = compare_partial_auc_between_settings(
        fpr0,
        tpr0,
        fpr1,
        tpr1,
    )

    assert np.isclose(result["fpr_bounds"][0], 0.1)
    assert np.isclose(result["fpr_bounds"][1], 0.7)


def test_compare_partial_auc_between_settings_accepts_explicit_min_max():
    fpr0 = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.4, 0.4, 0.4],
            [0.7, 0.7, 0.7],
        ]
    )
    tpr0 = np.array(
        [
            [0.50, 0.52, 0.51],
            [0.62, 0.63, 0.61],
            [0.73, 0.74, 0.72],
            [0.86, 0.87, 0.85],
        ]
    )
    fpr1 = fpr0.copy()
    tpr1 = tpr0 + 0.02

    result = compare_partial_auc_between_settings(
        fpr0,
        tpr0,
        fpr1,
        tpr1,
        fpr_min=0.2,
        fpr_max=0.7,
    )

    assert np.isclose(result["fpr_bounds"][0], 0.2)
    assert np.isclose(result["fpr_bounds"][1], 0.7)


def test_compare_partial_auc_between_settings_raises_on_bad_shapes():
    fpr0 = np.array([0.1, 0.2, 0.4])
    tpr0 = np.array([0.5, 0.6, 0.7])
    fpr1 = np.array([0.1, 0.2, 0.4])
    tpr1 = np.array([0.5, 0.6, 0.7])

    with pytest.raises(ValueError):
        compare_partial_auc_between_settings(
            fpr0, tpr0, fpr1, tpr1
        )
