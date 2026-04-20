import os

import arviz as az
import pytest
from mrmc_baybac.model import BaseModel, BalancedModel
from mrmc_baybac.simulation import (
    simulate_aggregated_data,
    mock_reading_data,
)
import numpy as np


def _fake_aggregated_idata():
    return az.from_dict(
        posterior={
            "alpha": np.array(
                [[[0.2, -0.1], [0.0, 0.1], [-0.2, 0.3]]]
            ),
            "beta": np.array(
                [[[0.15, 0.05], [0.1, -0.05], [0.05, 0.1]]]
            ),
            "gamma": np.array([[0.15, 0.2, 0.25]]),
        },
        coords={"reader": [0, 1]},
        dims={"alpha": ["reader"], "beta": ["reader"]},
    )


@pytest.fixture
def aggregated_negative_sim_data():
    return simulate_aggregated_data(
        n_readers=4,
        n_cases=120,
        mu_baseline=0.79,
        effect_size=0.04,
        gamma=0.1,
    )


@pytest.fixture
def aggregated_positive_sim_data():
    return simulate_aggregated_data(
        n_readers=4,
        n_cases=80,
        mu_baseline=0.75,
        effect_size=0.06,
        gamma=0.1,
    )


@pytest.fixture
def simulated_aggregated_data(
    aggregated_negative_sim_data,
    aggregated_positive_sim_data,
):
    return mock_reading_data(
        aggregated_negative_sim_data,
        aggregated_positive_sim_data,
    )


class TestRunModel:
    """Tests model inference on simulated data"""

    def test_run_inference_on_simulated_data(
        self, simulated_aggregated_data
    ):
        m = BalancedModel(
            obs_data=simulated_aggregated_data,
            priors="weakly informative",
        )
        idatas = m.run_inference()
        assert "posterior" in idatas[0].keys()
        assert "posterior_predictive" in idatas[0].keys()


def test__run_inference_with_default_priors(vandyke_df):
    obs_data = vandyke_df
    rating_threshold = 3
    m = BaseModel(obs_data=obs_data)

    idata, model = m.run_inference(
        obs_data=obs_data, rating_threshold=rating_threshold
    )
    assert "posterior" in idata.keys()
    assert "posterior_predictive" in idata.keys()


def test_balanced_model_run_inference(vandyke_df):
    """Test BalancedModel._run_inference returns inference data for both truth values."""
    obs_data = vandyke_df
    rating_threshold = 3

    balanced_model = BalancedModel(obs_data=obs_data)
    idatas = balanced_model.run_inference(
        rating_threshold=rating_threshold
    )

    # Check that we get a list of 2 inference data objects
    assert isinstance(idatas, list)
    assert len(idatas) == 2

    # Check that each idata has the required keys
    for idata in idatas:
        assert "posterior" in idata.keys()
        assert "posterior_predictive" in idata.keys()


def test_roc_curve_analysis(vandyke_df):
    """Test BalancedModel.roc_curve_analysis returns ROC curve data with AUC."""
    obs_data = vandyke_df

    balanced_model = BalancedModel(obs_data=obs_data)
    roc_results = balanced_model.roc_curve_analysis()

    # Check that we get results for both treatment settings
    assert isinstance(roc_results, dict)
    assert "0" in roc_results
    assert "1" in roc_results

    # Check that each treatment setting has the required ROC components
    for setting in ["0", "1"]:
        assert "fpr" in roc_results[setting]
        assert "tpr" in roc_results[setting]
        assert "auc" in roc_results[setting]
        assert "partial_auc" in roc_results[setting]
        assert "partial_fpr_range" in roc_results[setting]

        # Validate FPR and TPR are lists
        assert isinstance(roc_results[setting]["fpr"], list)
        assert isinstance(roc_results[setting]["tpr"], list)

        # Validate AUC is a float between 0 and 1
        assert isinstance(
            roc_results[setting]["auc"], float
        )
        assert 0 <= roc_results[setting]["auc"] <= 1

        # Validate partial AUC is a float between 0 and 1
        assert isinstance(
            roc_results[setting]["partial_auc"], float
        )
        assert 0 <= roc_results[setting]["partial_auc"] <= 1

        # Validate partial FPR range is a tuple of two floats
        assert isinstance(
            roc_results[setting]["partial_fpr_range"], tuple
        )
        assert (
            len(roc_results[setting]["partial_fpr_range"])
            == 2
        )
        assert all(
            isinstance(x, float)
            for x in roc_results[setting][
                "partial_fpr_range"
            ]
        )

        # Validate FPR and TPR have the same length
        assert len(roc_results[setting]["fpr"]) == len(
            roc_results[setting]["tpr"]
        )

    # additionally, ensure partial range corresponds to the shared numeric
    # FPR interval across treatment settings
    for setting in ["0", "1"]:
        other = "1" if setting == "0" else "0"
        fpr = np.array(roc_results[setting]["fpr"])
        fpr_other = np.array(roc_results[other]["fpr"])
        expect_min = max(
            float(fpr.min()), float(fpr_other.min())
        )
        expect_max = min(
            float(fpr.max()), float(fpr_other.max())
        )
        got_min, got_max = roc_results[setting][
            "partial_fpr_range"
        ]
        assert np.isclose(got_min, expect_min)
        assert np.isclose(got_max, expect_max)


def test_roc_curve_analysis_cxr(cxr_df):
    """Test BalancedModel.roc_curve_analysis returns ROC curve data with AUC."""
    obs_data = cxr_df

    balanced_model = BalancedModel(obs_data=obs_data)
    roc_results = balanced_model.roc_curve_analysis()

    # Check that we get results for both treatment settings
    assert isinstance(roc_results, dict)
    assert "0" in roc_results
    assert "1" in roc_results

    # Check that each treatment setting has the required ROC components
    for setting in ["0", "1"]:
        assert "fpr" in roc_results[setting]
        assert "tpr" in roc_results[setting]
        assert "auc" in roc_results[setting]
        assert "partial_auc" in roc_results[setting]
        assert "partial_fpr_range" in roc_results[setting]

        # Validate FPR and TPR are lists
        assert isinstance(roc_results[setting]["fpr"], list)
        assert isinstance(roc_results[setting]["tpr"], list)

        # Validate AUC is a float between 0 and 1
        assert isinstance(
            roc_results[setting]["auc"], float
        )
        assert 0 <= roc_results[setting]["auc"] <= 1

        # Validate partial AUC is a float between 0 and 1
        assert isinstance(
            roc_results[setting]["partial_auc"], float
        )
        assert 0 <= roc_results[setting]["partial_auc"] <= 1

        # Validate partial FPR range is a tuple of two floats
        assert isinstance(
            roc_results[setting]["partial_fpr_range"], tuple
        )
        assert (
            len(roc_results[setting]["partial_fpr_range"])
            == 2
        )
        assert all(
            isinstance(x, float)
            for x in roc_results[setting][
                "partial_fpr_range"
            ]
        )

        # Validate FPR and TPR have the same length
        assert len(roc_results[setting]["fpr"]) == len(
            roc_results[setting]["tpr"]
        )


def test_compute_tpr_tnr_new_cases_balanced_model(
    simulated_aggregated_data, monkeypatch
):
    balanced_model = BalancedModel(
        obs_data=simulated_aggregated_data
    )
    neg_idata = _fake_aggregated_idata()
    pos_idata = _fake_aggregated_idata()
    monkeypatch.setattr(
        balanced_model,
        "run_inference",
        lambda threshold: [neg_idata, pos_idata],
    )

    small_tpr, small_tnr = balanced_model._compute_tpr_tnr(
        0.5,
        predictive_target="new_cases",
        n_new_cases=2,
    )
    large_tpr, _ = balanced_model._compute_tpr_tnr(
        0.5,
        predictive_target="new_cases",
        n_new_cases=25,
    )

    for setting in ["0", "1"]:
        assert small_tpr[setting].shape == (1, 3)
        assert small_tnr[setting].shape == (1, 3)
        assert np.all((0 <= small_tpr[setting]) & (small_tpr[setting] <= 1))
        assert np.all((0 <= small_tnr[setting]) & (small_tnr[setting] <= 1))

    assert not np.allclose(small_tpr["0"], large_tpr["0"])


def test_plot_tpr_tnr_by_threshold_forwards_new_case_arguments(
    vandyke_df, tmp_path, monkeypatch
):
    balanced_model = BalancedModel(obs_data=vandyke_df)
    calls = []

    def fake_compute(
        threshold,
        predictive_target="observed_panel",
        n_new_cases=None,
    ):
        calls.append((predictive_target, n_new_cases))
        tpr = np.array([[0.6, 0.7]])
        tnr = np.array([[0.8, 0.85]])
        return {"0": tpr, "1": tpr}, {"0": tnr, "1": tnr}

    monkeypatch.setattr(
        balanced_model,
        "_compute_tpr_tnr",
        fake_compute,
    )

    out_file = tmp_path / "tpr_tnr_new_cases.png"
    path = balanced_model.plot_tpr_tnr_by_threshold(
        filename=str(out_file),
        predictive_target="new_cases",
        n_new_cases=17,
    )

    assert os.path.isfile(path[0])
    assert calls
    assert all(target == "new_cases" for target, _ in calls)
    assert all(case_count == 17 for _, case_count in calls)


def test_plot_tpr_tnr_by_threshold_creates_file(
    vandyke_df, tmp_path
):
    """BalancedModel.plot_tpr_tnr_by_threshold should save a figure file."""
    balanced_model = BalancedModel(obs_data=vandyke_df)
    out_file = "./tests/.figures/tpr_tnr_by_threshold.png"
    path = balanced_model.plot_tpr_tnr_by_threshold(
        filename=str(out_file)
    )
    assert os.path.isfile(path)
    # optional: ensure extension matches
    assert path.endswith(".png")


def test_plot_tpr_tnr_by_threshold_cxr_data(
    cxr_df, tmp_path
):
    """BalancedModel.plot_tpr_tnr_by_threshold should save a figure file."""
    balanced_model = BalancedModel(obs_data=cxr_df)
    out_file = (
        "./tests/.figures/tpr_tnr_by_threshold_cxr.png"
    )
    path = balanced_model.plot_tpr_tnr_by_threshold(
        filename=str(out_file)
    )
    assert os.path.isfile(path)
    # optional: ensure extension matches
    assert path.endswith(".png")


def test_plot_tpr_tnr_by_threshold_cxr_data_informative(
    cxr_df, tmp_path
):
    """BalancedModel.plot_tpr_tnr_by_threshold should save a figure file."""
    balanced_model = BalancedModel(
        obs_data=cxr_df, priors="informative"
    )
    out_file = "./tests/.figures/tpr_tnr_by_threshold_cxr_informative.png"
    path = balanced_model.plot_tpr_tnr_by_threshold(
        filename=str(out_file)
    )
    assert os.path.isfile(path)
    # optional: ensure extension matches
    assert path.endswith(".png")


def test_plot_roc_curve_with_hdi_creates_file(
    vandyke_df, tmp_path
):
    """BalancedModel.plot_roc_curve_with_hdi should save a figure file."""
    balanced_model = BalancedModel(obs_data=vandyke_df)
    out_file = "./tests/.figures/roc_curve_with_hdi_vd.png"
    path = balanced_model.plot_roc_curve_with_hdi(
        filename=str(out_file)
    )
    assert os.path.isfile(path)
    # optional: ensure extension matches
    assert path.endswith(".png")


def test_plot_roc_curve_with_hdi_cxr_data(cxr_df, tmp_path):
    """BalancedModel.plot_roc_curve_with_hdi should save a figure file with CXR data."""
    balanced_model = BalancedModel(obs_data=cxr_df)
    out_file = "./tests/.figures/roc_curve_with_hdi_cxr.png"
    path = balanced_model.plot_roc_curve_with_hdi(
        filename=str(out_file)
    )
    assert os.path.isfile(path)
    # optional: ensure extension matches
    assert path.endswith(".png")


def test_plot_roc_curve_with_hdi_cxr_data_weakly_inf_priors(
    cxr_df, tmp_path
):
    """BalancedModel.plot_roc_curve_with_hdi should save a figure file with CXR data."""
    balanced_model = BalancedModel(
        obs_data=cxr_df, priors="weakly informative"
    )
    out_file = "./tests/.figures/roc_curve_with_hdi_cxr.png"
    path = balanced_model.plot_roc_curve_with_hdi(
        filename=str(out_file)
    )
    assert os.path.isfile(path)
    # optional: ensure extension matches
    assert path.endswith(".png")
