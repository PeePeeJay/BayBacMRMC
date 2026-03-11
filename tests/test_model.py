import os
from mrmc_baybac.model import BaseModel, BalancedModel
import numpy as np


def test__run_inference_with_default_priors(vandyke_df):
    obs_data = vandyke_df
    rating_threshold = 3
    m = BaseModel(obs_data=obs_data)

    idata, model = m._run_inference(
        obs_data=obs_data, rating_threshold=rating_threshold
    )
    assert "posterior" in idata.keys()
    assert "posterior_predictive" in idata.keys()


def test_balanced_model_run_inference(vandyke_df):
    """Test BalancedModel._run_inference returns inference data for both truth values."""
    obs_data = vandyke_df
    rating_threshold = 3

    balanced_model = BalancedModel(obs_data=obs_data)
    idatas = balanced_model._run_inference(
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

    # additionally, ensure partial range corresponds to discrete overlap of FPR lists
    for setting in ["0", "1"]:
        other = "1" if setting == "0" else "0"
        fpr = np.array(roc_results[setting]["fpr"])
        fpr_other = np.array(roc_results[other]["fpr"])
        common = np.intersect1d(
            np.round(fpr, 6), np.round(fpr_other, 6)
        )
        if len(common) >= 2:
            expect_min, expect_max = float(
                common.min()
            ), float(common.max())
        else:
            # fallback to numeric intersection used by implementation
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
