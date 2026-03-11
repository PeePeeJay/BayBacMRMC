import os
import unittest.mock

import numpy as np
import pandas as pd
import pytest

from mrmc_baybac.model import (
    BalancedCaseInteractionModel,
    BalancedModel,
)


@pytest.fixture
def binary_obs_data():
    """Individual-level binary observations compatible with BalancedCaseInteractionModel._setup_model.

    Each row represents a single reader's decision on a single case under one
    treatment setting.  The `k` column is binary (0 or 1) as required by the
    Bernoulli likelihood used in this model.
    """
    rng = np.random.default_rng(42)
    n_readers = 3
    n_cases = 5
    records = []
    for reader in range(n_readers):
        for case in range(n_cases):
            for treatment in [0, 1]:
                records.append(
                    {
                        "reader": f"reader_{reader}",
                        "case": f"case",
                        "treatment": treatment,
                        "k": int(rng.integers(0, 2)),
                    }
                )
    return pd.DataFrame(records)


def test_balanced_case_interaction_model_instantiation(
    vandyke_df,
):
    """BalancedCaseInteractionModel should instantiate without errors."""
    model = BalancedCaseInteractionModel(
        obs_data=vandyke_df
    )
    assert model is not None
    assert model.roc_results is None


def test_balanced_case_interaction_model_is_balanced_model_subclass(
    vandyke_df,
):
    """BalancedCaseInteractionModel must be a subclass of BalancedModel."""
    model = BalancedCaseInteractionModel(
        obs_data=vandyke_df
    )
    assert isinstance(model, BalancedModel)


def test_balanced_case_interaction_model_default_priors(
    vandyke_df,
):
    """Default priors ('diffuse') should be parsed without errors."""
    model = BalancedCaseInteractionModel(
        obs_data=vandyke_df
    )
    for key in (
        "a_mu",
        "a_sigma",
        "b_mu",
        "b_sigma",
        "gamma_mu",
        "gamma_sigma",
    ):
        assert key in model.priors


def test_balanced_case_interaction_model_custom_priors(
    vandyke_df, custom_priors
):
    """Custom prior dict should be stored and parsed correctly."""
    model = BalancedCaseInteractionModel(
        obs_data=vandyke_df, priors=custom_priors
    )
    assert model.priors["a_mu"] == custom_priors["a"][0]
    assert model.priors["a_sigma"] == custom_priors["a"][1]
    assert (
        model.priors["gamma_mu"]
        == custom_priors["gamma"][0]
    )


def test_balanced_case_interaction_setup_model_contains_case_variability(
    vandyke_df, binary_obs_data
):
    """_setup_model should include a 'case_variability' free random variable."""
    model = BalancedCaseInteractionModel(
        obs_data=vandyke_df
    )
    pm_model = model._setup_model(
        binary_obs_data, model.priors
    )
    free_rv_names = [v.name for v in pm_model.free_RVs]
    assert any(
        "case_variability" in name for name in free_rv_names
    )


def test_balanced_case_interaction_setup_model_contains_reader_case_interaction(
    vandyke_df, binary_obs_data
):
    """_setup_model should include a 'reader_case_interaction' free random variable."""
    model = BalancedCaseInteractionModel(
        obs_data=vandyke_df
    )
    pm_model = model._setup_model(
        binary_obs_data, model.priors
    )
    free_rv_names = [v.name for v in pm_model.free_RVs]
    assert any(
        "reader_case_interaction" in name
        for name in free_rv_names
    )


def test_balanced_case_interaction_setup_model_contains_population_params(
    vandyke_df, binary_obs_data
):
    """_setup_model should retain the population-level parameters mu_a and mu_b."""
    model = BalancedCaseInteractionModel(
        obs_data=vandyke_df
    )
    pm_model = model._setup_model(
        binary_obs_data, model.priors
    )
    free_rv_names = [v.name for v in pm_model.free_RVs]
    assert any("mu_a" in name for name in free_rv_names)
    assert any("mu_b" in name for name in free_rv_names)


def test_balanced_case_interaction_run_inference_returns_list_of_idatas(
    vandyke_df,
):
    """_run_inference must return a list of exactly two InferenceData objects."""
    model = BalancedCaseInteractionModel(
        obs_data=vandyke_df
    )
    idatas = model._run_inference(rating_threshold=3)
    assert isinstance(idatas, list)
    assert len(idatas) == 2


def test_balanced_case_interaction_run_inference_idata_has_required_groups(
    vandyke_df,
):
    """Each InferenceData returned by _run_inference must have posterior and posterior_predictive."""
    model = BalancedCaseInteractionModel(
        obs_data=vandyke_df
    )
    idatas = model._run_inference(rating_threshold=3)
    for idata in idatas:
        assert "posterior" in idata.keys()
        assert "posterior_predictive" in idata.keys()


def test_balanced_case_interaction_run_inference_posterior_has_population_params(
    vandyke_df,
):
    """Posterior must include mu_a and mu_b for both truth-class idatas."""
    model = BalancedCaseInteractionModel(
        obs_data=vandyke_df
    )
    idatas = model._run_inference(rating_threshold=3)
    for idata in idatas:
        assert "mu_a" in idata["posterior"]
        assert "mu_b" in idata["posterior"]


def test_plot_tpr_tnr_by_threshold_returns_filepath(
    vandyke_df,
):
    """plot_tpr_tnr_by_threshold should return the path supplied as filename."""
    model = BalancedCaseInteractionModel(
        obs_data=vandyke_df
    )
    out_path = (
        f"./tests/.figures/tpr_tnr_case_interaction.png"
    )

    result = model.plot_tpr_tnr_by_threshold(
        filename=out_path
    )

    assert result == out_path


def test_plot_tpr_tnr_by_threshold_returns_filepath_cxr(
    cxr_df,
):
    """plot_tpr_tnr_by_threshold should return the path supplied as filename."""
    model = BalancedCaseInteractionModel(obs_data=cxr_df)
    out_path = (
        f"./tests/.figures/tpr_tnr_case_interaction_cxr.png"
    )

    result = model.plot_tpr_tnr_by_threshold(
        filename=out_path
    )

    assert result == out_path
