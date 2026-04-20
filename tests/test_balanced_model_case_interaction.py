import os
import unittest.mock
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest

from mrmc_baybac.model import (
    BalancedCaseInteractionModel,
)
from mrmc_baybac.simulation import (
    simulate_case_data,
    mock_case_reading_data,
)


def _fake_case_interaction_idata():
    return az.from_dict(
        posterior={
            "alpha": np.array(
                [[[0.2, -0.1], [0.1, 0.0], [-0.2, 0.25]]]
            ),
            "beta": np.array(
                [[[0.1, 0.05], [0.05, 0.1], [0.0, 0.15]]]
            ),
            "mu_gamma_c": np.array([[0.0, 0.1, -0.05]]),
            "sigma_gamma_c": np.array([[0.35, 0.35, 0.35]]),
            "mu_delta_rc": np.array([[0.0, 0.05, -0.05]]),
            "sigma_delta_rc": np.array([[0.25, 0.25, 0.25]]),
        },
        coords={"reader": [0, 1]},
        dims={"alpha": ["reader"], "beta": ["reader"]},
    )


@pytest.fixture
def case_negative_sim_data():
    return simulate_case_data(
        n_readers=4,
        n_cases=120,
        mu_baseline=0.79,
        effect_size=0.04,
    )


@pytest.fixture
def case_positive_sim_data():
    return simulate_case_data(
        n_readers=4,
        n_cases=80,
        mu_baseline=0.75,
        effect_size=0.06,
    )


@pytest.fixture
def simulated_case_data(
    case_negative_sim_data, case_positive_sim_data
):
    return mock_case_reading_data(
        case_negative_sim_data, case_positive_sim_data
    )


class TestRunModel:
    def test_run_inference_on_simulated_case_data(
        self, simulated_case_data
    ):
        m = BalancedCaseInteractionModel(
            obs_data=simulated_case_data,
            priors="weakly informative",
        )
        idatas = m.run_inference()
        assert isinstance(idatas, list)
        assert "posterior" in idatas[0].keys()
        assert "posterior_predictive" in idatas[0].keys()
        assert "posterior" in idatas[1].keys()


def test_compute_tpr_tnr_new_cases_case_interaction(
    simulated_case_data, monkeypatch
):
    model = BalancedCaseInteractionModel(
        obs_data=simulated_case_data,
        priors="weakly informative",
    )
    neg_idata = _fake_case_interaction_idata()
    pos_idata = _fake_case_interaction_idata()
    monkeypatch.setattr(
        model,
        "run_inference",
        lambda threshold: [neg_idata, pos_idata],
    )

    small_tpr, small_tnr = model._compute_tpr_tnr(
        0.5,
        predictive_target="new_cases",
        n_new_cases=3,
    )
    large_tpr, _ = model._compute_tpr_tnr(
        0.5,
        predictive_target="new_cases",
        n_new_cases=30,
    )

    for setting in ["0", "1"]:
        assert small_tpr[setting].shape == (1, 3)
        assert small_tnr[setting].shape == (1, 3)
        assert np.all((0 <= small_tpr[setting]) & (small_tpr[setting] <= 1))
        assert np.all((0 <= small_tnr[setting]) & (small_tnr[setting] <= 1))

    assert not np.allclose(small_tpr["1"], large_tpr["1"])


def test_compute_tpr_tnr_case_interaction_complements_negative_predictions(
    simulated_case_data, monkeypatch
):
    model = BalancedCaseInteractionModel(
        obs_data=simulated_case_data,
        priors="weakly informative",
    )

    neg_idata = SimpleNamespace(
        posterior_predictive={
            "k": SimpleNamespace(
                values=np.array([[[0.2, 0.4, 0.6, 0.8]]])
            )
        },
        constant_data={
            "treatment_idx": SimpleNamespace(
                values=np.array([0, 0, 1, 1])
            )
        },
    )
    pos_idata = SimpleNamespace(
        posterior_predictive={
            "k": SimpleNamespace(
                values=np.array([[[0.7, 0.9, 0.5, 0.6]]])
            )
        },
        constant_data={
            "treatment_idx": SimpleNamespace(
                values=np.array([0, 0, 1, 1])
            )
        },
    )

    monkeypatch.setattr(
        model,
        "run_inference",
        lambda threshold: [neg_idata, pos_idata],
    )

    tpr_dict, tnr_dict = model._compute_tpr_tnr(0.5)

    assert np.allclose(tnr_dict["0"], np.array([[0.7]]))
    assert np.allclose(tnr_dict["1"], np.array([[0.3]]))
    assert np.allclose(tpr_dict["0"], np.array([[0.8]]))
    assert np.allclose(tpr_dict["1"], np.array([[0.55]]))


def test_compute_tpr_tnr_new_cases_case_interaction_complements_negative_predictions(
    simulated_case_data, monkeypatch
):
    model = BalancedCaseInteractionModel(
        obs_data=simulated_case_data,
        priors="weakly informative",
    )
    neg_idata = object()
    pos_idata = object()

    monkeypatch.setattr(
        model,
        "run_inference",
        lambda threshold: [neg_idata, pos_idata],
    )

    def fake_simulate(idata, treatment, n_new_cases, random_seed):
        if idata is neg_idata:
            return np.full((1, 3), 0.2 + 0.1 * treatment)
        return np.full((1, 3), 0.6 + 0.1 * treatment)

    monkeypatch.setattr(
        model,
        "_simulate_new_case_accuracy",
        fake_simulate,
    )

    tpr_dict, tnr_dict = model._compute_tpr_tnr(
        0.5,
        predictive_target="new_cases",
        n_new_cases=5,
    )

    assert np.allclose(tnr_dict["0"], np.full((1, 3), 0.8))
    assert np.allclose(tnr_dict["1"], np.full((1, 3), 0.7))
    assert np.allclose(tpr_dict["0"], np.full((1, 3), 0.6))
    assert np.allclose(tpr_dict["1"], np.full((1, 3), 0.7))
