import os
import unittest.mock

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
