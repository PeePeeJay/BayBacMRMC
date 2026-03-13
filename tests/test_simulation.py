import numpy as np
import pandas as pd
import pytest
from mrmc_baybac.simulation import simulate_aggregated_data, mock_reading_data

EXPECTED_KEYS = {"k", "alpha", "beta", "p", "a_beta", "b_beta"}


@pytest.fixture
def sim_inputs():
    n_readers = 3
    true_params = {
        "mu_a": 1.0,
        "sigma_a": 0.5,
        "mu_b": 0.5,
        "sigma_b": 0.3,
        "gamma": 0.2,
    }
    return dict(
        n_readers=n_readers,
        n_cases=30,
        true_params=true_params,
    )


class TestSimulateData:
    def test_returns_dict(self, sim_inputs):
        result = simulate_aggregated_data(**sim_inputs)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, sim_inputs):
        result = simulate_aggregated_data(**sim_inputs)
        assert EXPECTED_KEYS.issubset(result.keys())

    def test_alpha_shape(self, sim_inputs):
        result = simulate_aggregated_data(**sim_inputs)
        assert result["alpha"].shape == (sim_inputs["n_readers"],)

    def test_beta_shape(self, sim_inputs):
        result = simulate_aggregated_data(**sim_inputs)
        assert result["beta"].shape == (sim_inputs["n_readers"],)

    def test_p_in_unit_interval(self, sim_inputs):
        result = simulate_aggregated_data(**sim_inputs)
        assert (result["p"] >= 0).all()
        assert (result["p"] <= 1).all()


    def test_k_bounded_by_n_cases(self, sim_inputs):
        result = simulate_aggregated_data(**sim_inputs)
        assert (result["k"] >= 0).all()
        assert (result["k"] <= sim_inputs["n_cases"]).all()


    def test_a_beta_positive(self, sim_inputs):
        result = simulate_aggregated_data(**sim_inputs)
        assert (result["a_beta"] > 0).all()

    def test_b_beta_positive(self, sim_inputs):
        result = simulate_aggregated_data(**sim_inputs)
        assert (result["b_beta"] > 0).all()

    def test_p_shape(self, sim_inputs):
        result = simulate_aggregated_data(**sim_inputs)
        assert result["p"].shape == (sim_inputs["n_readers"], 2)

    def test_k_shape(self, sim_inputs):
        result = simulate_aggregated_data(**sim_inputs)
        assert result["k"].shape == (sim_inputs["n_readers"], 2)

    def test_k_reader_setting_access(self, sim_inputs):
        """k[reader, setting] returns a scalar count per reader per setting."""
        n_readers = sim_inputs["n_readers"]
        result = simulate_aggregated_data(**sim_inputs)
        for reader in range(n_readers):
            for setting in range(2):
                val = result["k"][reader, setting]
                assert 0 <= val <= sim_inputs["n_cases"]

    def test_reproducible_with_same_rng(self, sim_inputs):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        r1 = simulate_aggregated_data(**sim_inputs, rng=rng1)
        r2 = simulate_aggregated_data(**sim_inputs, rng=rng2)
        np.testing.assert_array_equal(r1["k"], r2["k"])
        np.testing.assert_array_equal(r1["alpha"], r2["alpha"])

    def test_different_rng_produces_different_alpha(self, sim_inputs):
        r1 = simulate_aggregated_data(**sim_inputs, rng=np.random.default_rng(1))
        r2 = simulate_aggregated_data(**sim_inputs, rng=np.random.default_rng(2))
        assert not np.array_equal(r1["alpha"], r2["alpha"])


MOCK_READING_EXPECTED_COLUMNS = {"reader", "case", "treatment", "rating", "truth"}

TRUE_PARAMS = {
    "mu_a": 1.0,
    "sigma_a": 0.5,
    "mu_b": 0.5,
    "sigma_b": 0.3,
    "gamma": 0.2,
}


@pytest.fixture
def negative_sim_data():
    return simulate_aggregated_data(n_readers=3, n_cases=5, true_params=TRUE_PARAMS)


@pytest.fixture
def positive_sim_data():
    return simulate_aggregated_data(n_readers=3, n_cases=5, true_params=TRUE_PARAMS)


class TestMockReadingData:
    def test_returns_dataframe(self, negative_sim_data, positive_sim_data):
        result = mock_reading_data(negative_sim_data, positive_sim_data)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, negative_sim_data, positive_sim_data):
        result = mock_reading_data(negative_sim_data, positive_sim_data)
        assert MOCK_READING_EXPECTED_COLUMNS.issubset(result.columns)

    def test_row_count(self, negative_sim_data, positive_sim_data):
        n_readers = negative_sim_data["n_readers"]
        n_cases = negative_sim_data["n_cases"]
        result = mock_reading_data(negative_sim_data, positive_sim_data)
        assert len(result) == 2 * 2 * n_readers * n_cases

    def test_reader_values_in_range(self, negative_sim_data, positive_sim_data):
        n_readers = negative_sim_data["n_readers"]
        result = mock_reading_data(negative_sim_data, positive_sim_data)
        assert set(result["reader"].unique()) == set(range(n_readers))

    def test_case_values_in_range(self, negative_sim_data, positive_sim_data):
        n_cases = negative_sim_data["n_cases"]
        result = mock_reading_data(negative_sim_data, positive_sim_data)
        assert set(result["case"].unique()) == set(range(n_cases))

    def test_correct_rating_count_negative(self, negative_sim_data, positive_sim_data):
        """For negative cases, correct ratings (rating==0==truth) per reader/treatment equal k_neg."""
        result = mock_reading_data(negative_sim_data, positive_sim_data)
        neg = result[result["truth"] == 0]
        for reader in range(negative_sim_data["n_readers"]):
            for treatment in [0, 1]:
                mask = (neg["reader"] == reader) & (neg["treatment"] == treatment)
                n_correct = (neg.loc[mask, "rating"] == 0).sum()
                assert n_correct == negative_sim_data["k"][reader, treatment]

    def test_correct_rating_count_positive(self, negative_sim_data, positive_sim_data):
        """For positive cases, correct ratings (rating==1==truth) per reader/treatment equal k_pos."""
        result = mock_reading_data(negative_sim_data, positive_sim_data)
        pos = result[result["truth"] == 1]
        for reader in range(positive_sim_data["n_readers"]):
            for treatment in [0, 1]:
                mask = (pos["reader"] == reader) & (pos["treatment"] == treatment)
                n_correct = (pos.loc[mask, "rating"] == 1).sum()
                assert n_correct == positive_sim_data["k"][reader, treatment]
