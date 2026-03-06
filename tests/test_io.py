import pytest
import pandas as pd
from pathlib import Path
import os
from mrmc_baybac.model import BaseModel, BalancedModel
from mrmc_baybac.utils import get_thresholds_from_ratings


@pytest.fixture
def vandyke_csv_path():
    """Provides the relative path to the vandyke.csv file as a string."""
    return "data/vandyke.csv"


@pytest.fixture
def vandyke_df(vandyke_csv_path):
    """Provides the vandyke.csv data as a pandas DataFrame."""
    return pd.read_csv(vandyke_csv_path)

@pytest.fixture
def cxr_df():
    """Provides the vandyke.csv data as a pandas DataFrame."""
    return pd.read_csv("data/cxr_bsi_mrmc.csv")

@pytest.fixture
def vandyke_df_invalid(vandyke_csv_path):
    """Provides the vandyke.csv data as a pandas DataFrame,
    with one invalid column"""
    df = pd.read_csv(vandyke_csv_path)
    df.rename(columns={"reader": "readers"}, inplace=True)
    return df


@pytest.fixture
def custom_priors():
    return {
        "a": [0.0, 1.0],
        "b": [0.0, 1.0],
        "gamma": [0.0, 1.0],
    }


def test_validate_obs_data_with_csv_path(vandyke_csv_path):
    """Test validate_obs_data with CSV file path as string."""
    result = BaseModel.validate_obs_data(vandyke_csv_path)
    assert result is True


def test_validate_obs_data_with_dataframe(vandyke_df):
    """Test validate_obs_data with pandas DataFrame."""
    result = BaseModel.validate_obs_data(vandyke_df)
    assert result is True


def test_validate_obs_data_with_invalid_dataframe(
    vandyke_df_invalid,
):
    """Test validate_obs_data raises KeyError with missing columns."""
    with pytest.raises(KeyError):
        BaseModel.validate_obs_data(vandyke_df_invalid)


def test_transform_obs_data(vandyke_df):
    """Test transform_obs_data with valid pandas DataFrame"""
    result = BaseModel.transform_obs_data(
        vandyke_df, rating_threshold=3
    )
    assert all(
        [
            col in result.columns
            for col in ["reader", "treatment", "k"]
        ]
    )

    with pytest.raises(ValueError):
        BaseModel.transform_obs_data(
            vandyke_df,
            rating_threshold=vandyke_df.rating.max() + 1,
        )


def test_priors_setter_string_success(vandyke_df):
    df = vandyke_df
    m = BaseModel(obs_data=df)
    assert isinstance(m.priors, dict)
    keys = set(m.priors.keys())
    assert {
        "a_mu",
        "a_sigma",
        "b_mu",
        "b_sigma",
        "gamma_mu",
    }.issubset(keys)


def test_priors_setter_string_invalid_raises(vandyke_df):
    df = vandyke_df
    with pytest.raises(NotImplementedError):
        m = BaseModel(obs_data=df, priors="not-a-prior")


def test_priors_setter_dict_success(
    vandyke_df, custom_priors
):
    df = vandyke_df
    priors = custom_priors
    m = BaseModel(obs_data=df, priors=priors)
    assert isinstance(m.priors, dict)
    keys = set(m.priors.keys())
    assert "a_mu" in keys
    assert "gamma_sigma" in keys


def test_priors_setter_dict_missing_raises(vandyke_df):
    df = vandyke_df
    priors_incomplete = {"a": [0.0, 1.0], "b": [0.0, 1.0]}
    with pytest.raises(ValueError):
        m = BaseModel(obs_data=df, priors=priors_incomplete)



def test_get_thresholds_from_ratings(vandyke_df):
    df = vandyke_df
    thresholds = get_thresholds_from_ratings(df.rating)
    assert all([val in [0,1, 2, 3, 4] for val in thresholds])
