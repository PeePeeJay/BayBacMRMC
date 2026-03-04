import pytest
import pandas as pd


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
    """Provides the cxr_bsi_mrmc.csv data as a pandas DataFrame."""
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
    """Provides a custom priors dictionary for testing."""
    return {
        "a": [0.0, 1.0],
        "b": [0.0, 1.0],
        "gamma": [0.0, 1.0],
    }
