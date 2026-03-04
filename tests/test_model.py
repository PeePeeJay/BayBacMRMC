import pytest
import pandas as pd
from pathlib import Path
import os
from mrmc_baybac.model import BaseModel, BalancedModel


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

    assert len(result) == len(
        vandyke_df.reader.unique()
    ) * len(vandyke_df.treatment.unique())

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


def test__run_inference_with_default_priors(vandyke_df):
    obs_data = vandyke_df
    rating_threshold = 3
    m = BaseModel(obs_data=obs_data)

    idata, model = m._run_inference(
        obs_data=obs_data,
        rating_threshold=rating_threshold
    )
    assert "posterior" in idata.keys()
    assert "posterior_predictive" in idata.keys()


def test_get_thresholds_from_ratings(vandyke_df):
    df = vandyke_df
    tresholds = BalancedModel.get_thresholds_from_ratings(df.rating)
    assert all([val in [0,1, 2, 3, 4] for val in tresholds])


def test_balanced_model_run_inference(vandyke_df):
    """Test BalancedModel._run_inference returns inference data for both truth values."""
    obs_data = vandyke_df
    rating_threshold = 3
    
    balanced_model = BalancedModel(obs_data=obs_data)
    idatas = balanced_model._run_inference(rating_threshold=rating_threshold)
    
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
        
        # Validate FPR and TPR are lists
        assert isinstance(roc_results[setting]["fpr"], list)
        assert isinstance(roc_results[setting]["tpr"], list)
        
        # Validate AUC is a float between 0 and 1
        assert isinstance(roc_results[setting]["auc"], float)
        assert 0 <= roc_results[setting]["auc"] <= 1
        
        # Validate FPR and TPR have the same length
        assert len(roc_results[setting]["fpr"]) == len(roc_results[setting]["tpr"])

def test_roc_curve_analysis_cxr_mrmc(cxr_df):
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
        
        # Validate FPR and TPR are lists
        assert isinstance(roc_results[setting]["fpr"], list)
        assert isinstance(roc_results[setting]["tpr"], list)
        
        # Validate AUC is a float between 0 and 1
        assert isinstance(roc_results[setting]["auc"], float)
        assert 0 <= roc_results[setting]["auc"] <= 1
        
        # Validate FPR and TPR have the same length
        assert len(roc_results[setting]["fpr"]) == len(roc_results[setting]["tpr"])


def test_plot_tpr_tnr_by_threshold_creates_file(vandyke_df, tmp_path):
    """BalancedModel.plot_tpr_tnr_by_threshold should save a figure file."""
    balanced_model = BalancedModel(obs_data=vandyke_df)
    out_file =  "./tpr_tnr_by_threshold.png"
    path = balanced_model.plot_tpr_tnr_by_threshold(filename=str(out_file))
    assert os.path.isfile(path)
    # optional: ensure extension matches
    assert path.endswith(".png")

def test_plot_tpr_tnr_by_threshold_cxr_data(cxr_df, tmp_path):
    """BalancedModel.plot_tpr_tnr_by_threshold should save a figure file."""
    balanced_model = BalancedModel(obs_data=cxr_df)
    out_file =  "./tpr_tnr_by_threshold_cxr.png"
    path = balanced_model.plot_tpr_tnr_by_threshold(filename=str(out_file))
    assert os.path.isfile(path)
    # optional: ensure extension matches
    assert path.endswith(".png")

def test_plot_roc_curve_with_hdi_creates_file(vandyke_df, tmp_path):
    """BalancedModel.plot_roc_curve_with_hdi should save a figure file."""
    balanced_model = BalancedModel(obs_data=vandyke_df)
    out_file = "./roc_curve_with_hdi.png"
    path = balanced_model.plot_roc_curve_with_hdi(filename=str(out_file))
    assert os.path.isfile(path)
    # optional: ensure extension matches
    assert path.endswith(".png")

def test_plot_roc_curve_with_hdi_cxr_data(cxr_df, tmp_path):
    """BalancedModel.plot_roc_curve_with_hdi should save a figure file with CXR data."""
    balanced_model = BalancedModel(obs_data=cxr_df)
    out_file = "./roc_curve_with_hdi_cxr.png"
    path = balanced_model.plot_roc_curve_with_hdi(filename=str(out_file))
    assert os.path.isfile(path)
    # optional: ensure extension matches
    assert path.endswith(".png")




