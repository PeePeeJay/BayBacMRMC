import os
from mrmc_baybac.model import BaseModel, BalancedModel

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
    out_file =  "./tests/.figures/tpr_tnr_by_threshold.png"
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
    out_file = "./tests/.figures/roc_curve_with_hdi_cxr.png"
    path = balanced_model.plot_roc_curve_with_hdi(filename=str(out_file))
    assert os.path.isfile(path)
    # optional: ensure extension matches
    assert path.endswith(".png")

def test_plot_roc_curve_with_hdi_cxr_data_weakly_inf_priors(cxr_df, tmp_path):
    """BalancedModel.plot_roc_curve_with_hdi should save a figure file with CXR data."""
    balanced_model = BalancedModel(obs_data=cxr_df, priors="weakly informative")
    out_file = "./tests/.figures/roc_curve_with_hdi_cxr.png"
    path = balanced_model.plot_roc_curve_with_hdi(filename=str(out_file))
    assert os.path.isfile(path)
    # optional: ensure extension matches
    assert path.endswith(".png")





