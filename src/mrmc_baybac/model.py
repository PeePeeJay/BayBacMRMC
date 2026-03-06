import pymc as pm
import pandas as pd
import xarray as xr
import numpy as np
from typing import Optional
import arviz as az
import logging
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import os

from mrmc_baybac.utils import compute_posterior_effect_size, compute_posterior_accuracy_by_treatment, get_thresholds_from_ratings
from .plotting import plot_tpr_tnr_by_threshold, plot_roc_curve_with_hdi

class BaseModel:
    def __init__(
        self,
        obs_data: pd.DataFrame | str,
        priors: Optional[dict] | Optional[str] = "diffuse",
    ):

        self.obs_data = obs_data  # observed data
        self.priors = priors  # model priors
        self.idata = None  # inference data

    @staticmethod
    def _setup_model(obs_data, priors) -> pm.Model:
        # setup coords
        reader, study_readers = obs_data.reader.factorize()
        case, study_cases = obs_data.case.factorize()
        treatment = obs_data.treatment.values

        coords = {"reader": study_readers, "case": study_cases}

        with pm.Model(coords=coords) as model:
            treatment_idx = pm.Data(
                "treatment_idx", treatment, dims="obs_id"
            )
            reader_idx = pm.Data(
                "reader_idx", reader, dims="obs_id"
            )
            case_idx = pm.Data(
                "case_idx", case, dims="obs_id"
            )

            # model definition
            epsilon = 1e-2

            ### population level parameters
            mu_a = pm.Normal(
                "mu_a",
                mu=priors["a_mu"],
                sigma=priors["a_sigma"],
            )
            sigma_a = pm.HalfNormal(
                "sigma_a",
                1,
            )

            mu_b = pm.Normal(
                "mu_b",
                mu=0,
                sigma=1,
            )
            sigma_b = pm.HalfNormal(
                "sigma_b",
                1,
            )

            ### reader level parameters
            # non-centered parameterization for intercepts
            z_a = pm.Normal(
                "z_a", mu=0, sigma=1, dims="reader"
            )
            alpha = pm.Deterministic(
                "alpha", mu_a + z_a * sigma_a, dims="reader"
            )

            # Non-centered random slopes
            z_b = pm.Normal(
                "z_b", mu=0, sigma=1, dims="reader"
            )
            beta = pm.Deterministic(
                "beta", mu_b + z_b * sigma_b, dims="reader"
            )

            # # Reader-case interaction (non-centered)
            # sigma_rc = pm.HalfNormal(
            #     "sigma_rc",
            #     1,
            # )
            # z_rc = pm.Normal(
            #     "z_rc", mu=0, sigma=1, dims=("reader", "case")
            # )
            # reader_case_interaction = pm.Deterministic(
            #     "reader_case_interaction",
            #     z_rc * sigma_rc,
            #     dims=("reader", "case"),
            # )

            # overdispersion
            gamma = pm.TruncatedNormal(
                "gamma",
                mu=priors["gamma_mu"],
                sigma=priors["gamma_sigma"],
                lower=0.05,
                upper=0.95,
            )

            # probability of correct classification
            p = pm.math.clip(
                pm.math.invlogit(
                    alpha[reader_idx]
                    + beta[reader_idx] * treatment_idx
                    # + reader_case_interaction[reader_idx, case_idx]
                ),
                epsilon,
                1 - epsilon,
            )
            kappa = (1 - p) / p

            # compute alpha and beta parameters for the beta-binomial likelihood
            a_beta = (1 - gamma) / (gamma * (1 + kappa))
            b_beta = (
                kappa * (1 - gamma) / (gamma * (1 + kappa))
            )

            # likelihood
            y = pm.BetaBinomial(
                "k",
                n=len(obs_data.case.unique()),
                alpha=a_beta,
                beta=b_beta,
                observed=obs_data.k,
                dims="obs_id",
            )
        return model

    @property
    def idata(self):
        if self._idata is None:
            raise AttributeError(
                "Inference data was not yet computed. "
                "Please run inference method to trigger inferece."
            )
        else:
            return self._idata

    @idata.setter
    def idata(self, idata):
        # TODO: validate idata
        self._idata = idata

    @property
    def obs_data(self):
        return self._obs_data

    @obs_data.setter
    def obs_data(self, obs_data):
        logging.info("Check if obs_data is valid.")
        if self.validate_obs_data(obs_data):
            logging.info("obs_data is valid.")

        self._obs_data = obs_data

    @obs_data.getter
    def obs_data(self):
        return self._obs_data

    @property
    def priors(self):
        return self._priors

    @priors.setter
    def priors(self, priors):
        prior_params = self.validate_priors(priors)
        self._priors = prior_params

    @staticmethod
    def validate_obs_data(obs_data):
        if isinstance(obs_data, str):
            logging.info(
                f"{obs_data} is a string. "
                "Check if it's a valid path to a csv"
            )
            try:
                obs_data = pd.read_csv(obs_data)
            except Exception as e:
                logging.error(
                    f"Could not read {obs_data}. " "{e}"
                )
        elif isinstance(obs_data, pd.DataFrame):
            logging.info(
                f"Got {type(obs_data)}. Check if all columns present."
            )
        else:
            raise TypeError(
                f"{obs_data} must be instance of {str} or {pd.DataFrame}"
            )

        required_columns = [
            "reader",
            "case",
            "truth",
            "treatment",
        ]
        missing_columns = [
            col
            for col in required_columns
            if col not in obs_data.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Missing columns in obs_data: {missing_columns}."
                "Please keep the format as described in the MRMCaov R package. "
                "See: https://brian-j-smith.github.io/MRMCaov/using.html for more information."
            )
        else:
            logging.info("All columns present")
        return True

    @staticmethod
    def transform_obs_data(
        obs_data: pd.DataFrame, rating_threshold
    ) -> pd.DataFrame:
        df = obs_data.copy()
        if rating_threshold < 0 or rating_threshold > df.rating.max():
            raise ValueError(
                f"Specified rating_threshold {rating_threshold}"
                " is not a valid rating value."
            )
        logging.info(
            f"Binarize rating data with rating threshold {rating_threshold}"
        )
        df["rating_binary"] = df["rating"].copy().apply(
            lambda x: (
                int(0) if x <= rating_threshold else int(1)
            )
        )

        # Create a boolean column for correct predictions
        df["correct"] = df["rating_binary"] == df["truth"]

        # Group by reader, case, and treatment to preserve case information for reader-case interactions
        result = (
            df.groupby(["reader", "case", "treatment"])["correct"]
            .sum()
            .reset_index()
        )
        result.rename(
            columns={"correct": "k"}, inplace=True
        )

        # finally map the treatment column to 0 and 1 for control and treatment group
        # find the two unique levels, sorted
        levels = sorted(df['treatment'].unique())

        # make a mapping dict: first level →0, second →1
        mapping = {levels[0]: 0, levels[1]: 1}

        # apply it in‑place or on a copy
        result['treatment'] = result['treatment'].copy().map(mapping)

        return result

    @staticmethod
    def validate_priors(priors: dict | str) -> dict:
        required_priors = ["a", "b", "gamma"]

        coords = {
            "priors": [
                "diffuse",
                "weakly informative",
                "informative",
            ],
            "params": ["mu", "sigma"],
        }

        default_prior_settings = xr.Dataset(
            {
                "a": (
                    ["priors", "params"],
                    [
                        [np.log(1), 2],
                        [np.log(75 / 25), 1],
                        [np.log(75 / 25), 1],
                    ],
                ),
                "b": (
                    ["priors", "params"],
                    [[0, 2], [0, 1], [0.2, 0.5]],
                ),
                "gamma": (
                    ["priors", "params"],
                    [[0, 10**2], [0, 1], [0, 0.5]],
                ),
            },
            coords=coords,
        )

        if isinstance(priors, str):
            logging.info(
                "Got str as priors argument. Check if specified prior setting is implemented."
            )
            if priors in coords["priors"]:
                logging.info(
                    f"Set priors to default {priors} prior setting."
                )
                prior_params = {
                    "a_mu": default_prior_settings["a"]
                    .sel(priors=priors, params="mu")
                    .values,
                    "a_sigma": default_prior_settings["a"]
                    .sel(priors=priors, params="sigma")
                    .values,
                    "b_mu": default_prior_settings["b"]
                    .sel(priors=priors, params="mu")
                    .values,
                    "b_sigma": default_prior_settings["b"]
                    .sel(priors=priors, params="sigma")
                    .values,
                    "gamma_mu": default_prior_settings[
                        "gamma"
                    ]
                    .sel(priors=priors, params="mu")
                    .values,
                    "gamma_sigma": default_prior_settings[
                        "gamma"
                    ]
                    .sel(priors=priors, params="sigma")
                    .values,
                }
            else:
                raise NotImplementedError(
                    f"Specified prior setting {priors} is not implemented."
                    f"Please choose one of {coords['priors']}"
                )
        elif isinstance(priors, dict):
            logging.info(
                "Got dict as priors argument. Check if all required priors are specified."
            )
            missing_priors = [
                value
                for value in required_priors
                if value not in priors.keys()
            ]
            if len(missing_priors) > 0:
                raise ValueError(
                    f"Missing prior specification in: {missing_priors}"
                )
            incomplete_priors = [
                key
                for key in priors.keys()
                if not (
                    len(priors[key]) == 2
                    and all(
                        [
                            isinstance(
                                priors[key][i], float
                            )
                            for i in range(2)
                        ]
                    )
                )
            ]
            if len(incomplete_priors) > 0:
                raise ValueError(
                    f"Prior definition incomplete in the following priors: {incomplete_priors}. "
                    "Please specify priors {'prior_parameter': [mu, sigma], }."
                )

            prior_params = {
                "a_mu": priors["a"][0],
                "a_sigma": priors["a"][1],
                "b_mu": priors["b"][0],
                "b_sigma": priors["b"][1],
                "gamma_mu": priors["gamma"][0],
                "gamma_sigma": priors["gamma"][1],
            }
        return prior_params

    def _run_inference(self, obs_data, rating_threshold):
        data = self.transform_obs_data(
            obs_data.copy(),
            rating_threshold,
        )
        # data = obs_data.copy()
        model = self._setup_model(
            data, self.priors,
        )

        with model:
            idata = pm.sample(draws=4000)
            pm.sample_posterior_predictive(
                idata, extend_inferencedata=True
            )
            
        return idata, model

    def summary(
        idata: az.data.inference_data.InferenceData,
        **kwargs,
    ):
        print(
            pm.summary(
                idata,
                var_names=[
                    "mu_a",
                    "sigma_a",
                    "mu_b",
                    "sigma_b",
                ],
                kind=kwargs.get("kind", "stats"),
            )
        )


class BalancedModel(BaseModel):
    def __init__(
        self,
        obs_data: pd.DataFrame | str,
        priors: Optional[dict] | Optional[str] = "diffuse",
        ):
        super().__init__(obs_data, priors)
        self.roc_results = None #TODO: refactor as property 
    
    def _run_inference(self, rating_threshold):
        negative_data = self.obs_data[self.obs_data.truth == 0].copy()
        positive_data = self.obs_data[self.obs_data.truth == 1].copy()
        
        # run inference for negative cases and positive cases seperately 
        idatas = []
        for data in [negative_data, positive_data]:
            idata, model = super()._run_inference(
                obs_data=data,
                rating_threshold=rating_threshold
                )
            idatas.append(idata)
        return idatas

    def _compute_tpr_tnr(self, threshold):
        """Compute TPR and TNR for a given threshold.
        
        Args:
            threshold: Rating threshold for binarization
            
        Returns:
            tuple: (tpr_dict, tnr_dict) where each dict has keys "0" and "1" for treatment settings.
                   Each value is a posterior sample array.
        """
        idatas = self._run_inference(threshold)
        
        # idatas[0] is for negative cases (truth==0), idatas[1] is for positive cases (truth==1)
        # Compute TNR (accuracy for negative cases) - posterior samples
        neg_idata = idatas[0]
        neg_a_mu = neg_idata["posterior"]["mu_a"].values
        neg_b_mu = neg_idata["posterior"]["mu_b"].values
        neg_acc_0, neg_acc_1 = compute_posterior_accuracy_by_treatment(neg_a_mu, neg_b_mu)
        
        # Compute TPR (accuracy for positive cases) - posterior samples
        pos_idata = idatas[1]
        pos_a_mu = pos_idata["posterior"]["mu_a"].values
        pos_b_mu = pos_idata["posterior"]["mu_b"].values
        pos_acc_0, pos_acc_1 = compute_posterior_accuracy_by_treatment(pos_a_mu, pos_b_mu)
        
        tnr_dict = {"0": neg_acc_0, "1": neg_acc_1}
        tpr_dict = {"0": pos_acc_0, "1": pos_acc_1}
        return tpr_dict, tnr_dict

    def roc_curve_analysis(self):
        """Perform ROC curve analysis across multiple thresholds.
        
        Returns:
            dict: Contains ROC curve coordinates and AUC for each treatment setting:
                {
                    "0": {"fpr": [...], "tpr": [...], "auc": float, "partial_auc": float, "partial_fpr_range": (min_fpr, max_fpr)},
                    "1": {"fpr": [...], "tpr": [...], "auc": float, "partial_auc": float, "partial_fpr_range": (min_fpr, max_fpr)}
                }
        """
        thresholds = get_thresholds_from_ratings(self.obs_data.rating)
        tprs, tnrs = {"0": [], "1": []}, {"0": [], "1": []}
        
        for threshold in thresholds:
            try:
                tpr_dict, tnr_dict = self._compute_tpr_tnr(threshold)
            except Exception as e:
                print(e) 
                continue     

            tprs["0"].append(tpr_dict["0"])
            tprs["1"].append(tpr_dict["1"])
            tnrs["0"].append(tnr_dict["0"])
            tnrs["1"].append(tnr_dict["1"])
        
        # Compute ROC curve and AUC
        roc_results = self._compute_roc_auc(tprs, tnrs)
        self.roc_results = roc_results # Store results as instance variable

        return roc_results

    def _compute_roc_auc(self, tprs, tnrs):
        """Compute ROC curve coordinates and AUC for each treatment setting.
        
        Args:
            tprs: dict with keys "0" and "1" containing lists of TPR values per threshold
            tnrs: dict with keys "0" and "1" containing lists of TNR values per threshold
            
        Returns:
            dict: Contains ROC curve coordinates and AUC for each treatment setting:
                {
                    "0": {"fpr": [...], "tpr": [...], "auc": float, "partial_auc": float},
                    "1": {"fpr": [...], "tpr": [...], "auc": float, "partial_auc": float}
                }
        """
        roc_results = {}
        
        # First compute individual ROC curves
        individual_results = {}
        for setting in ["0", "1"]:
            # Convert posterior samples to mean values
            tnr_values = [np.mean(tnr) if isinstance(tnr, np.ndarray) else tnr for tnr in tnrs[setting]]
            tpr_values = [np.mean(tpr) if isinstance(tpr, np.ndarray) else tpr for tpr in tprs[setting]]
            
            # Compute FPR from TNR: FPR = 1 - TNR
            fpr = [1 - tnr for tnr in tnr_values]
            tpr = tpr_values
        
            # Sort by FPR for proper ROC curve
            sorted_pairs = sorted(zip(fpr, tpr), key=lambda x: x[0])
            fpr_sorted = [pair[0] for pair in sorted_pairs]
            tpr_sorted = [pair[1] for pair in sorted_pairs]
            
            # Compute AUC
            auc_score = auc(fpr_sorted, tpr_sorted)
            
            individual_results[setting] = {
                "fpr": fpr_sorted,
                "tpr": tpr_sorted,
                "auc": auc_score
            }
        
        # Compute partial AUC for overlapping FPR range
        fpr_0 = np.array(individual_results["0"]["fpr"])
        fpr_1 = np.array(individual_results["1"]["fpr"])
        
        # Attempt to compute partial range from discrete intersection of observed points
        common = []
        for val in fpr_0:
            if np.any(np.isclose(val, fpr_1, atol=1e-6)):
                common.append(val)
        for val in fpr_1:
            if np.any(np.isclose(val, fpr_0, atol=1e-6)):
                common.append(val)
        if len(common) >= 2:
            fpr_min = float(np.min(common))
            fpr_max = float(np.max(common))
        else:
            # fall back to numeric interval intersection
            fpr_min = max(float(np.min(fpr_0)), float(np.min(fpr_1)))
            fpr_max = min(float(np.max(fpr_0)), float(np.max(fpr_1)))
        
        for setting in ["0", "1"]:
            fpr_vals = individual_results[setting]["fpr"]
            tpr_vals = individual_results[setting]["tpr"]
            
            # Filter points within overlapping FPR range
            overlapping_indices = [i for i, fpr_val in enumerate(fpr_vals) 
                                 if fpr_min <= fpr_val <= fpr_max]
            
            if len(overlapping_indices) >= 2:
                # Extract overlapping FPR and TPR values
                fpr_partial = [fpr_vals[i] for i in overlapping_indices]
                tpr_partial = [tpr_vals[i] for i in overlapping_indices]
                
                # Compute partial AUC
                partial_auc = auc(fpr_partial, tpr_partial)
            else:
                # If insufficient overlapping points, use full AUC as fallback
                partial_auc = individual_results[setting]["auc"]
            
            # Combine results
            roc_results[setting] = {
                **individual_results[setting],
                "partial_auc": partial_auc,
                "partial_fpr_range": (fpr_min, fpr_max)
            }
        
        return roc_results

    def plot_tpr_tnr_by_threshold(self, filename: str = "figures/tpr_tnr_by_threshold.png"):
        """Generate and save TPR/TNR plot with 95% HDI across thresholds.

        Args:
            filename: path where the figure will be saved. The directory
                portion of the path will be created if necessary.

        Returns:
            str: path to the saved figure file.
        """
        return plot_tpr_tnr_by_threshold(self, filename)

    def plot_roc_curve_with_hdi(self, filename: str = "figures/roc_curve_with_hdi.png"):
        """Generate and save ROC curve plot with 95% HDI band and partial AUC uncertainty.
        
        This function builds ROC curves by varying classification thresholds across the dataset,
        computing posterior uncertainty for both the curve and partial AUC metric within the
        overlapping FPR range between treatment settings.

        Args:
            filename: path where the figure will be saved. The directory
                portion of the path will be created if necessary.

        Returns:
            str: path to the saved figure file.
        """
        return plot_roc_curve_with_hdi(self, filename)
    
