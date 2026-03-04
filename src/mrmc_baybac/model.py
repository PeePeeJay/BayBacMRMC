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

from mrmc_baybac.utils import compute_posterior_effect_size, compute_posterior_accuracy_by_treatment

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

            # Reader-case interaction (non-centered)
            sigma_rc = pm.HalfNormal(
                "sigma_rc",
                1,
            )
            z_rc = pm.Normal(
                "z_rc", mu=0, sigma=1, dims=("reader", "case")
            )
            reader_case_interaction = pm.Deterministic(
                "reader_case_interaction",
                z_rc * sigma_rc,
                dims=("reader", "case"),
            )

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
                    + reader_case_interaction[reader_idx, case_idx]
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
                n=1,
                alpha=a_beta,
                beta=b_beta,
                # observed=obs_data.k,
                observed=obs_data.truth,
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
        # data = self.transform_obs_data(
        #     obs_data.copy(),
        #     rating_threshold,
        # )
        # n_cases = len(obs_data.case.unique())
        data = obs_data.copy()
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
                    "0": {"fpr": [...], "tpr": [...], "auc": float},
                    "1": {"fpr": [...], "tpr": [...], "auc": float}
                }
        """
        thresholds = self.get_thresholds_from_ratings(self.obs_data.rating)
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
        
        return roc_results

    def _compute_roc_auc(self, tprs, tnrs):
        """Compute ROC curve coordinates and AUC for each treatment setting.
        
        Args:
            tprs: dict with keys "0" and "1" containing lists of TPR values per threshold
            tnrs: dict with keys "0" and "1" containing lists of TNR values per threshold
            
        Returns:
            dict: Contains ROC curve coordinates and AUC for each treatment setting:
                {
                    "0": {"fpr": [...], "tpr": [...], "auc": float},
                    "1": {"fpr": [...], "tpr": [...], "auc": float}
                }
        """
        roc_results = {}
        
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
            
            roc_results[setting] = {
                "fpr": fpr_sorted,
                "tpr": tpr_sorted,
                "auc": auc_score
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
        thresholds = self.get_thresholds_from_ratings(self.obs_data.rating)
        # Sort thresholds in ascending order
        thresholds = np.sort(thresholds)
        
        tprs, tnrs = {"0": [], "1": []}, {"0": [], "1": []}
        
        # Collect posterior samples for all thresholds in sorted order
        for threshold in thresholds:
            tpr_dict, tnr_dict = self._compute_tpr_tnr(threshold)
            tprs["0"].append(tpr_dict["0"])
            tprs["1"].append(tpr_dict["1"])
            tnrs["0"].append(tnr_dict["0"])
            tnrs["1"].append(tnr_dict["1"])
        
        # Compute mean and HDI for each threshold and setting
        metrics = {}
        for setting in ["0", "1"]:
            means_tpr = [np.mean(s) for s in tprs[setting]]
            means_tnr = [np.mean(s) for s in tnrs[setting]]
            
            # Compute 95% HDI (flatten posterior samples first)
            hdi_tpr = [az.hdi(s.flatten(), hdi_prob=0.95) for s in tprs[setting]]
            hdi_tnr = [az.hdi(s.flatten(), hdi_prob=0.95) for s in tnrs[setting]]
            
            # Extract lower and upper bounds as scalars
            tpr_lower = [float(hdi[0]) for hdi in hdi_tpr]
            tpr_upper = [float(hdi[1]) for hdi in hdi_tpr]
            tnr_lower = [float(hdi[0]) for hdi in hdi_tnr]
            tnr_upper = [float(hdi[1]) for hdi in hdi_tnr]
            
            metrics[setting] = {
                "thresholds": thresholds,
                "tpr_mean": means_tpr,
                "tpr_lower": tpr_lower,
                "tpr_upper": tpr_upper,
                "tnr_mean": means_tnr,
                "tnr_lower": tnr_lower,
                "tnr_upper": tnr_upper,
            }
        
        # Ensure output directory exists
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        # Create figure with TPR and TNR subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for setting in ["0", "1"]:
            m = metrics[setting]
            thresholds_x = range(len(m["thresholds"]))
            
            # TPR subplot
            axes[0].plot(thresholds_x, m["tpr_mean"], marker="o", label=f"Setting {setting}")
            axes[0].fill_between(
                thresholds_x,
                m["tpr_lower"],
                m["tpr_upper"],
                alpha=0.2
            )
            
            # TNR subplot
            axes[1].plot(thresholds_x, m["tnr_mean"], marker="o", label=f"Setting {setting}")
            axes[1].fill_between(
                thresholds_x,
                m["tnr_lower"],
                m["tnr_upper"],
                alpha=0.2
            )
        
        # Configure TPR subplot
        axes[0].set_xlabel("Threshold Index")
        axes[0].set_ylabel("TPR (True Positive Rate)")
        axes[0].set_title("TPR by Threshold (with 95% HDI)")
        axes[0].set_xticks(range(len(thresholds)))
        axes[0].set_xticklabels([f"{t:.1f}" for t in thresholds], rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Configure TNR subplot
        axes[1].set_xlabel("Threshold Index")
        axes[1].set_ylabel("TNR (True Negative Rate)")
        axes[1].set_title("TNR by Threshold (with 95% HDI)")
        axes[1].set_xticks(range(len(thresholds)))
        axes[1].set_xticklabels([f"{t:.1f}" for t in thresholds], rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(filename, dpi=100)
        plt.close()
        return filename

    def plot_roc_curve_with_hdi(self, filename: str = "figures/roc_curve_with_hdi.png"):
        """Generate and save ROC curve plot with 95% HDI band and AUC uncertainty.
        
        This function builds ROC curves by varying classification thresholds across the dataset,
        computing posterior uncertainty for both the curve and AUC metric.

        Args:
            filename: path where the figure will be saved. The directory
                portion of the path will be created if necessary.

        Returns:
            str: path to the saved figure file.
        """
        thresholds = self.get_thresholds_from_ratings(self.obs_data.rating)
        thresholds = np.sort(thresholds)
        
        # Collect TPR and TNR posterior samples across all thresholds
        all_tprs = {"0": [], "1": []}
        all_tnrs = {"0": [], "1": []}
        
        for threshold in thresholds:
            try:
                tpr_dict, tnr_dict = self._compute_tpr_tnr(threshold)
                for setting in ["0", "1"]:
                    all_tprs[setting].append(tpr_dict[setting])
                    all_tnrs[setting].append(tnr_dict[setting])
            except Exception as e:
                print(f"Error at threshold {threshold}: {e}")
                continue
        
        # Ensure output directory exists
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        # Create figure with subplots for each treatment setting
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, setting in enumerate(["0", "1"]):
            ax = axes[idx]
            
            # Stack posterior samples to get shape: (n_thresholds, n_posterior_samples)
            tprs_stacked = []
            tnrs_stacked = []
            
            for tpr, tnr in zip(all_tprs[setting], all_tnrs[setting]):
                if isinstance(tpr, np.ndarray):
                    tprs_stacked.append(tpr.flatten())
                else:
                    tprs_stacked.append(np.array(tpr).flatten())
                
                if isinstance(tnr, np.ndarray):
                    tnrs_stacked.append(tnr.flatten())
                else:
                    tnrs_stacked.append(np.array(tnr).flatten())
            
            # Ensure all have same length (number of posterior samples)
            n_posteriors = max([len(t) for t in tprs_stacked])
            
            tprs_array = np.array([
                np.pad(t, (0, n_posteriors - len(t))) for t in tprs_stacked
            ])
            tnrs_array = np.array([
                np.pad(t, (0, n_posteriors - len(t))) for t in tnrs_stacked
            ])
            
            # Compute FPR for each threshold and posterior sample (FPR = 1 - TNR)
            fprs_array = 1 - tnrs_array
            
            # For each posterior sample, compute ROC curve and AUC
            auc_samples = []
            
            for post_idx in range(n_posteriors):
                fpr_curve = fprs_array[:, post_idx]
                tpr_curve = tprs_array[:, post_idx]
                
                # Sort by FPR for proper ROC curve
                sorted_indices = np.argsort(fpr_curve)
                fpr_sorted = fpr_curve[sorted_indices]
                tpr_sorted = tpr_curve[sorted_indices]
                
                # Add boundary points (0,0) if needed
                if fpr_sorted[0] > 0 or tpr_sorted[0] > 0:
                    fpr_sorted = np.concatenate([[0], fpr_sorted])
                    tpr_sorted = np.concatenate([[0], tpr_sorted])
                
                # Add boundary point (1,1) if needed
                if fpr_sorted[-1] < 1 or tpr_sorted[-1] < 1:
                    fpr_sorted = np.concatenate([fpr_sorted, [1]])
                    tpr_sorted = np.concatenate([tpr_sorted, [1]])
                
                # Compute AUC using trapezoidal rule
                auc_val = auc(fpr_sorted, tpr_sorted)
                auc_samples.append(auc_val)
            
            # Compute AUC statistics
            auc_samples = np.array(auc_samples)
            auc_mean = np.mean(auc_samples)
            auc_hdi = az.hdi(auc_samples, hdi_prob=0.95)
            
            # Compute normalized AUC: normalize by the maximum possible AUC in the observed FPR range
            # Maximum AUC occurs when TPR = 1 for all FPR points
            fpr_min = np.min(fprs_array)
            fpr_max = np.max(fprs_array)
            max_possible_auc = fpr_max - fpr_min  # Area of rectangle with height 1
            normalized_auc_samples = auc_samples / max_possible_auc if max_possible_auc > 0 else auc_samples
            normalized_auc_mean = np.mean(normalized_auc_samples)
            normalized_auc_hdi = az.hdi(normalized_auc_samples, hdi_prob=0.95)
            
            # Compute mean ROC curve by averaging FPR and TPR across posterior samples
            mean_fpr = np.mean(fprs_array, axis=1)
            mean_tpr = np.mean(tprs_array, axis=1)
            
            # Sort mean curve by FPR
            sorted_indices = np.argsort(mean_fpr)
            mean_fpr_sorted = mean_fpr[sorted_indices]
            mean_tpr_sorted = mean_tpr[sorted_indices]
            
            # Add boundary points (0,0) and (1,1) to the mean curve if needed
            if mean_fpr_sorted[0] > 0 or mean_tpr_sorted[0] > 0:
                mean_fpr_sorted = np.concatenate([[0], mean_fpr_sorted])
                mean_tpr_sorted = np.concatenate([[0], mean_tpr_sorted])
            
            if mean_fpr_sorted[-1] < 1 or mean_tpr_sorted[-1] < 1:
                mean_fpr_sorted = np.concatenate([mean_fpr_sorted, [1]])
                mean_tpr_sorted = np.concatenate([mean_tpr_sorted, [1]])
            
            # Plot mean ROC curve
            ax.plot(mean_fpr_sorted, mean_tpr_sorted, 'b-', linewidth=2.5, label="Mean ROC")
            
            # Compute 95% HDI band using interpolation
            # Collect all FPR and TPR samples for each threshold
            fpr_tpr_by_threshold = []
            for threshold_idx in range(len(thresholds)):
                fpr_samples = fprs_array[threshold_idx, :]
                tpr_samples = tprs_array[threshold_idx, :]
                for fpr, tpr in zip(fpr_samples, tpr_samples):
                    fpr_tpr_by_threshold.append((fpr, tpr))
            
            # Create a dense grid of FPR values from 0 to 1
            fpr_grid = np.linspace(0, 1, 100)
            tpr_lower_interp = []
            tpr_upper_interp = []
            
            # For each FPR grid point, find which thresholds give FPR values nearby
            # and interpolate the TPR bounds
            for fpr_target in fpr_grid:
                tpr_values_at_fpr = []
                
                # Find all threshold/posterior combinations with FPR close to target
                tolerance = 0.05
                for fpr_val, tpr_val in fpr_tpr_by_threshold:
                    if abs(fpr_val - fpr_target) < tolerance:
                        tpr_values_at_fpr.append(tpr_val)
                
                # If we have TPR values at this FPR, compute HDI
                if len(tpr_values_at_fpr) > 0:
                    tpr_hdi = az.hdi(np.array(tpr_values_at_fpr), hdi_prob=0.95)
                    tpr_lower_interp.append(tpr_hdi[0])
                    tpr_upper_interp.append(tpr_hdi[1])
                else:
                    # If no samples available, use the mean curve value or skip
                    if len(tpr_lower_interp) > 0:
                        tpr_lower_interp.append(tpr_lower_interp[-1])
                        tpr_upper_interp.append(tpr_upper_interp[-1])
            
            # Fill between HDI bounds if we have interpolated values
            if len(tpr_lower_interp) > 0:
                # Remove any NaN values
                valid_idx = ~np.isnan(tpr_lower_interp) & ~np.isnan(tpr_upper_interp)
                fpr_grid_valid = fpr_grid[valid_idx]
                tpr_lower_valid = np.array(tpr_lower_interp)[valid_idx]
                tpr_upper_valid = np.array(tpr_upper_interp)[valid_idx]
                
                ax.fill_between(fpr_grid_valid, tpr_lower_valid, tpr_upper_valid, 
                               alpha=0.25, color='blue', label="95% HDI")
            
            # Add diagonal reference line for random classifier
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="Random Classifier")
            
            # Add AUC statistics text box
            ax.text(0.6, 0.15, 
                   f"AUC = {auc_mean:.3f}\n95% HDI: [{auc_hdi[0]:.3f}, {auc_hdi[1]:.3f}]\n" +
                   f"Normalized AUC = {normalized_auc_mean:.3f}\n95% HDI: [{normalized_auc_hdi[0]:.3f}, {normalized_auc_hdi[1]:.3f}]",
                   fontsize=11, 
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
            
            # Configure subplot
            ax.set_xlabel("FPR (1 - TNR)", fontsize=11)
            ax.set_ylabel("TPR", fontsize=11)
            ax.set_title(f"ROC Curve - Treatment {setting}\n(with 95% HDI)", fontsize=12)
            ax.legend(fontsize=10, loc="lower right")
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.set_aspect('equal')
        
        fig.tight_layout()
        fig.savefig(filename, dpi=100)
        plt.close()
        return filename

    @staticmethod
    def get_thresholds_from_ratings(ratings, min_rating=0, max_rating=None):
        if not all([val >= 0 for val in ratings]):
            raise ValueError("Ratings must be non-negative. Please transform ratings.")

        if all([val % 1 == 0 for val in ratings]) and (np.max(ratings) <= 10):
            thresholds = np.arange(
                min_rating, np.max(ratings) if max_rating is None else max_rating)
        elif all([val <= 1 for val in ratings]):
            thresholds = np.arange(0.0, 1, 0.1)
        else: 
            thresholds = ratings.unique()
        return thresholds
    
