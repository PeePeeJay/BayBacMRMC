import pymc as pm
import pandas as pd
import xarray as xr
import numpy as np
from typing import Optional
import arviz as az
import logging
import sys
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import os

from mrmc_baybac.utils import (
    compute_posterior_effect_size,
    compute_posterior_accuracy_by_treatment,
    get_thresholds_from_ratings,
)
from mrmc_baybac.plotting import (
    plot_tpr_fpr_by_threshold,
    plot_roc_curve_with_hdi,
)


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
    def _sampling_kwargs() -> dict:
        """Return PyMC sampling kwargs that are safe under a debugger.

        PyMC runs chains in parallel processes by default. Debuggers often do not
        follow those worker processes unless explicitly configured, so breakpoints
        may appear to be skipped. When a debugger is attached, force single-process
        sampling to keep execution in the current debugged process.
        """
        if sys.gettrace() is not None:
            return {
                "chains": 1,
                "cores": 1,
                "progressbar": False,
            }
        return {}

    @staticmethod
    def _validate_predictive_target(
        predictive_target: str,
    ) -> str:
        valid_targets = {"observed_panel", "new_cases"}
        if predictive_target not in valid_targets:
            raise ValueError(
                "predictive_target must be one of "
                f"{sorted(valid_targets)}, got {predictive_target!r}."
            )
        return predictive_target

    @staticmethod
    def _flatten_chain_draws(values: np.ndarray):
        n_chains, n_draws = values.shape[:2]
        return (
            values.reshape(
                n_chains * n_draws, *values.shape[2:]
            ),
            n_chains,
            n_draws,
        )

    @staticmethod
    def _invlogit(values: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-values))

    @staticmethod
    def _setup_model(obs_data, priors, n_cases) -> pm.Model:
        # setup coords
        reader, study_readers = obs_data.reader.factorize()
        treatment = obs_data.treatment.values

        coords = {"reader": study_readers}

        with pm.Model(coords=coords) as model:
            treatment_idx = pm.Data(
                "treatment_idx", treatment, dims="obs_id"
            )
            reader_idx = pm.Data(
                "reader_idx", reader, dims="obs_id"
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
                n=n_cases,
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
        if (
            rating_threshold
            < 0
            # or rating_threshold > df.rating.max()
        ):
            raise ValueError(
                f"Specified rating_threshold {rating_threshold}"
                " is not a valid rating value."
            )
        logging.info(
            f"Binarize rating data with rating threshold {rating_threshold}"
        )
        df["rating_binary"] = (
            df["rating"]
            .copy()
            .apply(
                lambda x: (
                    int(0)
                    if x < rating_threshold
                    else int(1)
                )
            )
        )

        # Create a boolean column for correct predictions
        df["correct"] = df["rating_binary"] == df["truth"]

        # Group by reader, case, and treatment to preserve case information for reader-case interactions
        result = (
            df.groupby(["reader", "treatment"])["correct"]
            .sum()
            .reset_index()
        )
        result.rename(
            columns={"correct": "k"}, inplace=True
        )

        # finally map the treatment column to 0 and 1 for control and treatment group
        # find the two unique levels, sorted
        levels = sorted(df["treatment"].unique())

        # make a mapping dict: first level →0, second →1
        mapping = {levels[0]: 0, levels[1]: 1}

        # apply it in‑place or on a copy
        result["treatment"] = (
            result["treatment"].copy().map(mapping)
        )

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

    def run_inference(self, obs_data, rating_threshold):
        n_cases = len(obs_data.case.unique())
        data = self.transform_obs_data(
            obs_data.copy(),
            rating_threshold,
        )
        # data = obs_data.copy()
        model = self._setup_model(
            data, self.priors, n_cases
        )

        with model:
            idata = pm.sample(
                draws=4000,
                **self._sampling_kwargs(),
                # nuts_sampler="blackjax", 
                progressbar=False,
            )
            pm.sample_posterior_predictive(
                idata, extend_inferencedata=True
            )

        logging.info(f'Inference completed. \n {pm.summary(
                idata,
                var_names=[
                    "mu_a",
                    "sigma_a",
                    "mu_b",
                    "sigma_b",
                ],
                
            )}')

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
        self.roc_results = (
            None  # TODO: refactor as property
        )
        self._roc_results_predictive_target = None
        self._roc_results_n_new_cases = None

    def run_inference(self, rating_threshold=0.5):
        negative_data = self.obs_data[
            self.obs_data.truth == 0
        ].copy()
        positive_data = self.obs_data[
            self.obs_data.truth == 1
        ].copy()

        # run inference for negative cases and positive cases seperately
        idatas = []
        for data in [negative_data, positive_data]:
            idata, model = super().run_inference(
                obs_data=data,
                rating_threshold=rating_threshold,
            )
            idatas.append(idata)
        return idatas

    def _simulate_new_case_accuracy(
        self,
        idata,
        treatment: int,
        n_new_cases: int,
        random_seed: int,
    ):
        epsilon = 1e-2
        alpha, n_chains, n_draws = self._flatten_chain_draws(
            idata.posterior["alpha"].values
        )
        beta, _, _ = self._flatten_chain_draws(
            idata.posterior["beta"].values
        )
        gamma, _, _ = self._flatten_chain_draws(
            idata.posterior["gamma"].values
        )
        gamma = gamma.reshape(-1, 1)

        eta = alpha + beta * treatment
        p = np.clip(self._invlogit(eta), epsilon, 1 - epsilon)
        kappa = (1 - p) / p
        a_beta = (1 - gamma) / (gamma * (1 + kappa))
        b_beta = (
            kappa * (1 - gamma) / (gamma * (1 + kappa))
        )

        rng = np.random.default_rng(random_seed)
        q = rng.beta(a_beta, b_beta)
        y = rng.binomial(n_new_cases, q) / n_new_cases
        mean_accuracy = y.mean(axis=1)
        return mean_accuracy.reshape(n_chains, n_draws)

    def _compute_tpr_tnr(
        self,
        threshold,
        predictive_target: str = "observed_panel",
        n_new_cases: int | None = None,
    ):
        """Compute TPR and TNR for a given threshold using posterior predictive samples.

        Args:
            threshold: Rating threshold for binarization
            predictive_target: either ``observed_panel`` for uncertainty
                conditional on the fitted cases or ``new_cases`` for
                posterior predictive uncertainty on unseen cases.
            n_new_cases: number of future cases per truth subset used
                when ``predictive_target='new_cases'``. Defaults to the
                number of observed cases in each truth subset.

        Returns:
            tuple: (tpr_dict, tnr_dict) where each dict has keys "0" and "1" for treatment settings.
                   Each value is a (chain, draw) array of posterior predictive accuracy samples.
        """
        predictive_target = self._validate_predictive_target(
            predictive_target
        )
        idatas = self.run_inference(threshold)

        # idatas[0] is for negative cases (truth==0), idatas[1] is for positive cases (truth==1)
        neg_idata = idatas[0]
        pos_idata = idatas[1]

        # Number of cases used to normalise the BetaBinomial count k
        neg_n_cases = len(
            self.obs_data[self.obs_data.truth == 0].case.unique()
        )
        pos_n_cases = len(
            self.obs_data[self.obs_data.truth == 1].case.unique()
        )

        if predictive_target == "new_cases":
            neg_case_count = n_new_cases or neg_n_cases
            pos_case_count = n_new_cases or pos_n_cases
            tnr_dict = {
                "0": self._simulate_new_case_accuracy(
                    neg_idata,
                    treatment=0,
                    n_new_cases=neg_case_count,
                    random_seed=101,
                ),
                "1": self._simulate_new_case_accuracy(
                    neg_idata,
                    treatment=1,
                    n_new_cases=neg_case_count,
                    random_seed=102,
                ),
            }
            tpr_dict = {
                "0": self._simulate_new_case_accuracy(
                    pos_idata,
                    treatment=0,
                    n_new_cases=pos_case_count,
                    random_seed=201,
                ),
                "1": self._simulate_new_case_accuracy(
                    pos_idata,
                    treatment=1,
                    n_new_cases=pos_case_count,
                    random_seed=202,
                ),
            }
            return tpr_dict, tnr_dict

        # Posterior predictive k: shape (chain, draw, obs_id)
        neg_k_pred = neg_idata.posterior_predictive["k"].values
        pos_k_pred = pos_idata.posterior_predictive["k"].values

        # Treatment index for each obs_id (0 = control, 1 = treatment)
        neg_treatment = neg_idata.constant_data["treatment_idx"].values
        pos_treatment = pos_idata.constant_data["treatment_idx"].values

        # Average k across readers for each treatment, then normalise by n_cases
        tnr_0 = neg_k_pred[..., neg_treatment == 0].mean(axis=-1) / neg_n_cases
        tnr_1 = neg_k_pred[..., neg_treatment == 1].mean(axis=-1) / neg_n_cases
        tpr_0 = pos_k_pred[..., pos_treatment == 0].mean(axis=-1) / pos_n_cases
        tpr_1 = pos_k_pred[..., pos_treatment == 1].mean(axis=-1) / pos_n_cases

        tnr_dict = {"0": tnr_0, "1": tnr_1}
        tpr_dict = {"0": tpr_0, "1": tpr_1}
        return tpr_dict, tnr_dict

    def roc_curve_analysis(
        self,
        predictive_target: str = "observed_panel",
        n_new_cases: int | None = None,
    ):
        """Perform ROC curve analysis across multiple thresholds.

        Returns:
            dict: Contains ROC curve coordinates and AUC for each treatment setting:
                {
                    "0": {"fpr": [...], "tpr": [...], "auc": float, "partial_auc": float, "partial_fpr_range": (min_fpr, max_fpr)},
                    "1": {"fpr": [...], "tpr": [...], "auc": float, "partial_auc": float, "partial_fpr_range": (min_fpr, max_fpr)}
                }
        """
        predictive_target = self._validate_predictive_target(
            predictive_target
        )
        thresholds = get_thresholds_from_ratings(
            self.obs_data.rating
        )
        tprs, tnrs = {"0": [], "1": []}, {"0": [], "1": []}

        for threshold in thresholds:
            try:
                tpr_dict, tnr_dict = self._compute_tpr_tnr(
                    threshold,
                    predictive_target=predictive_target,
                    n_new_cases=n_new_cases,
                )
            except Exception as e:
                print(e)
                continue

            tprs["0"].append(tpr_dict["0"])
            tprs["1"].append(tpr_dict["1"])
            tnrs["0"].append(tnr_dict["0"])
            tnrs["1"].append(tnr_dict["1"])

        # Compute ROC curve and AUC
        roc_results = self._compute_roc_auc(tprs, tnrs)
        self.roc_results = roc_results  # Store results as instance variable
        self._roc_results_predictive_target = (
            predictive_target
        )
        self._roc_results_n_new_cases = n_new_cases

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
            tnr_values = [
                (
                    np.mean(tnr)
                    if isinstance(tnr, np.ndarray)
                    else tnr
                )
                for tnr in tnrs[setting]
            ]
            tpr_values = [
                (
                    np.mean(tpr)
                    if isinstance(tpr, np.ndarray)
                    else tpr
                )
                for tpr in tprs[setting]
            ]

            # Compute FPR from TNR: FPR = 1 - TNR
            fpr = [1 - tnr for tnr in tnr_values]
            tpr = tpr_values

            # Sort by FPR for proper ROC curve
            sorted_pairs = sorted(
                zip(fpr, tpr), key=lambda x: x[0]
            )
            fpr_sorted = [pair[0] for pair in sorted_pairs]
            tpr_sorted = [pair[1] for pair in sorted_pairs]

            # Compute AUC
            auc_score = auc(fpr_sorted, tpr_sorted)

            individual_results[setting] = {
                "fpr": fpr_sorted,
                "tpr": tpr_sorted,
                "auc": auc_score,
            }

        # Compute partial AUC for overlapping FPR range
        fpr_0 = np.array(individual_results["0"]["fpr"])
        fpr_1 = np.array(individual_results["1"]["fpr"])

        # Use the numeric interval intersection so both settings share one
        # common FPR window even when they do not have matching discrete points.
        fpr_min = max(
            float(np.min(fpr_0)), float(np.min(fpr_1))
        )
        fpr_max = min(
            float(np.max(fpr_0)), float(np.max(fpr_1))
        )

        for setting in ["0", "1"]:
            fpr_vals = individual_results[setting]["fpr"]
            tpr_vals = individual_results[setting]["tpr"]

            # Filter points within overlapping FPR range
            overlapping_indices = [
                i
                for i, fpr_val in enumerate(fpr_vals)
                if fpr_min <= fpr_val <= fpr_max
            ]

            if len(overlapping_indices) >= 2:
                # Extract overlapping FPR and TPR values
                fpr_partial = [
                    fpr_vals[i] for i in overlapping_indices
                ]
                tpr_partial = [
                    tpr_vals[i] for i in overlapping_indices
                ]

                # Compute partial AUC
                partial_auc = auc(fpr_partial, tpr_partial)
            else:
                # If insufficient overlapping points, use full AUC as fallback
                partial_auc = individual_results[setting][
                    "auc"
                ]

            # Combine results
            roc_results[setting] = {
                **individual_results[setting],
                "partial_auc": partial_auc,
                "partial_fpr_range": (fpr_min, fpr_max),
            }

        return roc_results

    def plot_tpr_tnr_by_threshold(
        self,
        filename: str = "figures/tpr_tnr_by_threshold.png",
        predictive_target: str = "observed_panel",
        n_new_cases: int | None = None,
    ):
        """Generate and save TPR/TNR plot with 95% HDI across thresholds.

        Args:
            filename: path where the figure will be saved. The directory
                portion of the path will be created if necessary.
            predictive_target: either ``observed_panel`` or
                ``new_cases``.
            n_new_cases: number of future cases per truth subset used
                when ``predictive_target='new_cases'``.

        Returns:
            str: path to the saved figure file.
        """
        return plot_tpr_fpr_by_threshold(
            self,
            filename,
            predictive_target=predictive_target,
            n_new_cases=n_new_cases,
        )

    def plot_roc_curve_with_hdi(
        self,
        filename: str = "figures/roc_curve_with_hdi.png",
        predictive_target: str = "observed_panel",
        n_new_cases: int | None = None,
    ):
        """Generate and save ROC curve plot with 95% HDI band and partial AUC uncertainty.

        This function builds ROC curves by varying classification thresholds across the dataset,
        computing posterior uncertainty for both the curve and partial AUC metric within the
        overlapping FPR range between treatment settings.

        Args:
            filename: path where the figure will be saved. The directory
                portion of the path will be created if necessary.
            predictive_target: either ``observed_panel`` or
                ``new_cases``.
            n_new_cases: number of future cases per truth subset used
                when ``predictive_target='new_cases'``.

        Returns:
            str: path to the saved figure file.
        """
        return plot_roc_curve_with_hdi(
            self,
            filename,
            predictive_target=predictive_target,
            n_new_cases=n_new_cases,
        )


class BalancedCaseInteractionModel(BalancedModel):
    def __init__(
        self,
        obs_data: pd.DataFrame | str,
        priors: Optional[dict] | Optional[str] = "diffuse",
    ):
        super().__init__(obs_data, priors)
        self.roc_results = (
            None  # TODO: refactor as property
        )

    def _setup_model(self, obs_data, priors) -> pm.Model:
        # setup coords
        reader, study_readers = obs_data.reader.factorize()
        case, study_cases = obs_data.case.factorize()
        treatment = obs_data.treatment.values

        coords = {
            "reader": study_readers,
            "case": study_cases,
        }
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
                mu=priors["b_mu"],
                sigma=priors["b_sigma"],
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

            # case variability
            mu_gamma_c = pm.Normal(
                "mu_gamma_c", mu=0, sigma=1
            )
            sigma_gamma_c = pm.HalfNormal(
                "sigma_gamma_c", 1
            )
            z_gamma_c = pm.Normal(
                "z_gamma_c", mu=0, sigma=1, dims="case"
            )
            gamma_c = pm.Deterministic(
                "case_variability",
                mu_gamma_c + z_gamma_c * sigma_gamma_c,
                dims="case",
            )

            # Reader-case interaction
            mu_delta_rc = pm.Normal(
                "mu_delta_rc", mu=0, sigma=1
            )
            sigma_delta_rc = pm.HalfNormal(
                "sigma_delta_rc", 1
            )
            z_delta_rc = pm.Normal(
                "z_delta_rc",
                mu=0,
                sigma=1,
                dims=["reader", "case"],
            )
            delta_rc = pm.Deterministic(
                "reader_case_interaction",
                mu_delta_rc + z_delta_rc * sigma_delta_rc,
                dims=["reader", "case"],
            )

            # probability of correct classification
            p = pm.math.clip(
                pm.math.invlogit(
                    alpha[reader_idx]
                    + beta[reader_idx] * treatment_idx
                    + gamma_c[case_idx]
                    + delta_rc[reader_idx, case_idx]
                ),
                epsilon,
                1 - epsilon,
            )

            # likelihood
            y = pm.Bernoulli(
                "k",
                p=p,
                observed=obs_data.rating_binary,
                dims="obs_id",
            )
        return model
    
    def _simulate_new_case_accuracy(
        self,
        idata,
        treatment: int,
        n_new_cases: int,
        random_seed: int,
    ):
        epsilon = 1e-2
        alpha, n_chains, n_draws = self._flatten_chain_draws(
            idata.posterior["alpha"].values
        )
        beta, _, _ = self._flatten_chain_draws(
            idata.posterior["beta"].values
        )
        mu_gamma_c, _, _ = self._flatten_chain_draws(
            idata.posterior["mu_gamma_c"].values
        )
        sigma_gamma_c, _, _ = self._flatten_chain_draws(
            idata.posterior["sigma_gamma_c"].values
        )
        mu_delta_rc, _, _ = self._flatten_chain_draws(
            idata.posterior["mu_delta_rc"].values
        )
        sigma_delta_rc, _, _ = self._flatten_chain_draws(
            idata.posterior["sigma_delta_rc"].values
        )

        rng = np.random.default_rng(random_seed)
        n_readers = alpha.shape[1]
        gamma_c = rng.normal(
            loc=mu_gamma_c.reshape(-1, 1),
            scale=sigma_gamma_c.reshape(-1, 1),
            size=(alpha.shape[0], n_new_cases),
        )
        delta_rc = rng.normal(
            loc=mu_delta_rc.reshape(-1, 1, 1),
            scale=sigma_delta_rc.reshape(-1, 1, 1),
            size=(alpha.shape[0], n_readers, n_new_cases),
        )

        eta = (
            alpha[:, :, None]
            + beta[:, :, None] * treatment
            + gamma_c[:, None, :]
            + delta_rc
        )
        p = np.clip(self._invlogit(eta), epsilon, 1 - epsilon)
        y = rng.binomial(1, p)
        mean_accuracy = y.mean(axis=(1, 2))
        return mean_accuracy.reshape(n_chains, n_draws)

    def _compute_tpr_tnr(
        self,
        threshold,
        predictive_target: str = "observed_panel",
        n_new_cases: int | None = None,
    ):
        """Compute TPR and TNR for a given threshold using posterior predictive samples.

        Args:
            threshold: Rating threshold for binarization
            predictive_target: either ``observed_panel`` for uncertainty
                conditional on the fitted cases or ``new_cases`` for
                posterior predictive uncertainty on unseen cases.
            n_new_cases: number of future cases per truth subset used
                when ``predictive_target='new_cases'``. Defaults to the
                number of observed cases in each truth subset.

        Returns:
            tuple: (tpr_dict, tnr_dict) where each dict has keys "0" and "1" for treatment settings.
                   Each value is a (chain, draw) array of posterior predictive accuracy samples.
        """
        predictive_target = self._validate_predictive_target(
            predictive_target
        )
        idatas = self.run_inference(threshold)

        # idatas[0] is for negative cases (truth==0), idatas[1] is for positive cases (truth==1)
        neg_idata = idatas[0]
        pos_idata = idatas[1]

        # Number of cases used to normalise the BetaBinomial count k
        neg_n_cases = len(
            self.obs_data[self.obs_data.truth == 0].case.unique()
        )
        pos_n_cases = len(
            self.obs_data[self.obs_data.truth == 1].case.unique()
        )

        if predictive_target == "new_cases":
            neg_case_count = n_new_cases or neg_n_cases
            pos_case_count = n_new_cases or pos_n_cases
            neg_fpr_dict = {
                "0": self._simulate_new_case_accuracy(
                    neg_idata,
                    treatment=0,
                    n_new_cases=neg_case_count,
                    random_seed=301,
                ),
                "1": self._simulate_new_case_accuracy(
                    neg_idata,
                    treatment=1,
                    n_new_cases=neg_case_count,
                    random_seed=302,
                ),
            }
            tnr_dict = {
                setting: 1 - neg_fpr
                for setting, neg_fpr in neg_fpr_dict.items()
            }
            tpr_dict = {
                "0": self._simulate_new_case_accuracy(
                    pos_idata,
                    treatment=0,
                    n_new_cases=pos_case_count,
                    random_seed=401,
                ),
                "1": self._simulate_new_case_accuracy(
                    pos_idata,
                    treatment=1,
                    n_new_cases=pos_case_count,
                    random_seed=402,
                ),
            }
            return tpr_dict, tnr_dict

        # Posterior predictive k: shape (chain, draw, obs_id)
        neg_k_pred = neg_idata.posterior_predictive["k"].values
        pos_k_pred = pos_idata.posterior_predictive["k"].values

        # Treatment index for each obs_id (0 = control, 1 = treatment)
        neg_treatment = neg_idata.constant_data["treatment_idx"].values
        pos_treatment = pos_idata.constant_data["treatment_idx"].values

        # For truth == 0, the Bernoulli likelihood observes rating_binary directly,
        # so posterior predictive means are FPR and must be complemented to TNR.
        fpr_0 = neg_k_pred[..., neg_treatment == 0].mean(axis=-1)
        fpr_1 = neg_k_pred[..., neg_treatment == 1].mean(axis=-1)
        tnr_0 = 1 - fpr_0
        tnr_1 = 1 - fpr_1
        tpr_0 = pos_k_pred[..., pos_treatment == 0].mean(axis=-1)
        tpr_1 = pos_k_pred[..., pos_treatment == 1].mean(axis=-1)

        tnr_dict = {"0": tnr_0, "1": tnr_1}
        tpr_dict = {"0": tpr_0, "1": tpr_1}
        return tpr_dict, tnr_dict

    # def _compute_tpr_tnr(self, threshold):
    #     """Compute TPR and TNR for a given threshold.

    #     Overrides BalancedModel._compute_tpr_tnr because this model's
    #     likelihood observes ``rating_binary`` directly (Bernoulli), so
    #     the negative-case model estimates P(rating >= threshold | truth=0)
    #     which is FPR, not TNR.  TNR is therefore 1 - that estimate.
    #     The positive-case model correctly estimates TPR.
    #     """
    #     idatas = self.run_inference(threshold)

    #     neg_idata = idatas[0]
    #     neg_a_mu = neg_idata["posterior_predictive"]["mu_a"].values
    #     neg_b_mu = neg_idata["posterior"]["mu_b"].values
    #     # Model estimates P(rating_binary==1 | truth=0) = FPR
    #     fpr_0, fpr_1 = compute_posterior_accuracy_by_treatment(
    #         neg_a_mu, neg_b_mu
    #     )
    #     # TNR = 1 - FPR
    #     tnr_dict = {"0": 1 - fpr_0, "1": 1 - fpr_1}

    #     pos_idata = idatas[1]
    #     pos_a_mu = pos_idata["posterior_predictive"]["mu_a"].values
    #     pos_b_mu = pos_idata["posterior_predictive"]["mu_b"].values
    #     # Model estimates P(rating_binary==1 | truth=1) = TPR
    #     tpr_0, tpr_1 = compute_posterior_accuracy_by_treatment(
    #         pos_a_mu, pos_b_mu
    #     )
    #     tpr_dict = {"0": tpr_0, "1": tpr_1}

    #     return tpr_dict, tnr_dict

    def run_inference(self, rating_threshold=0.5):
        print(
            "Using run inference method from BalancedCaseInteractionModel"
        )
        df = self.obs_data.copy()
        if (
            rating_threshold < 0
            or rating_threshold > df.rating.max()
        ):
            raise ValueError(
                f"Specified rating_threshold {rating_threshold}"
                " is not a valid rating value."
            )
        logging.info(
            f"Binarize rating data with rating threshold {rating_threshold}"
        )
        df["rating_binary"] = (
            df["rating"]
            .copy()
            .apply(
                lambda x: (
                    int(0)
                    if x < rating_threshold
                    else int(1)
                )
            )
        )
        negative_data = df[self.obs_data.truth == 0].copy()
        positive_data = df[self.obs_data.truth == 1].copy()

        # run inference for negative cases and positive cases seperately
        idatas = []
        for data in [negative_data, positive_data]:
            # data = obs_data.copy()
            model = self._setup_model(
                data,
                self.priors,
            )

            with model:
                idata = pm.sample(
                    draws=2000,
                    # nuts_sampler="blackjax",
                    **self._sampling_kwargs(),
                )
                pm.sample_posterior_predictive(
                    idata, extend_inferencedata=True
                )

            idatas.append(idata)
        return idatas
