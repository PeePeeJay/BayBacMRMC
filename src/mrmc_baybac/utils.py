import os
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import xarray as xr
import pickle
import arviz as az

from scipy.special import expit
from sklearn.metrics import auc


def invlogit(x):
    return 1 / (1 + np.exp(-x))


def compute_posterior_effect_size(a_sample, b_sample):
    baseline_accuracy, treatment_accuracy = (
        compute_posterior_accuracy_by_treatment(
            a_sample, b_sample
        )
    )
    return treatment_accuracy - baseline_accuracy


def compute_posterior_accuracy_by_treatment(
    a_sample, b_sample
):
    baseline_accuracy = invlogit(a_sample)
    treatment_accuracy = invlogit(a_sample + b_sample)
    return baseline_accuracy, treatment_accuracy


def get_thresholds_from_ratings(
    ratings, min_rating=1, max_rating=None
):
    if not all([val >= 0 for val in ratings]):
        raise ValueError(
            "Ratings must be non-negative. Please transform ratings."
        )

    # Exclude rating value 0 from threshold construction.
    ratings = np.asarray(ratings)
    ratings = ratings[ratings != 0]
    if ratings.size == 0:
        raise ValueError(
            "No non-zero ratings available to construct thresholds."
        )

    if all([val % 1 == 0 for val in ratings]) and (
        np.max(ratings) <= 10
    ):
        thresholds = np.arange(
            min_rating,
            (
                np.max(ratings)
                if max_rating is None
                else max_rating
            ),
        )
    elif all([val <= 1 for val in ratings]):
        thresholds = np.arange(0.1, 1, 0.1)
    else:
        ratings = ratings[(ratings != 0) & (ratings != 100)]
        thresholds = pd.Series(ratings).unique()
        
    return thresholds

def bayes_p_value(a, b):
    """Compute the Bayesian p-value for the hypothesis that a > b based on posterior samples."""
    return np.sum(a > b) / len(a) 


def common_fpr_interval(*fpr_curves):
    """Return the numeric overlap of multiple FPR curves.

    Args:
        *fpr_curves: One or more arrays/lists of FPR values.

    Returns:
        tuple[float, float]: (lower_bound, upper_bound) of common interval.
    """
    minima = []
    maxima = []

    for curve in fpr_curves:
        arr = np.asarray(curve, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        minima.append(float(arr.min()))
        maxima.append(float(arr.max()))

    if not minima:
        raise ValueError("No valid FPR curves were provided.")

    fpr_min = max(minima)
    fpr_max = min(maxima)
    if fpr_min >= fpr_max:
        raise ValueError(
            "No common FPR interval exists: "
            f"lower bound {fpr_min:.4f} is not smaller than upper bound {fpr_max:.4f}."
        )

    return fpr_min, fpr_max


def get_or_reference_curves(
    or_specs: pd.DataFrame,
    fpr_col: str = "sens",
    setting0_col: str = "mean_0",
    setting1_col: str = "mean_1",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract and sort OR reference ROC curves for both treatment settings."""
    if fpr_col not in or_specs.columns:
        raise KeyError(f"Missing column in OR specs: {fpr_col}")
    if setting0_col not in or_specs.columns:
        raise KeyError(f"Missing column in OR specs: {setting0_col}")
    if setting1_col not in or_specs.columns:
        raise KeyError(f"Missing column in OR specs: {setting1_col}")

    fpr = 1 - np.asarray(or_specs[fpr_col], dtype=float)
    tpr_0 = np.asarray(or_specs[setting0_col], dtype=float)
    tpr_1 = np.asarray(or_specs[setting1_col], dtype=float)

    finite = np.isfinite(fpr) & np.isfinite(tpr_0) & np.isfinite(tpr_1)
    if not np.any(finite):
        raise ValueError("OR specs do not contain valid finite ROC values.")

    fpr = fpr[finite]
    tpr_0 = tpr_0[finite]
    tpr_1 = tpr_1[finite]

    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    return {
        "0": (fpr_sorted, tpr_0[order]),
        "1": (fpr_sorted, tpr_1[order]),
    }


def extract_partial_roc_curve(
    fpr,
    tpr,
    fpr_min: float = 0.1,
    fpr_max: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip/interpolate an ROC curve to a specific FPR interval."""
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    finite = np.isfinite(fpr) & np.isfinite(tpr)
    fpr = fpr[finite]
    tpr = tpr[finite]

    if fpr.size < 2:
        raise ValueError("Need at least two finite ROC points.")
    if fpr_min >= fpr_max:
        raise ValueError("fpr_min must be smaller than fpr_max.")

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    if fpr_min < fpr[0] or fpr_max > fpr[-1]:
        raise ValueError(
            "Requested FPR bounds are outside curve support: "
            f"[{fpr[0]:.4f}, {fpr[-1]:.4f}]"
        )

    mask = (fpr >= fpr_min) & (fpr <= fpr_max)
    fpr_partial = fpr[mask]
    tpr_partial = tpr[mask]

    tpr_at_min = np.interp(fpr_min, fpr, tpr)
    tpr_at_max = np.interp(fpr_max, fpr, tpr)

    if fpr_partial.size == 0 or fpr_partial[0] > fpr_min:
        fpr_partial = np.insert(fpr_partial, 0, fpr_min)
        tpr_partial = np.insert(tpr_partial, 0, tpr_at_min)
    else:
        fpr_partial[0] = fpr_min
        tpr_partial[0] = tpr_at_min

    if fpr_partial[-1] < fpr_max:
        fpr_partial = np.append(fpr_partial, fpr_max)
        tpr_partial = np.append(tpr_partial, tpr_at_max)
    else:
        fpr_partial[-1] = fpr_max
        tpr_partial[-1] = tpr_at_max

    return fpr_partial, tpr_partial


def partial_roc_auc_from_curve(
    fpr,
    tpr,
    fpr_min: float = 0.1,
    fpr_max: float = 0.7,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute partial AUC from one ROC curve inside FPR bounds."""
    fpr_partial, tpr_partial = extract_partial_roc_curve(
        fpr,
        tpr,
        fpr_min=fpr_min,
        fpr_max=fpr_max,
    )
    return float(auc(fpr_partial, tpr_partial)), fpr_partial, tpr_partial


def compare_partial_auc_between_settings(
    fpr_samples_0,
    tpr_samples_0,
    fpr_samples_1,
    tpr_samples_1,
    fpr_min: float = 0.1,
    fpr_max: float = 0.7,
    hdi_prob: float = 0.95,
) -> dict[str, Any]:
    """Compare partial AUC posterior samples between setting 1 and setting 0.

    The inputs are expected as arrays with shape (n_threshold_points, n_samples)
    or (n_samples, n_threshold_points). The function computes a shared FPR
    interval, sample-wise partial AUC for each setting, HDIs, and
    Pr(AUC_1 > AUC_0).
    """

    def _coerce_to_point_by_sample(name: str, arr_like) -> np.ndarray:
        arr = np.asarray(arr_like, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")
        # Expected orientation: (n_threshold_points, n_samples).
        # Rows = threshold points (few), columns = posterior samples (many).
        # No automatic transpose — callers must supply the correct orientation.
        return arr

    fpr0 = _coerce_to_point_by_sample("fpr_samples_0", fpr_samples_0)
    tpr0 = _coerce_to_point_by_sample("tpr_samples_0", tpr_samples_0)
    fpr1 = _coerce_to_point_by_sample("fpr_samples_1", fpr_samples_1)
    tpr1 = _coerce_to_point_by_sample("tpr_samples_1", tpr_samples_1)

    if fpr0.shape != tpr0.shape:
        raise ValueError(
            "fpr_samples_0 and tpr_samples_0 must have matching shape."
        )
    if fpr1.shape != tpr1.shape:
        raise ValueError(
            "fpr_samples_1 and tpr_samples_1 must have matching shape."
        )
    if fpr0.shape[1] != fpr1.shape[1]:
        raise ValueError(
            "Both settings must have the same number of posterior samples."
        )

    if fpr_min is None and fpr_max is None:
        shared_min, shared_max = common_fpr_interval(fpr0, fpr1)
        fpr_min = max(float(fpr_min), shared_min)
        fpr_max = min(float(fpr_max), shared_max)

    if fpr_min >= fpr_max:
        raise ValueError(
            "Invalid partial FPR bounds after intersection: "
            f"[{fpr_min:.4f}, {fpr_max:.4f}]."
        )

    auc0 = []
    auc1 = []
    for i in range(fpr0.shape[1]):
        try:
            auc0_i, _, _ = partial_roc_auc_from_curve(
                fpr0[:, i], tpr0[:, i], fpr_min=fpr_min, fpr_max=fpr_max
            )
            auc1_i, _, _ = partial_roc_auc_from_curve(
                fpr1[:, i], tpr1[:, i], fpr_min=fpr_min, fpr_max=fpr_max
            )
        except ValueError:
            continue
        auc0.append(auc0_i)
        auc1.append(auc1_i)

    auc0 = np.asarray(auc0, dtype=float)
    auc1 = np.asarray(auc1, dtype=float)
    if auc0.size == 0 or auc1.size == 0:
        raise ValueError(
            "No valid posterior samples available to compute partial AUC comparison. "
            f"All {fpr0.shape[1]} samples failed the FPR bounds check "
            f"[{fpr_min:.4f}, {fpr_max:.4f}]. "
            "Check that fpr_min/fpr_max lie within the ROC curve support."
        )

    auc_diff = auc1 - auc0
    prob_auc1_gt_auc0 = float(np.mean(auc_diff > 0))

    hdi0 = az.hdi(auc0, hdi_prob=hdi_prob)
    hdi1 = az.hdi(auc1, hdi_prob=hdi_prob)
    hdi_diff = az.hdi(auc_diff, hdi_prob=hdi_prob)

    return {
        "fpr_bounds": (float(fpr_min), float(fpr_max)),
        "auc_samples_0": auc0,
        "auc_samples_1": auc1,
        "auc_diff_samples": auc_diff,
        "auc_mean_0": float(np.mean(auc0)),
        "auc_mean_1": float(np.mean(auc1)),
        "auc_diff_mean": float(np.mean(auc_diff)),
        "auc_hdi_0": (float(hdi0[0]), float(hdi0[1])),
        "auc_hdi_1": (float(hdi1[0]), float(hdi1[1])),
        "auc_diff_hdi": (float(hdi_diff[0]), float(hdi_diff[1])),
        "prob_auc1_gt_auc0": prob_auc1_gt_auc0,
        "n_valid_samples": int(auc0.size),
    }

def compute_empirical_ba(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute empirical balanced accuracy per treatment from binary case-level ratings.

    BA = 0.5 * (TPR + TNR)

    Args:
        data: DataFrame with columns reader, case, treatment, rating, truth.
              rating and truth are expected to be binary (0/1).

    Returns:
        DataFrame with columns treatment and emp_balanced_accuracy.
    """
    records = []
    for treatment in sorted(data["treatment"].unique()):
        subset = data[data["treatment"] == treatment]
        neg = subset[subset["truth"] == 0]
        pos = subset[subset["truth"] == 1]
        tnr = (
            (neg["rating"] == 0).mean()
            if len(neg) > 0
            else np.nan
        )
        tpr = (
            (pos["rating"] == 1).mean()
            if len(pos) > 0
            else np.nan
        )
        records.append(
            {
                "treatment": int(treatment),
                "emp_balanced_accuracy": 0.5 * (tpr + tnr),
            }
        )
    return pd.DataFrame(records)


def get_simulation_configs(path: str | os.PathLike) -> dict:
    """Load simulation configuration JSON and return it as a dictionary.

    Accepts either a direct path to `sim_config.json` or a directory that
    contains `sim_config.json`.
    """
    config_path = Path(path)
    if config_path.is_dir():
        config_path = config_path / "sim_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Simulation config file not found: {config_path}"
        )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Simulation config path is not a file: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a JSON object in {config_path}, got {type(config).__name__}."
        )

    return config


def read_psa_estimates_from_directory(
    path, case_interaction=False
):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Specified path does not exist: {path}"
        )
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Specified path is not a directory: {path}"
        )
    posterior_vars = [
        "mu_a_neg",
        "mu_b_neg",
        "mu_a_pos",
        "mu_b_pos",
        "intercept_freq",
        "slope_freq",
    ]
    if case_interaction:
        posterior_vars += [
            "mu_gamma_neg",
            "mu_gamma_pos",
            "mu_delta_neg",
            "mu_delta_pos",
        ]
    else:
        posterior_vars += ["gamma_neg", "gamma_pos"]

    sim_config = get_simulation_configs(path)
    reader_settings = list(sim_config["readers_sim"])
    overdispersion_settings = list(
        sim_config.get("overdispersion", [])
    )
    num_sims = int(sim_config["num_sims"])
    priors = list(sim_config["priors"])

    reader_to_idx = {
        value: idx
        for idx, value in enumerate(reader_settings)
    }
    prior_to_idx = {
        value: idx for idx, value in enumerate(priors)
    }

    def _gamma_index(gamma_value: float) -> int:
        for idx, value in enumerate(
            overdispersion_settings
        ):
            if np.isclose(value, gamma_value):
                return idx
        raise KeyError(
            "Gamma value not found in sim_config['overdispersion']: "
            f"{gamma_value}"
        )

    if not case_interaction:
        output_vars = {
            var: np.full(
                (
                    num_sims,
                    len(reader_settings),
                    len(overdispersion_settings),
                    len(priors),
                ),
                None,
                dtype=object,
            )
            for var in posterior_vars
        }
        for file in sorted(os.listdir(path)):
            if file.endswith(".pkl"):
                file_path = os.path.join(path, file)
                with open(file_path, "rb") as f:
                    est = pickle.load(f)
                replicate_idx = int(est["replicate"])
                reader_idx = reader_to_idx[
                    est["true_params"]["n_readers"]
                ]
                gamma_idx = _gamma_index(
                    est["true_params"]["gamma"]
                )
                prior_idx = prior_to_idx[
                    est["true_params"]["prior"]
                ]
                posterior_source = est.get(
                    "posterior_samples", est
                )

                for var in posterior_vars:
                    if var not in posterior_source:
                        continue
                    output_vars[var][
                        replicate_idx,
                        reader_idx,
                        gamma_idx,
                        prior_idx,
                    ] = posterior_source[var]

        simulation_results = xr.Dataset(
            {
                var: (
                    [
                        "replicate",
                        "num_readers",
                        "overdispersion",
                        "priors",
                    ],
                    output_vars[var],
                )
                for var in posterior_vars
            },
            coords={
                "replicate": np.arange(num_sims),
                "num_readers": reader_settings,
                "overdispersion": overdispersion_settings,
                "priors": priors,
            },
        )
    else:
        output_vars = {
            var: np.full(
                (
                    num_sims,
                    len(reader_settings),
                    len(priors),
                ),
                None,
                dtype=object,
            )
            for var in posterior_vars
        }
        for file in sorted(os.listdir(path)):
            if file.endswith(".pkl"):
                file_path = os.path.join(path, file)
                with open(file_path, "rb") as f:
                    est = pickle.load(f)
                replicate_idx = int(est["replicate"])
                reader_idx = reader_to_idx[
                    est["true_params"]["n_readers"]
                ]
                prior_idx = prior_to_idx[
                    est["true_params"]["prior"]
                ]
                posterior_source = est.get(
                    "posterior_samples", est
                )

                for var in posterior_vars:
                    if var not in posterior_source:
                        continue
                    output_vars[var][
                        replicate_idx,
                        reader_idx,
                        prior_idx,
                    ] = posterior_source[var]

        simulation_results = xr.Dataset(
            {
                var: (
                    ["replicate", "num_readers", "priors"],
                    output_vars[var],
                )
                for var in posterior_vars
            },
            coords={
                "replicate": np.arange(num_sims),
                "num_readers": reader_settings,
                "priors": priors,
            },
        )

    paired_var_names = sorted(
        {
            var_name.removesuffix("_neg")
            for var_name in simulation_results.data_vars
            if var_name.endswith("_neg")
            and f"{var_name.removesuffix('_neg')}_pos"
            in simulation_results.data_vars
        }
    )
    for var_name in paired_var_names:
        simulation_results[var_name] = xr.apply_ufunc(
            average_posterior_pair,
            simulation_results[f"{var_name}_neg"],
            simulation_results[f"{var_name}_pos"],
            vectorize=True,
        )

    simulation_results = add_metrics_to_results(
        simulation_results,
        sim_config,
        case_interaction=case_interaction,
    )

    return simulation_results


def add_metrics_to_results(
    simulation_results: xr.Dataset,
    sim_config: dict,
    case_interaction: bool = False,
) -> xr.Dataset:
    true_mu_a = sim_config["mu_a"]
    true_mu_b = sim_config["mu_b"]

    true_slope = expit(true_mu_b)
    true_intercept = expit(true_mu_a)

    if case_interaction:
        simulation_results["intercept_bayes"] = xr.apply_ufunc(
                accuracy_at_baseline_case_interaction,
                simulation_results["mu_a"],
                simulation_results["mu_b"],
                simulation_results["mu_gamma"],
                simulation_results["mu_delta"],
        )
        simulation_results["slope_bayes"] = xr.apply_ufunc(
            effect_size_case_interaction,
            simulation_results["mu_a"],
            simulation_results["mu_b"],
            simulation_results["mu_gamma"],
            simulation_results["mu_delta"],
        )
        
    else:
        simulation_results["intercept_bayes"] = (
            xr.apply_ufunc(
                accuracy_at_baseline,
                simulation_results["mu_a"],
                simulation_results["mu_b"],
            )
        )
        simulation_results["slope_bayes"] = xr.apply_ufunc(
            effect_size,
            simulation_results["mu_a"],
            simulation_results["mu_b"],
        )
    simulation_results["slope_bayes_abs_error"] = (
        xr.apply_ufunc(
            compute_absolute_error,
            simulation_results["slope_bayes"],
            true_slope,
            input_core_dims=[["replicate"], []],
            output_core_dims=[["replicate"]],
            vectorize=True,
        )
    )
    simulation_results["slope_bayes_bias"] = xr.apply_ufunc(
        compute_percent_bias,
        simulation_results["slope_bayes"],
        true_slope,
        input_core_dims=[["replicate"], []],
        output_core_dims=[["replicate"]],
        vectorize=True,
    )
    simulation_results["intercept_bayes_abs_error"] = (
        xr.apply_ufunc(
            compute_absolute_error,
            simulation_results["intercept_bayes"],
            true_intercept,
            input_core_dims=[["replicate"], []],
            output_core_dims=[["replicate"]],
            vectorize=True,
        )
    )
    simulation_results["intercept_bayes_bias"] = (
        xr.apply_ufunc(
            compute_percent_bias,
            simulation_results["intercept_bayes"],
            true_intercept,
            input_core_dims=[["replicate"], []],
            output_core_dims=[["replicate"]],
            vectorize=True,
        )
    )
    simulation_results["slope_freq_abs_error"] = (
        xr.apply_ufunc(
            compute_absolute_error,
            simulation_results["slope_freq"],
            true_slope,
            input_core_dims=[["replicate"], []],
            output_core_dims=[["replicate"]],
            vectorize=True,
        )
    )
    simulation_results["slope_freq_bias"] = xr.apply_ufunc(
        compute_percent_bias,
        simulation_results["slope_freq"],
        true_slope,
        input_core_dims=[["replicate"], []],
        output_core_dims=[["replicate"]],
        vectorize=True,
    )
    simulation_results["intercept_freq_abs_error"] = (
        xr.apply_ufunc(
            compute_absolute_error,
            simulation_results["intercept_freq"],
            true_intercept,
            input_core_dims=[["replicate"], []],
            output_core_dims=[["replicate"]],
            vectorize=True,
        )
    )
    simulation_results["intercept_freq_bias"] = (
        xr.apply_ufunc(
            compute_percent_bias,
            simulation_results["intercept_freq"],
            true_intercept,
            input_core_dims=[["replicate"], []],
            output_core_dims=[["replicate"]],
            vectorize=True,
        )
    )

    if not case_interaction:
        simulation_results["overdispersion_abs_error"] = (
            xr.apply_ufunc(
                compute_absolute_error,
                simulation_results["gamma"],
                simulation_results["overdispersion"],
                input_core_dims=[["replicate"], []],
                output_core_dims=[["replicate"]],
                vectorize=True,
            )
        )
        simulation_results["overdispersion_bias"] = (
            xr.apply_ufunc(
                compute_percent_bias,
                simulation_results["gamma"],
                simulation_results["overdispersion"],
                input_core_dims=[["replicate"], []],
                output_core_dims=[["replicate"]],
                vectorize=True,
            )
        )
        simulation_results["overdispersion_se"] = (
            xr.apply_ufunc(
                compute_se,
                simulation_results["gamma"],
                simulation_results["overdispersion"],
                input_core_dims=[["replicate"], []],
                output_core_dims=[["replicate"]],
                vectorize=True,
            )
        )

    # cast data variable to float
    for variable in simulation_results.data_vars:
        if simulation_results[variable].dtype == object:
            simulation_results[variable] = (
                simulation_results[variable].astype(float)
            )

    return simulation_results


def average_posterior_pair(neg_value, pos_value):
    if neg_value is None or pos_value is None:
        return None
    return 0.5 * (neg_value + pos_value)


def accuracy_at_baseline(a, b):
    a = a.astype(float)
    b = b.astype(float)
    eta = a
    ba = expit(eta)
    return ba


def accuracy_at_baseline_case_interaction(
    a, b, gamma, delta
):
    a = a.astype(float)
    b = b.astype(float)
    gamma = gamma.astype(float)
    delta = delta.astype(float)
    eta = a + gamma + delta
    ba = expit(eta)
    return ba


def effect_size(a, b):
    a = a.astype(float)
    b = b.astype(float)
    ba_baseline = accuracy_at_baseline(a, b)
    eta_intervention = a + b * 1
    ba_intervention = expit(eta_intervention)
    return ba_intervention - ba_baseline


def effect_size_case_interaction(a, b, gamma, delta):
    a = a.astype(float)
    b = b.astype(float)
    gamma = gamma.astype(float)
    delta = delta.astype(float)
    ba_baseline = accuracy_at_baseline_case_interaction(
        a, b, gamma, delta
    )
    eta_intervention = a + b * 1 + gamma + delta
    ba_intervention = expit(eta_intervention)
    return ba_intervention - ba_baseline


def compute_se(estimates, true_value):
    y_true = np.full(len(estimates), true_value)
    return (y_true - estimates) / y_true * 100


def compute_absolute_error(estimates, true_value):
    y_true = np.full(len(estimates), true_value)
    return np.abs(y_true - estimates)


def compute_mean(estimates):
    return estimates.mean()


def compute_percent_bias(estimates, true_value):
    y_true = np.full(len(estimates), true_value)
    percent_bias = (
        np.abs((estimates - true_value) / true_value) * 100
    )
    return percent_bias


def get_plot_values(
    results: xr.Dataset,
    var_name: str,
    metric: str,
    prior: str,
    case_interaction: bool = False,
    gamma: float | bool = None,
):
    if case_interaction:
        df_mean = (
            results[[f"{var_name}_{metric}"]]
            .sel(priors=prior)
            .mean("replicate")
            .to_dataframe()
        )
        df_std = (
            results[[f"{var_name}_{metric}"]]
            .sel(priors=prior)
            .std("replicate")
            .to_dataframe()
        )
    else:
        df_mean = (
            results[[f"{var_name}_{metric}"]]
            .sel(overdispersion=gamma, priors=prior)
            .mean("replicate")
            .to_dataframe()
        )
        df_std = (
            results[[f"{var_name}_{metric}"]]
            .sel(overdispersion=gamma, priors=prior)
            .std("replicate")
            .to_dataframe()
        )
    x = (
        df_mean.index.get_level_values("num_readers")
        .unique()
        .values
    )
    y = df_mean.loc[:, f"{var_name}_{metric}"]
    y_err = df_std.loc[:, f"{var_name}_{metric}"].values
    return x, y, y_err
