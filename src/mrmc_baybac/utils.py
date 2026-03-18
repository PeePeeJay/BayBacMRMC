import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import xarray as xr
import pickle

from scipy.special import expit

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
    ratings, min_rating=0, max_rating=None
):
    if not all([val >= 0 for val in ratings]):
        raise ValueError(
            "Ratings must be non-negative. Please transform ratings."
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
        thresholds = np.arange(0.0, 1, 0.1)
    else:
        thresholds = ratings.unique()
    return thresholds


# ---------------------------------------------------------------------------
# Step 4 helper: empirical balanced accuracy
# ---------------------------------------------------------------------------


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

    if not case_interaction:
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
        for idx, value in enumerate(overdispersion_settings):
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
                with open(
                    file_path, "rb"
                ) as f:
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
                with open(
                    file_path, "rb"
                ) as f:
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

    return simulation_results


def average_posterior_pair(neg_value, pos_value):
        if neg_value is None or pos_value is None:
            return None
        return 0.5 * (neg_value + pos_value)

def balanced_accuracy_at_baseline(a, b):
    a = a.astype(float)
    b = b.astype(float)
    eta = a + b * (-0.5)
    ba = expit(eta)
    return ba


def effect_size(a, b):
    a = a.astype(float)
    b = b.astype(float)
    ba_baseline = balanced_accuracy_at_baseline(a, b)
    eta_intervention = a + b * (0.5)
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
    percent_bias = np.abs((estimates - true_value) / true_value) * 100
    return percent_bias


def get_plot_values(
    results: xr.Dataset, var_name: str, metric: str, gamma: float, prior: str
):
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
    x = df_mean.index.get_level_values("size").unique().values
    y = df_mean.loc[:, f"{var_name}_{metric}"]
    y_err = df_std.loc[:, f"{var_name}_{metric}"].values
    return x, y, y_err