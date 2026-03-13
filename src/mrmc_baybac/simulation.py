import pandas as pd
import pymc as pm
import numpy as np
from scipy.special import expit

RNG = np.random.default_rng(42)


def simulate_aggregated_data(
    n_readers,
    n_cases,
    true_params,
    rng=None
):
    """Simulate Beta-Binomial reading data for a full factorial (reader x setting) design.

    Returns arrays indexed by [reader, setting] where setting 0 is baseline
    and setting 1 is treatment.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    # --- unpack true parameters ---
    mu_a = true_params["mu_a"]
    sigma_a = true_params["sigma_a"]
    mu_b = true_params["mu_b"]
    sigma_b = true_params["sigma_b"]
    gamma = true_params["gamma"]

    # --- reader-level random effects ---
    z_a = rng.normal(0, 1, size=n_readers)
    z_b = rng.normal(0, 1, size=n_readers)

    alpha = mu_a + z_a * sigma_a  # shape (n_readers,)
    beta = mu_b + z_b * sigma_b   # shape (n_readers,)

    # --- linear predictor: shape (n_readers, 2) ---
    settings = np.array([0, 1])
    eta = alpha[:, np.newaxis] + beta[:, np.newaxis] * settings
    p = expit(eta)

    # --- Beta-Binomial parameters ---
    kappa = (1 - p) / p
    a_beta = (1 - gamma) / (gamma * (1 + kappa))
    b_beta = kappa * (1 - gamma) / (gamma * (1 + kappa))

    # --- draw outcomes: shape (n_readers, 2) ---
    k = rng.beta(a_beta, b_beta)  # draw latent probability
    k = rng.binomial(n_cases, k)  # convert to counts

    return {
        "n_readers": n_readers,
        "n_cases": n_cases,
        "k": k,
        "alpha": alpha,
        "beta": beta,
        "p": p,
        "a_beta": a_beta,
        "b_beta": b_beta,
    }


def mock_reading_data(
        negative_sim_data,
        positive_sim_data,

):
    neg_records = pd.DataFrame(columns=["reader", "case", "treatment", "rating", "truth"]) 
    neg_records["reader"] = np.tile(np.arange(negative_sim_data["n_readers"]), negative_sim_data["n_cases"]*2)
    neg_records["case"] = np.repeat(np.arange(negative_sim_data["n_cases"]), negative_sim_data["n_readers"]*2)
    neg_records["treatment"] = np.tile(np.repeat([0, 1], negative_sim_data["n_readers"]), negative_sim_data["n_cases"])
    neg_records["truth"] = 0
    for reader in range(negative_sim_data["n_readers"]):
        reader = int(reader)
        for treatment in [0, 1]:
            treatment = int(treatment)
            neg_records.loc[(neg_records["reader"] == reader) & (neg_records["treatment"] == treatment), "rating"] = np.concatenate([
                np.tile(0, negative_sim_data["k"][reader][treatment]),
                np.tile(1, negative_sim_data["n_cases"] - negative_sim_data["k"][reader][treatment]),
            ])

    pos_records = pd.DataFrame(columns=["reader", "case", "treatment", "rating", "truth"]) 
    pos_records["reader"] = np.tile(np.arange(positive_sim_data["n_readers"]), positive_sim_data["n_cases"]*2)
    pos_records["case"] = np.repeat(np.arange(positive_sim_data["n_cases"]), positive_sim_data["n_readers"]*2)
    pos_records["treatment"] = np.tile(np.repeat([0, 1], positive_sim_data["n_readers"]), positive_sim_data["n_cases"])
    pos_records["truth"] = 1
    for reader in range(positive_sim_data["n_readers"]):
        reader = int(reader)
        for treatment in [0, 1]:
            treatment = int(treatment)
            pos_records.loc[(pos_records["reader"] == reader) & (pos_records["treatment"] == treatment), "rating"] = np.concatenate([
                np.tile(1, positive_sim_data["k"][reader][treatment]),
                np.tile(0, positive_sim_data["n_cases"] - positive_sim_data["k"][reader][treatment]),
            ])
    records = pd.concat([neg_records, pos_records], ignore_index=True)
    return records


def simulate_case_data(
    n_readers,
    n_cases,
    true_params,
    rng=None,
):
    """Simulate case-level Bernoulli reading data with reader, case, and interaction effects.

    Implements the model:
        alpha[r] = mu_a + z_a[r] * sigma_a
        beta[r]  = mu_b + z_b[r] * sigma_b
        gamma_c[c] ~ Normal(0, sigma_gamma)
        delta_rc[r, c] ~ Normal(0, sigma_delta)
        eta[r, c, t] = alpha[r] + beta[r] * t + gamma_c[c] + delta_rc[r, c]
        p[r, c, t] = clip(invlogit(eta), epsilon, 1 - epsilon)

    Returns p indexed by [reader, case, setting] where setting 0 is baseline
    and setting 1 is treatment.

    true_params keys: mu_a, sigma_a, mu_b, sigma_b, sigma_gamma, sigma_delta.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    epsilon = 1e-2

    mu_a = true_params["mu_a"]
    sigma_a = true_params["sigma_a"]
    mu_b = true_params["mu_b"]
    sigma_b = true_params["sigma_b"]
    sigma_gamma = true_params.get("sigma_gamma", 1.0)
    sigma_delta = true_params.get("sigma_delta", 1.0)

    z_a = rng.normal(0, 1, size=n_readers)
    z_b = rng.normal(0, 1, size=n_readers)
    alpha = mu_a + z_a * sigma_a           # (n_readers,)
    beta = mu_b + z_b * sigma_b            # (n_readers,)

    gamma_c = rng.normal(0, sigma_gamma, size=n_cases)             # (n_cases,)
    delta_rc = rng.normal(0, sigma_delta, size=(n_readers, n_cases))  # (n_readers, n_cases)

    settings = np.array([0, 1])
    # broadcast to (n_readers, n_cases, 2)
    eta = (
        alpha[:, np.newaxis, np.newaxis]
        + beta[:, np.newaxis, np.newaxis] * settings[np.newaxis, np.newaxis, :]
        + gamma_c[np.newaxis, :, np.newaxis]
        + delta_rc[:, :, np.newaxis]
    )
    p = np.clip(expit(eta), epsilon, 1 - epsilon)

    return {
        "n_readers": n_readers,
        "n_cases": n_cases,
        "alpha": alpha,
        "beta": beta,
        "gamma_c": gamma_c,
        "delta_rc": delta_rc,
        "p": p,
    }


def mock_case_reading_data(negative_sim_data, positive_sim_data, rng=None):
    """Generate a tidy DataFrame of case-level Bernoulli ratings.

    Each row is one (reader, case, treatment) observation.
    Negative cases have truth=0; positive cases have truth=1 and are numbered
    starting from n_cases_neg to avoid collisions with negative case indices.
    Ratings are drawn independently from Bernoulli(p[reader, case, treatment]).
    """
    if rng is None:
        rng = np.random.default_rng(456)

    n_readers = negative_sim_data["n_readers"]
    n_cases_neg = negative_sim_data["n_cases"]
    n_cases_pos = positive_sim_data["n_cases"]

    # Draw case-level ratings: shape (n_readers, n_cases, 2)
    rating_neg = rng.binomial(1, negative_sim_data["p"])
    rating_pos = rng.binomial(1, positive_sim_data["p"])

    def _to_df(rating, n_cases, truth, case_offset=0):
        # Flattening order matches (n_readers, n_cases, 2) in C (row-major) order
        readers = np.repeat(np.arange(n_readers), n_cases * 2)
        cases = np.tile(np.repeat(np.arange(n_cases) + case_offset, 2), n_readers)
        treatments = np.tile([0, 1], n_readers * n_cases)
        return pd.DataFrame({
            "reader": readers,
            "case": cases,
            "treatment": treatments,
            "rating": rating.flatten(),
            "truth": truth,
        })

    neg_df = _to_df(rating_neg, n_cases_neg, truth=0)
    pos_df = _to_df(rating_pos, n_cases_pos, truth=1, case_offset=n_cases_neg)
    records = pd.concat([neg_df, pos_df], ignore_index=True)
    return records



