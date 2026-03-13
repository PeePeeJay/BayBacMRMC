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
              
    
    


