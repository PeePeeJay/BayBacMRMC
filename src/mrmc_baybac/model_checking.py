### Prior sensitivity analysis
# Case B case level data
# Step 1: Chose true parameters for simulation
# Step 2: Simulate data
# Step 3: Fit BalancedCaseInteractionModel with different priors
# Step 4: compute empirical balanced accuracy (1/2 * (TPR + TNR)) from simulated data
#         and fit a linear model: smf.ols("emp_balanced_accuracy ~ treatment", data=data).fit()
# Step 5: repeat step 2 to 4 for multiple iterations and save results
# Step 6: Compare posterior estimates with true parameters

import gc
import json
import logging
from logging import config
import os
import pickle
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import statsmodels.formula.api as smf
from tqdm import tqdm
from scipy.special import logit, expit


from mrmc_baybac.model import (
    BalancedModel,
    BalancedCaseInteractionModel,
)
from mrmc_baybac.simulation import (
    mock_case_reading_data,
    mock_reading_data,
    simulate_aggregated_data,
    simulate_case_data,
)
from mrmc_baybac.utils import compute_empirical_ba

logging.getLogger("pymc").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["JAX_PLATFORMS"] = "cpu"


def run_model_inference(
    data: pd.DataFrame,
    priors: dict | str,
    n_draws: int = 2000,
    case_interaction: bool = False,
    rng=None,
) -> list:
    if case_interaction:

        model = BalancedCaseInteractionModel(
            obs_data=data, priors=priors
        )
    else:
        BalancedModel(obs_data=data, priors=priors)
    return model.run_inference()


def run_psa(
    mu_baseline_positive: float,
    mu_baseline_negative: float,
    effect_size_negative: float,
    effect_size_positive: float,
    num_sims: int = 50,
    case_interaction: bool = False,
    n_readers_sim: list | None = None,
    n_cases_neg: int = 120,
    n_cases_pos: int = 80,
    true_params: dict | None = None,
    n_draws: int = 2000,
    output_dir: str = "psa_results/estimates",
):
    """Run prior sensitivity analysis for the case-level reader interaction model.

    Steps performed for each replicate × reader-count × sigma_gamma × prior:
      1. True parameters are fixed (or passed via true_params).
      2. Simulate negative and positive case-level reading data.
      3. Fit Model with diffuse / weakly informative /
         informative priors
      4. Compute empirical balanced accuracy per treatment and fit OLS.
      5. Save estimates to
         <output_dir>/estimate_<sim>.<prior_idx>.<size_idx>.<gamma_idx>.pkl

    Each .pkl file contains a dict with keys:
        mu_a_neg, mu_b_neg  — posterior means for negative-case model
        mu_a_pos, mu_b_pos  — posterior means for positive-case model
        intercept_freq      — OLS intercept (BA at treatment=0)
        slope_freq          — OLS slope (BA treatment effect)
        true_params         — simulation ground truth (including sigma_gamma used)

    Args:
        num_sims: number of independent simulation replicates.
        case_interaction: use BalancedCaseInteractionModel when True,
                          BalancedModel otherwise.
        n_readers_sim: list of reader-count values to sweep.
        sigma_gamma_sim: list of case-variability SD values to sweep.
                         sigma_gamma in true_params is overridden each iteration.
        n_cases_neg: number of negative cases per simulation.
        n_cases_pos: number of positive cases per simulation.
        true_params: dict with mu_a, sigma_a, mu_b, sigma_b, sigma_gamma,
                     sigma_delta.  Defaults to a moderate-effect scenario.
        n_draws: MCMC draws per chain.
        output_dir: directory for saved estimate pickles.
    """
    # Step 1: True parameters

    balanced_mu_baseline = (
        mu_baseline_negative + mu_baseline_positive
    ) / 2
    balanced_effect_size = (
        effect_size_negative + effect_size_positive
    ) / 2
    mu_a = logit(balanced_mu_baseline)
    mu_b = logit(balanced_effect_size)

    true_params = {
        "mu_a": mu_a,
        "mu_b": mu_b,
    }
    if n_readers_sim is None:
        n_readers_sim = [2, 4, 6, 8, 10, 20, 100, 500, 1000]

    priors_options = [
        "diffuse",
        "weakly informative",
        "informative",
        "frequentist",
    ]

    sim_config = {
        "num_sims": num_sims,
        "n_cases_neg": n_cases_neg,
        "n_cases_pos": n_cases_pos,
        "n_draws": n_draws,
        "readers_sim": n_readers_sim,
        "priors": priors_options,
        "mu_baseline_negative": mu_baseline_negative,
        "mu_baseline_positive": mu_baseline_positive,
        "effect_size_negative": effect_size_negative,
        "effect_size_positive": effect_size_positive,
        "mu_a": mu_a,
        "mu_b": mu_b,
    }
    if not case_interaction:
        gamma_sim = np.arange(0.1, 0.6, 0.1)
        sim_config.update({"gamma_sim": list(gamma_sim)})

    os.makedirs(output_dir, exist_ok=True)

    parent_rng = np.random.default_rng(42)

    fpath = os.path.join(
        output_dir,
        f"sim_config.json",
    )
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(sim_config, f, indent=4)

    estimates = None

    for sim in tqdm(range(num_sims), desc="simulation"):
        # sim_rng = np.random.default_rng(
        #     parent_rng.integers(0, 2**31)
        # )
        for size_idx, n_readers in enumerate(
            tqdm(
                n_readers_sim,
                desc="reader size",
                leave=False,
            )
        ):
            if not case_interaction:
                for gamma_idx, gamma in enumerate(
                    tqdm(
                        gamma_sim, desc="gamma", leave=False
                    )
                ):
                    # Step 2: Simulate data with current sigma_gamma
                    iter_params = {
                        **true_params,
                        "gamma": gamma,
                    }
                    neg_sim = simulate_aggregated_data(
                        n_readers,
                        n_cases_neg,
                        mu_baseline=mu_baseline_negative,
                        effect_size=effect_size_negative,
                        rng=None,
                    )
                    pos_sim = simulate_aggregated_data(
                        n_readers,
                        n_cases_pos,
                        mu_baseline=mu_baseline_positive,
                        effect_size=effect_size_positive,
                        rng=None,
                    )

                    data = mock_reading_data(
                        neg_sim, pos_sim
                    )

                    # Step 4 (partial): Empirical balanced accuracy for frequentist model
                    emp_ba = compute_empirical_ba(data)

                    # Step 3 + 4: Fit each prior setting
                    for prior_idx, prior in enumerate(
                        priors_options
                    ):
                        if prior == "frequentist":
                            freq_res = smf.ols(
                                "emp_balanced_accuracy ~ treatment",
                                data=emp_ba,
                            ).fit()
                            estimates = {
                                "mu_a_neg": np.nan,
                                "mu_b_neg": np.nan,
                                "mu_a_pos": np.nan,
                                "mu_b_pos": np.nan,
                                "intercept_freq": freq_res.params[
                                    "Intercept"
                                ],
                                "slope_freq": freq_res.params[
                                    "treatment"
                                ],
                            }
                        else:
                            infer_rng = (
                                np.random.default_rng(
                                    parent_rng.integers(
                                        0, 2**31
                                    )
                                )
                            )
                            idatas = run_model_inference(
                                data,
                                prior,
                                n_draws=n_draws,
                                case_interaction=case_interaction,
                                rng=infer_rng,
                            )
                            idata_neg, idata_pos = idatas

                            summary_neg = az.summary(
                                idata_neg,
                                var_names=[
                                    "mu_a",
                                    "mu_b",
                                    "gamma",
                                   
                                ],
                                stat_focus="mean",
                            )
                            summary_pos = az.summary(
                                idata_pos,
                                var_names=[
                                    "mu_a",
                                    "mu_b",
                                    "gamma",
                                ],
                                stat_focus="mean",
                            )

                            estimates = {
                                "mu_a_neg": summary_neg.loc[
                                    "mu_a", "mean"
                                ],
                                "mu_b_neg": summary_neg.loc[
                                    "mu_b", "mean"
                                ],
                                "mu_a_pos": summary_pos.loc[
                                    "mu_a", "mean"
                                ],
                                "mu_b_pos": summary_pos.loc[
                                    "mu_b", "mean"
                                ],
                                "gamma_neg": summary_neg.loc[
                                    "gamma", "mean"
                                ],
                                "gamma_pos": summary_pos.loc[
                                    "gamma", "mean"
                                ],
                                "intercept_freq": np.nan,
                                "slope_freq": np.nan,
                            }

                            del idatas, idata_neg, idata_pos
                            gc.collect()

                        ### Save results
                        estimates["true_params"] = {
                            "mu_a": mu_a,
                            "mu_b": mu_b,
                            "sigma_params": 1.0,
                            "n_readers": n_readers,
                        }
                        if isinstance(prior, str):
                            estimates["true_params"][
                                "prior"
                            ] = prior
                        else:
                            estimates["true_params"][
                                "prior"
                            ] = prior_idx
                        estimates["sim_config"] = sim_config
                        estimates["replicate"] = sim
                        path = os.path.join(
                            output_dir,
                            f"nointeraction_{sim}.{prior_idx}.{n_readers}.{gamma_idx}.pkl",
                        )
                        with open(path, "wb") as f:
                            pickle.dump(estimates, f)
            else:
                ### case interaction model
                neg_sim = simulate_case_data(
                    n_readers,
                    n_cases_neg,
                    mu_baseline=mu_baseline_negative,
                    effect_size=effect_size_negative,
                )
                pos_sim = simulate_case_data(
                    n_readers,
                    n_cases_pos,
                    mu_baseline=mu_baseline_positive,
                    effect_size=effect_size_positive,
                )

                data = mock_case_reading_data(
                    neg_sim, pos_sim
                )

                # Step 4 (partial): Empirical balanced accuracy for frequentist model
                emp_ba = compute_empirical_ba(data)

                # Step 3 + 4: Fit each prior setting
                for prior_idx, prior in enumerate(
                    priors_options
                ):
                    if prior == "frequentist":
                        freq_res = smf.ols(
                            "emp_balanced_accuracy ~ treatment",
                            data=emp_ba,
                        ).fit()
                        estimates = {
                            "mu_a_neg": np.nan,
                            "mu_b_neg": np.nan,
                            "mu_a_pos": np.nan,
                            "mu_b_pos": np.nan,
                            "mu_gamma_neg": np.nan,
                            "mu_gamma_pos": np.nan,
                            "mu_delta_neg": np.nan,
                            "mu_delta_pos": np.nan,
                            "intercept_freq": freq_res.params[
                                "Intercept"
                            ],
                            "slope_freq": freq_res.params[
                                "treatment"
                            ],
                        }
                    else:

                        infer_rng = np.random.default_rng(
                            parent_rng.integers(0, 2**31)
                        )
                        idatas = run_model_inference(
                            data,
                            prior,
                            n_draws=n_draws,
                            case_interaction=case_interaction,
                            rng=infer_rng,
                        )
                        idata_neg, idata_pos = idatas

                        summary_neg = az.summary(
                            idata_neg,
                            var_names=[
                                "mu_a",
                                "mu_b",
                            ],
                            stat_focus="mean",
                        )
                        summary_pos = az.summary(
                            idata_pos,
                            var_names=[
                                "mu_a",
                                "mu_b",
                            ],
                            stat_focus="mean",
                        )

                        estimates = {
                            "mu_a_neg": summary_neg.loc[
                                "mu_a", "mean"
                            ],
                            "mu_b_neg": summary_neg.loc[
                                "mu_b", "mean"
                            ],
                            "mu_gamma_neg": idata_neg.posterior["case_variability"].values.flatten().mean(),
                            "mu_delta_neg": idata_neg.posterior["reader_case_interaction"].values.flatten().mean(),
                            "mu_a_pos": summary_pos.loc[
                                "mu_a", "mean"
                            ],
                            "mu_b_pos": summary_pos.loc[
                                "mu_b", "mean"
                            ],
                            "mu_gamma_pos": idata_pos.posterior["case_variability"].values.flatten().mean(),
                            "mu_delta_pos": idata_pos.posterior["reader_case_interaction"].values.flatten().mean(),
                            "intercept_freq": np.nan,
                            "slope_freq": np.nan,
                        }

                        del idatas, idata_neg, idata_pos
                        gc.collect()

                    estimates["true_params"] = {
                        "mu_a": mu_a,
                        "mu_b": mu_b,
                        "sigma_params": 1.0,
                    }
                    estimates["true_params"][
                        "n_readers"
                    ] = n_readers
                    if isinstance(prior, str):
                        estimates["true_params"][
                            "prior"
                        ] = prior
                    else:
                        estimates["true_params"][
                            "prior"
                        ] = prior_idx
                    estimates["sim_config"] = sim_config
                    estimates["replicate"] = sim
                    # logging.info(f'Prior | Param | Estimate | Difference \n {estimates["true_params"]["prior"]} | mu_a_neg | {estimates["mu_a_neg"]} | {estimates["mu_a_neg"] - estimates["true_params"]["mu_a"]} \n {estimates["true_params"]["prior"]} | mu_b_neg | {estimates["mu_b_neg"]} | {estimates["mu_b_neg"] - estimates["true_params"]["mu_b"]} \n {estimates["true_params"]["prior"]} | mu_a_pos | {estimates["mu_a_pos"]} | {estimates["mu_a_pos"] - estimates["true_params"]["mu_a"]} \n {estimates["true_params"]["prior"]} | mu_b_pos | {estimates["mu_b_pos"]} | {estimates["mu_b_pos"] - estimates["true_params"]["mu_b"]}')
                    #              )
                    path = os.path.join(
                        output_dir,
                        f"interaction_{sim}.{prior_idx}.{n_readers}.pkl",
                    )
                    with open(path, "wb") as f:
                        pickle.dump(estimates, f)
    return estimates


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run prior sensitivity analysis"
    )
    parser.add_argument(
        "--n-readers",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Reader counts to sweep (e.g. --n-readers 2 4 6 10). "
        "Defaults to [2, 4, 6, 8, 10, 20, 100, 500, 1000].",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="psa_results/estimates",
        metavar="DIR",
        help="Directory for saved estimate pickles (default: psa_results/estimates).",
    )
    parser.add_argument(
        "--case-interaction",
        action="store_true",
        default=False,
        help="Use BalancedCaseInteractionModel instead of BalancedModel.",
    )
    parser.add_argument(
        "--mu-baseline-negative",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Population log-odds intercept mu_a (default: None).",
    )
    parser.add_argument(
        "--mu-baseline-positive",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Population log-odds intercept mu_a (default: None).",
    )
    parser.add_argument(
        "--effect-size-negative",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Population log-odds treatment effect mu_b (default: None).",
    )
    parser.add_argument(
        "--effect-size-positive",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Population log-odds treatment effect mu_b (default: None).",
    )
    parser.add_argument(
        "--true-params",
        type=dict,
        default=None,
        metavar="FLOAT",
        help="Population log-odds treatment effect mu_b (default: 0.3).",
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=50,
        metavar="INT",
        help="Number of simulation replicates (default: 50).",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=2000,
        metavar="INT",
        help="MCMC draws per chain (default: 2000).",
    )
    parser.add_argument(
        "--n-cases-neg",
        type=int,
        default=120,
        metavar="INT",
        help="Number of negative cases per simulation (default: 120).",
    )
    parser.add_argument(
        "--n-cases-pos",
        type=int,
        default=80,
        metavar="INT",
        help="Number of positive cases per simulation (default: 80).",
    )
    args = parser.parse_args(argv)
    run_psa(
        n_readers_sim=args.n_readers,
        output_dir=args.output_dir,
        case_interaction=args.case_interaction,
        mu_baseline_negative=args.mu_baseline_negative,
        mu_baseline_positive=args.mu_baseline_positive,
        effect_size_negative=args.effect_size_negative,
        effect_size_positive=args.effect_size_positive,
        true_params=args.true_params,
        num_sims=args.num_sims,
        n_draws=args.n_draws,
        n_cases_neg=args.n_cases_neg,
        n_cases_pos=args.n_cases_pos,
    )


if __name__ == "__main__":
    main()
