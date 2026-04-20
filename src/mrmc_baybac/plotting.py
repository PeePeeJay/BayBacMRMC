import numpy as np
import arviz as az
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator
import os
import pickle

from mrmc_baybac.utils import (
    get_simulation_configs,
    get_thresholds_from_ratings,
    read_psa_estimates_from_directory,
    get_plot_values,
    bayes_p_value,
)


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.variant"] = "normal"
ax_label_fs = 18
sup_ax_label_fs = 24
ax_tick_fs = 14
title_fs = 16
legend_fs = 13
plt.style.use("grayscale")


def plot_tpr_fpr_by_threshold(
    model,
    filename: str = "figures/tpr_tnr_by_threshold.png",
    predictive_target: str = "observed_panel",
    n_new_cases: int | None = None,
):
    """Generate and save TPR/FPR plots with 95% HDI across thresholds.

    Also computes Bayesian p-values for posterior comparisons
    ``setting 1 > setting 0`` per threshold, for both TPR and FPR,
    and overlays those p-values on a right-side secondary y-axis.

    Args:
        model: The BalancedModel instance.
        filename: path where the figure will be saved. The directory
            portion of the path will be created if necessary.
        predictive_target: either ``observed_panel`` or ``new_cases``.
        n_new_cases: number of future cases per truth subset used when
            ``predictive_target='new_cases'``.

    Returns:
        str: path to the saved figure file.
    """
    thresholds = get_thresholds_from_ratings(
        model.obs_data.rating
    )
    # Sort thresholds in ascending order
    thresholds = np.sort(thresholds)

    tprs, fprs = {"0": [], "1": []}, {"0": [], "1": []}

    # Collect posterior samples for all thresholds in sorted order
    for threshold in thresholds:
        tpr_dict, tnr_dict = model._compute_tpr_tnr(
            threshold,
            predictive_target=predictive_target,
            n_new_cases=n_new_cases,
        )
        tprs["0"].append(tpr_dict["0"])
        tprs["1"].append(tpr_dict["1"])
        fprs["0"].append(1 - tnr_dict["0"])
        fprs["1"].append(1 - tnr_dict["1"])

    # Compute mean and HDI for each threshold and setting
    metrics = {}
    for setting in ["0", "1"]:
        means_tpr = [np.mean(s) for s in tprs[setting]]
        means_fpr = [np.mean(s) for s in fprs[setting]]

        # Compute 95% HDI (flatten posterior samples first)
        hdi_tpr = [
            az.hdi(s.flatten(), hdi_prob=0.95)
            for s in tprs[setting]
        ]
        hdi_fpr = [
            az.hdi(s.flatten(), hdi_prob=0.95)
            for s in fprs[setting]
        ]

        # Extract lower and upper bounds as scalars
        tpr_lower = [float(hdi[0]) for hdi in hdi_tpr]
        tpr_upper = [float(hdi[1]) for hdi in hdi_tpr]
        fpr_lower = [float(hdi[0]) for hdi in hdi_fpr]
        fpr_upper = [float(hdi[1]) for hdi in hdi_fpr]
        
        metrics[setting] = {
            "thresholds": thresholds,
            "tpr_mean": means_tpr,
            "tpr_lower": tpr_lower,
            "tpr_upper": tpr_upper,
            "fpr_mean": means_fpr,
            "fpr_lower": fpr_lower,
            "fpr_upper": fpr_upper,
            "tpr_samples": [np.asarray(s, dtype=float).ravel() for s in tprs[setting]],
            "fpr_samples": [np.asarray(s, dtype=float).ravel() for s in fprs[setting]],
        }

    # Bayesian posterior probability that setting 1 is larger than setting 0
    # for each threshold, computed from paired posterior samples.
    metrics["bayes_p_value"] = {
        "tpr_1_gt_0": [
            bayes_p_value(
                np.asarray(tprs["1"][i]).flatten(),
                np.asarray(tprs["0"][i]).flatten(),
            )
            for i in range(len(thresholds))
        ],
        "fpr_1_gt_0": [
            bayes_p_value(
                np.asarray(fprs["1"][i]).flatten(),
                np.asarray(fprs["0"][i]).flatten(),
            )
            for i in range(len(thresholds))
        ],
    }

    # Ensure output directory exists
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Create figure with TPR and TNR subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Grayscale style map to keep settings distinguishable in B/W output.
    setting_style = {
        "0": {
            "color": "dimgray",
            "linestyle": "--",
            "marker": "s",
            "fill_alpha": 0.15,
        },
        "1": {
            "color": "black",
            "linestyle": "-",
            "marker": "o",
            "fill_alpha": 0.25,
        },
    }
    settings = {"0": "Baseline", "1": "Enhanced"}
    for setting in ["0", "1"]:
        m = metrics[setting]
        style = setting_style[setting]
        thresholds_x = m["thresholds"]

        # TPR subplot
        axes[0].plot(
            thresholds_x,
            m["tpr_mean"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=f"{settings[setting]} imaging",
        )
        axes[0].fill_between(
            thresholds_x,
            m["tpr_lower"],
            m["tpr_upper"],
            color=style["color"],
            alpha=style["fill_alpha"],
        )

        # fpr subplot
        axes[1].plot(
            thresholds_x,
            m["fpr_mean"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=f"{settings[setting]} imaging",
        )
        axes[1].fill_between(
            thresholds_x,
            m["fpr_lower"],
            m["fpr_upper"],
            color=style["color"],
            alpha=style["fill_alpha"],
        )

    # Overlay Bayesian p-value on right-side axis for TPR
    ax_tpr_p = axes[0].twinx()
    ax_tpr_p.set_zorder(axes[0].get_zorder() + 1)
    ax_tpr_p.patch.set_visible(False)
    ax_tpr_p.plot(
        thresholds,
        metrics["bayes_p_value"]["tpr_1_gt_0"],
        color="black",
        linestyle=":",
        marker="x",
        label=r"$Pr(\Delta TPR>0)$",
        zorder=10,
    )
    ax_tpr_p.set_ylabel(
        "Posterior probability", color="black", fontsize=ax_label_fs
    )
    ax_tpr_p.set_ylim(0, 1)
    ax_tpr_p.tick_params(labelsize=ax_tick_fs)

    # Configure TPR subplot
    axes[0].set_xlabel("Threshold", fontsize=ax_label_fs)
    axes[0].set_ylabel(
        "Posterior TPR", fontsize=ax_label_fs
    )
    axes[0].set_title(
        "Posterior TPR (with 95% HDI) \n ", fontsize=title_fs
    )
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_locator(FixedLocator(np.arange(0, 1.1, 0.1)))
    axes[0].yaxis.set_major_formatter(FixedFormatter([f"{x:.1f}" for x in np.arange(0, 1.1, 0.1)]))
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    h_tpr, l_tpr = axes[0].get_legend_handles_labels()
    h_tpr_p, l_tpr_p = ax_tpr_p.get_legend_handles_labels()
    axes[0].legend(
        h_tpr + h_tpr_p, l_tpr + l_tpr_p, fontsize=legend_fs
    )
    axes[0].tick_params(labelsize=ax_tick_fs)
    axes[0].grid(True, alpha=0.3)

    # Overlay Bayesian p-value on right-side axis for FPR
    ax_fpr_p = axes[1].twinx()
    ax_fpr_p.set_zorder(axes[1].get_zorder() + 1)
    ax_fpr_p.patch.set_visible(False)
    ax_fpr_p.plot(
        thresholds,
        metrics["bayes_p_value"]["fpr_1_gt_0"],
        color="black",
        linestyle=":",
        marker="x",
        label=r"$Pr(\Delta FPR>0)$",
        zorder=10,
    )
    ax_fpr_p.set_ylabel(
        "Posterior probability", color="black", fontsize=ax_label_fs
    )
    ax_fpr_p.set_ylim(0, 1)
    ax_fpr_p.tick_params(labelsize=ax_tick_fs)

    # Configure FPR subplot
    axes[1].set_xlabel("Threshold", fontsize=ax_label_fs)
    axes[1].set_ylabel(
        "Posterior FPR", fontsize=ax_label_fs
    )
    axes[1].set_title(
        "Posterior FPR (with 95% HDI) \n ", fontsize=title_fs
    )
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_locator(FixedLocator(np.arange(0, 1.1, 0.1)))
    axes[1].yaxis.set_major_formatter(FixedFormatter([f"{x:.1f}" for x in np.arange(0, 1.1, 0.1)]))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    h_fpr, l_fpr = axes[1].get_legend_handles_labels()
    h_fpr_p, l_fpr_p = ax_fpr_p.get_legend_handles_labels()
    axes[1].legend(
        h_fpr + h_fpr_p, l_fpr + l_fpr_p, fontsize=legend_fs
    )
    axes[1].tick_params(labelsize=ax_tick_fs)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout(pad=2.0)
    fig.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close()
    return filename, metrics


def plot_roc_curve_with_hdi(
    model,
    filename: str = "figures/roc_curve_with_hdi.png",
    predictive_target: str = "observed_panel",
    n_new_cases: int | None = None,
):
    """Generate and save ROC curve plot with 95% HDI band and partial AUC uncertainty.

    This function builds ROC curves by varying classification thresholds across the dataset,
    computing posterior uncertainty for both the curve and partial AUC metric within the
    overlapping FPR range between treatment settings.

    Args:
        model: The BalancedModel instance.
        filename: path where the figure will be saved. The directory
            portion of the path will be created if necessary.
        predictive_target: either ``observed_panel`` or ``new_cases``.
        n_new_cases: number of future cases per truth subset used when
            ``predictive_target='new_cases'``.

    Returns:
        str: path to the saved figure file.
    """
    # Get ROC results including partial AUC
    if (
        model.roc_results is None
        or getattr(
            model,
            "_roc_results_predictive_target",
            None,
        )
        != predictive_target
        or getattr(model, "_roc_results_n_new_cases", None)
        != n_new_cases
    ):
        model.roc_curve_analysis(
            predictive_target=predictive_target,
            n_new_cases=n_new_cases,
        )

    roc_results = model.roc_results

    thresholds = get_thresholds_from_ratings(
        model.obs_data.rating
    )
    thresholds = np.sort(thresholds)

    # Collect TPR and TNR posterior samples across all thresholds
    all_tprs = {"0": [], "1": []}
    all_tnrs = {"0": [], "1": []}

    for threshold in thresholds:
        try:
            tpr_dict, tnr_dict = model._compute_tpr_tnr(
                threshold,
                predictive_target=predictive_target,
                n_new_cases=n_new_cases,
            )
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

        # Get partial AUC range for this setting
        partial_fpr_range = roc_results[setting][
            "partial_fpr_range"
        ]
        fpr_min, fpr_max = partial_fpr_range

        # Stack posterior samples to get shape: (n_thresholds, n_posterior_samples)
        tprs_stacked = []
        tnrs_stacked = []

        for tpr, tnr in zip(
            all_tprs[setting], all_tnrs[setting]
        ):
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

        tprs_array = np.array(
            [
                np.pad(t, (0, n_posteriors - len(t)))
                for t in tprs_stacked
            ]
        )
        tnrs_array = np.array(
            [
                np.pad(t, (0, n_posteriors - len(t)))
                for t in tnrs_stacked
            ]
        )

        # Compute FPR for each threshold and posterior sample (FPR = 1 - TNR)
        fprs_array = 1 - tnrs_array

        # For each posterior sample, compute partial ROC curve and AUC within overlapping range
        partial_auc_samples = []

        for post_idx in range(n_posteriors):
            fpr_curve = fprs_array[:, post_idx]
            tpr_curve = tprs_array[:, post_idx]

            # Filter points within partial FPR range
            valid_indices = (fpr_curve >= fpr_min) & (
                fpr_curve <= fpr_max
            )
            fpr_partial = fpr_curve[valid_indices]
            tpr_partial = tpr_curve[valid_indices]

            if len(fpr_partial) >= 2:
                # Sort by FPR for proper ROC curve
                sorted_indices = np.argsort(fpr_partial)
                fpr_sorted = fpr_partial[sorted_indices]
                tpr_sorted = tpr_partial[sorted_indices]

                # Compute partial AUC using trapezoidal rule (within observed points only)
                auc_val = auc(fpr_sorted, tpr_sorted)
                partial_auc_samples.append(auc_val)
            else:
                # Fallback to stored partial AUC if computation fails
                partial_auc_samples.append(
                    roc_results[setting]["partial_auc"]
                )

        # Compute partial AUC statistics
        partial_auc_samples = np.array(partial_auc_samples)
        partial_auc_mean = np.mean(partial_auc_samples)
        partial_auc_hdi = az.hdi(
            partial_auc_samples, hdi_prob=0.95
        )

        # Compute mean ROC curve within partial range
        mean_fpr = np.mean(fprs_array, axis=1)
        mean_tpr = np.mean(tprs_array, axis=1)

        # Filter mean curve to partial range and sort
        valid_mean_indices = (mean_fpr >= fpr_min) & (
            mean_fpr <= fpr_max
        )
        mean_fpr_sorted = np.sort(
            mean_fpr[valid_mean_indices]
        )
        mean_tpr_sorted = mean_tpr[valid_mean_indices][
            np.argsort(mean_fpr[valid_mean_indices])
        ]

        # Plot mean ROC curve within partial range
        if len(mean_fpr_sorted) > 0:
            # ensure no values outside [fpr_min, fpr_max]
            mean_fpr_sorted = np.clip(
                mean_fpr_sorted, fpr_min, fpr_max
            )
            ax.plot(
                mean_fpr_sorted,
                mean_tpr_sorted,
                "b-",
                linewidth=2.5,
                label="Mean ROC (Partial)",
            )

        # Compute 95% HDI band within partial range
        fpr_tpr_by_threshold = []
        for threshold_idx in range(len(thresholds)):
            fpr_samples = fprs_array[threshold_idx, :]
            tpr_samples = tprs_array[threshold_idx, :]
            for fpr, tpr in zip(fpr_samples, tpr_samples):
                if fpr_min <= fpr <= fpr_max:
                    fpr_tpr_by_threshold.append((fpr, tpr))

        # Create a dense grid of FPR values within partial range
        fpr_grid = np.linspace(fpr_min, fpr_max, 100)
        tpr_lower_interp = []
        tpr_upper_interp = []

        # For each FPR grid point, find which thresholds give FPR values nearby
        for fpr_target in fpr_grid:
            tpr_values_at_fpr = []

            # Find all threshold/posterior combinations with FPR close to target
            tolerance = 0.05
            for fpr_val, tpr_val in fpr_tpr_by_threshold:
                if abs(fpr_val - fpr_target) < tolerance:
                    tpr_values_at_fpr.append(tpr_val)

            # If we have TPR values at this FPR, compute HDI
            if len(tpr_values_at_fpr) > 0:
                tpr_hdi = az.hdi(
                    np.array(tpr_values_at_fpr),
                    hdi_prob=0.95,
                )
                tpr_lower_interp.append(tpr_hdi[0])
                tpr_upper_interp.append(tpr_hdi[1])
            else:
                # If no samples available, use the mean curve value or skip
                if len(tpr_lower_interp) > 0:
                    tpr_lower_interp.append(
                        tpr_lower_interp[-1]
                    )
                    tpr_upper_interp.append(
                        tpr_upper_interp[-1]
                    )

        # Fill between HDI bounds if we have interpolated values
        if len(tpr_lower_interp) > 0:
            # Remove any NaN values
            valid_idx = ~np.isnan(
                tpr_lower_interp
            ) & ~np.isnan(tpr_upper_interp)
            fpr_grid_valid = fpr_grid[valid_idx]
            tpr_lower_valid = np.array(tpr_lower_interp)[
                valid_idx
            ]
            tpr_upper_valid = np.array(tpr_upper_interp)[
                valid_idx
            ]

            ax.fill_between(
                fpr_grid_valid,
                tpr_lower_valid,
                tpr_upper_valid,
                alpha=0.25,
                color="blue",
                label="95% HDI (Partial)",
            )

        # Add diagonal reference line for random classifier (only within partial range)
        ax.plot(
            [fpr_min, fpr_max],
            [fpr_min, fpr_max],
            "k--",
            alpha=0.3,
            label="Random Classifier",
        )
        # restrict axes to partial range explicitly

        # Add partial AUC statistics text box
        ax.text(
            0.05,
            0.95,
            f"Partial AUC = {partial_auc_mean:.3f}\n95% HDI: [{partial_auc_hdi[0]:.3f}, {partial_auc_hdi[1]:.3f}]\n"
            + f"FPR Range: [{fpr_min:.3f}, {fpr_max:.3f}]",
            fontsize=legend_fs,
            bbox=dict(
                boxstyle="round",
                facecolor="wheat",
                alpha=0.7,
            ),
            verticalalignment="top",
        )

        # Configure subplot
        ax.set_xlabel(
            "FPR (1 - TNR)", fontsize=ax_label_fs
        )
        ax.set_ylabel("TPR", fontsize=ax_label_fs)
        ax.set_title(
            f"ROC Curve - Treatment {setting}\n(Partial Range with 95% HDI)",
            fontsize=title_fs,
        )
        ax.legend(fontsize=legend_fs, loc="lower right")
        ax.tick_params(labelsize=ax_tick_fs)
        ax.grid(True, alpha=0.3)
        # restrict x-axis exactly to observed overlap range
        ax.set_xlim([fpr_min, fpr_max])
        ax.set_ylim([-0.05, 1.05])
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close()
    return filename


def psa_result(
    path: str,
    n_readers_sim: list | None = None,
    priors_options: list | None = None,
    case_interaction: bool = False,
) -> plt.figure:
    """Load and return a PSA result figure from the specified path."""
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Specified path is not a directory: {path}"
        )

    simulation_results = read_psa_estimates_from_directory(
        path, case_interaction=case_interaction
    )
    metric = "abs_error"
    priors = [
        "diffuse",
        "weakly informative",
        "informative",
        "Classical linear model",
    ]
    if case_interaction:
        fig, ax = plt.subplots(
            1,
            len(priors),
            sharey=True,
            sharex=True,
            figsize=(16, 8),
            constrained_layout=True,
        )
        for i, prior in enumerate(priors):
            if prior == "Classical linear model":
                ax[i].set_title(
                    f"{prior}", fontsize=title_fs
                )
            ax[i].set_title(
                f"{prior.capitalize()} prior",
                fontsize=title_fs,
            )
            if prior == "Classical linear model":
                x, y, y_err = get_plot_values(
                    simulation_results,
                    "slope_freq",
                    metric,
                    "frequentist",
                    gamma=None,
                    case_interaction=case_interaction,
                )
            else:
                x, y, y_err = get_plot_values(
                    simulation_results,
                    "slope_bayes",
                    metric,
                    prior,
                    gamma=None,
                    case_interaction=case_interaction,
                )
            ax[i].errorbar(
                x,
                y.values,
                yerr=y_err,
                fmt="o",
                label=r"$prior=$" f"{prior}",
                capsize=5,
            )
            ax[i].set_xscale("log")
            ax[i].grid(True, c="lightgrey")
            ax[i].tick_params(labelsize=ax_tick_fs)
        fig.supylabel(
            r"Mean absolute error", fontsize=sup_ax_label_fs
        )
        fig.supxlabel(r"n", fontsize=sup_ax_label_fs)
        fig.set_facecolor("white")
        plt.show()
    else:
        gammas = [0.1, 0.2, 0.5]
        fig, ax = plt.subplots(
            len(gammas),
            len(priors),
            sharey=True,
            sharex=True,
            figsize=(16, 8),
            constrained_layout=True,
        )
        for i, gamma in enumerate(gammas):
            for j, prior in enumerate(priors):
                if (
                    i == 0
                    and prior == "Classical linear model"
                ):
                    ax[i, j].set_title(
                        f"{prior}", fontsize=title_fs
                    )
                elif i == 0:
                    ax[i, j].set_title(
                        f"{prior.capitalize()} prior",
                        fontsize=title_fs,
                    )
                if prior == "Classical linear model":
                    x, y, y_err = get_plot_values(
                        simulation_results,
                        "slope_freq",
                        metric,
                        "frequentist",
                        gamma=gamma,
                        case_interaction=case_interaction,
                    )
                else:
                    x, y, y_err = get_plot_values(
                        simulation_results,
                        "slope_bayes",
                        metric,
                        prior,
                        gamma=gamma,
                        case_interaction=case_interaction,
                    )
                ax[i, j].errorbar(
                    x,
                    y.values,
                    yerr=y_err,
                    fmt="o",
                    label=r"$prior=$" f"{prior}",
                    capsize=5,
                )
                ax[i, j].set_xscale("log")
                ax[i, j].grid(True, c="lightgrey")
                ax[i, -1].text(
                    1.05,
                    0.5,
                    r"$\gamma=$" f"{gamma}",
                    va="center",
                    ha="left",
                    fontsize=ax_label_fs,
                    transform=ax[i, -1].transAxes,
                )
                ax[i, j].tick_params(labelsize=ax_tick_fs)
        fig.supylabel(
            r"Mean absolute error", fontsize=sup_ax_label_fs
        )
        fig.supxlabel(r"n", fontsize=sup_ax_label_fs)
        fig.set_facecolor("white")
        plt.show()
    return fig
