import numpy as np
import pandas as pd


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
