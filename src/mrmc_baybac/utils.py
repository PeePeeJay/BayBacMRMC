import numpy as np


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
