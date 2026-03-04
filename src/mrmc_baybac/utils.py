import numpy as np


def invlogit(x):
        return 1 / (1 + np.exp(-x))


def compute_posterior_effect_size(
        a_sample, b_sample
    ):
        baseline_accuracy, treatment_accuracy = compute_posterior_accuracy_by_treatment(a_sample, b_sample)
        return treatment_accuracy - baseline_accuracy


def compute_posterior_accuracy_by_treatment(a_sample, b_sample):
        baseline_accuracy = invlogit(a_sample)
        treatment_accuracy = invlogit(a_sample + b_sample)
        return baseline_accuracy, treatment_accuracy
