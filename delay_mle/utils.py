import numpy as np


def safe_times_log(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a * log(b), but manages when a = b = 0"""
    eps = 10**-6
    indeterminate_form_mask = np.nonzero(
        (b == 0) & (a == 0)
    )
    b = np.copy(b)
    b[indeterminate_form_mask] += eps
    return a * np.log(b)


def _non_censored_survival_log_likelihood(density_at_durations, event_observed) -> np.ndarray:
    non_censored_part = safe_times_log(
        event_observed,
        density_at_durations
    )
    return non_censored_part


def prob_observing_reward(
    survival: np.ndarray,
    reward_proba_estimate: float | np.ndarray,
):
    q = reward_proba_estimate * survival
    q = q / (1 - reward_proba_estimate * (1 - survival))
    return q


def _delay_model_weights(
    survival: np.ndarray,
    reward_proba_estimate: float | np.ndarray,
    is_observed: np.ndarray
) -> np.ndarray:
    weights = prob_observing_reward(survival, reward_proba_estimate)
    is_observed = is_observed.astype(int)
    # weights = is_observed + (1 - is_observed) * weights  # <- problems when weights are nan
    weights = np.where(is_observed == 1, is_observed, weights)
    return weights


def _censored_survival_log_likelihood(
    survival_at_durations, event_observed, weights
) -> np.ndarray:
    censored_part = (1 - event_observed)
    censored_part = censored_part * weights
    censored_part = censored_part * np.log(survival_at_durations)
    return censored_part


def survival_log_likelihood(
    survival_at_durations, density_at_durations, event_observed, weights
):
    non_censored_part = _non_censored_survival_log_likelihood(
        density_at_durations, event_observed
    )
    censored_part = _censored_survival_log_likelihood(
        survival_at_durations, event_observed, weights
    )
    return np.sum(non_censored_part) + np.sum(censored_part)


def classifier_log_likelihood(
    reward_probability, event_observed, weights
):
    non_censored_part = np.log(reward_probability)
    non_censored_part = non_censored_part * (event_observed + (1 - event_observed) * weights)
    censored_part = np.log(1 - reward_probability)
    censored_part = censored_part * (1 - event_observed) * (1 - weights)
    return np.sum(non_censored_part) + np.sum(censored_part)


def log_likelihood(
    survival_at_durations, density_at_durations,
    reward_probability, event_observed, weights
):
    return (
        survival_log_likelihood(survival_at_durations, density_at_durations, event_observed, weights) +
        classifier_log_likelihood(reward_probability, event_observed, weights)
    ).sum()