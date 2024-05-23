import pandas as pd
import numpy as np
from lifelines.fitters import UnivariateFitter as LifelinesFitter
from lifelines import KaplanMeierFitter

from delay_mle.kaplan_meier_weighted import KaplanMeierWeighted, density_at_times
from delay_mle.utils import prob_observing_reward, log_likelihood, _delay_model_weights


class EMUpdateModel:
    def __init__(
        self,
        current_delay_model: LifelinesFitter | KaplanMeierWeighted,
        current_reward_estimate: float
    ):
        self.current_delay_model = current_delay_model
        self.current_reward_estimate = current_reward_estimate
        self.updated_delay_model = None
        self.updated_reward_estimate = None

    def fit(self, observed_times: np.ndarray, rewards: np.ndarray):
        dataset = self._prepare_dataset(observed_times, rewards)
        self.updated_delay_model = self._fit_time_model(
            dataset['observed_time'], dataset['got_reward'], dataset['weight']
        )
        self.updated_reward_estimate = self._fit_p_reward(
            dataset['got_reward'], dataset['weight']
        )
        return self

    def _prepare_dataset(self, observed_times: np.ndarray, rewards: np.ndarray):
        df = pd.DataFrame(columns=['observed_time', 'got_reward'])
        df['observed_time'] = observed_times
        df['got_reward'] = rewards
        df['got_reward'] = df['got_reward'].astype(bool)

        survival = self.current_delay_model.survival_function_at_times(
            df['observed_time']
        ).values
        weights = _delay_model_weights(
            survival, self.current_reward_estimate, df['got_reward']
        )
        df['weight'] = weights
        return df

    @staticmethod
    def _fit_time_model(
        observed_times: np.ndarray,
        rewards: np.ndarray,
        weights: np.ndarray
    ):
        delay_model = KaplanMeierFitter()  # KaplanMeierWeighted()
        delay_model.fit(observed_times, rewards, weights=weights)
        return delay_model

    @staticmethod
    def _fit_p_reward(
        rewards: np.ndarray,
        weights: np.ndarray
    ):
        return weights.mean()

    def log_likelihood(self):
        density = density_at_times(
            self.updated_delay_model,
            self.updated_delay_model.durations,
            'density_label'  # TODO: useless, modify
        )
        density = density.values

        survival = self.updated_delay_model.survival_function_at_times(
            self.updated_delay_model.durations
        )
        survival = survival.values

        q = prob_observing_reward(survival, self.updated_reward_estimate)

        return log_likelihood(
            survival, density,
            self.updated_reward_estimate,
            self.updated_delay_model.event_observed,
            q
        )


