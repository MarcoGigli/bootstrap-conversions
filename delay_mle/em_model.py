import numpy as np

from delay_mle.em_update_model import EMUpdateModel
from delay_mle.kaplan_meier_weighted import KaplanMeierWeighted


class ExpectationMaximizationModel:
    def __init__(self, max_steps: int = 200, tol: float = 10**-5):
        self.max_steps = max_steps
        self.tol = tol
        self.reward_estimate = None
        self.delay_model = None
        self.update_model = None
        
    def fit(
        self,
        observed_times: np.ndarray,
        rewards: np.ndarray,
        p_reward: float | None = None
    ):
        delay_model = KaplanMeierWeighted()
        delay_model.fit(observed_times, rewards)

        if p_reward is None:
            p_reward = rewards.mean()

        for i in range(self.max_steps):
            model = EMUpdateModel(delay_model, p_reward)
            model.fit(observed_times, rewards)
            old_p_reward, p_reward = p_reward, model.updated_reward_estimate
            delay_model = model.updated_delay_model

            self.reward_estimate = model.updated_reward_estimate
            self.delay_model = model.updated_delay_model
            self.update_model = model

            if np.abs(old_p_reward - p_reward) < self.tol:
                break
        return self

    def log_likelihood(self):
        return self.update_model.log_likelihood()
