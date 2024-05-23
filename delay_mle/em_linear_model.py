import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from lifelines import KaplanMeierFitter
import warnings

# from delay_mle.kaplan_meier_weighted import KaplanMeierWeighted
from delay_mle.em_linear_update_model import EMLinearUpdateModel
from delay_mle.unit_sphere_regression import UnitSphereRegression


class KaplanMeierWeighted(KaplanMeierFitter):
    def _additive_var(self, population, deaths):
        return (deaths / (population * (population - deaths))).replace([np.inf], 0)


class EMLinearModel:
    def __init__(
        self,
        max_steps:
        int = 200,
        tol: float = 10**-3,
        fit_intercept: bool = True,
        how: str = 'linear'
    ):
        self.max_steps = max_steps
        self.tol = tol
        self.reward_model = None
        self.delay_model = None
        self.update_model = None
        self.fit_intercept = fit_intercept

        assert how in ['linear', 'unit_sphere']
        self.how = how

    def fit(
        self,
        X: np.ndarray,
        observed_times: np.ndarray,
        rewards: np.ndarray
    ) -> "EMLinearModel":

        delay_model = KaplanMeierWeighted()
        delay_model.fit(observed_times, rewards)

        # reward_model = LogisticRegression(penalty='none', fit_intercept=self.fit_intercept)
        if self.how == 'unit_sphere':
            reward_model = UnitSphereRegression()
        else:
            reward_model = LinearRegression(fit_intercept=self.fit_intercept)
        reward_model.fit(X, rewards)

        for i in range(self.max_steps):
            # print('\t\t', i)
            model = EMLinearUpdateModel(
                delay_model, reward_model, fit_intercept=self.fit_intercept
            )
            model.fit(X, observed_times, rewards)

            old_reward_model, reward_model = reward_model, model.updated_reward_model
            old_delay_model, delay_model = delay_model, model.updated_delay_model

            self.reward_model = model.updated_reward_model
            self.delay_model = model.updated_delay_model
            self.update_model = model

            if self._check_saturation(
                old_reward_model, reward_model,
                old_delay_model, delay_model,
                X, observed_times
            ):
                break
        else:
            warnings.warn("EMLinearModel: Max iterations reached")

        return self

    def predict_proba(self, X: np.ndarray, t: np.ndarray):
        return self._predict_proba(X, t, self.reward_model, self.delay_model)

    @staticmethod
    def _predict_proba(X, t, reward_model, delay_model):
        # reward_proba = reward_model.predict_proba(X)[:, 1]
        reward_proba = reward_model.predict(X)
        reward_proba = np.clip(reward_proba, 0, 1)
        cumulative_proba = 1 - delay_model.survival_function_at_times(t)
        return reward_proba * cumulative_proba

    def _check_saturation(
        self,
        old_reward_model,
        reward_model,
        old_delay_model,
        delay_model,
        X,
        observed_times
    ):
        old_proba = self._predict_proba(X, observed_times, old_reward_model, old_delay_model)
        new_proba = self._predict_proba(X, observed_times, reward_model, delay_model)
        return np.abs(old_proba - new_proba).max() < self.tol


