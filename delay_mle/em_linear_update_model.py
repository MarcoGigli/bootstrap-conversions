import numpy as np
import pandas as pd
from lifelines.fitters import UnivariateFitter as LifelinesFitter
from sklearn.linear_model import LinearRegression, LogisticRegression
from lifelines import KaplanMeierFitter

# from delay_mle.kaplan_meier_weighted import KaplanMeierWeighted
from delay_mle.utils import _delay_model_weights


class KaplanMeierWeighted(KaplanMeierFitter):
    def _additive_var(self, population, deaths):
        return (deaths / (population * (population - deaths))).replace([np.inf], 0)


class EMLinearUpdateModel:
    def __init__(
        self,
        current_delay_model: LifelinesFitter | KaplanMeierWeighted,
        current_reward_model: LinearRegression,
        fit_intercept: bool = True
    ):
        self.current_delay_model = current_delay_model
        self.current_reward_model = current_reward_model
        self.updated_delay_model = None
        self.updated_reward_model = None
        self.fit_intercept = fit_intercept

    def fit(
        self,
        X: np.ndarray,
        observed_times: np.ndarray,
        rewards: np.ndarray
    ) -> "EMLinearUpdateModel":
        delay_dataset = self._prepare_delay_dataset(X, observed_times, rewards)
        self.updated_delay_model = self._fit_time_model(
            delay_dataset['observed_time'],
            delay_dataset['got_reward'],
            delay_dataset['weight']
        )

        X_reward, y_reward, weight_reward = self._prepare_reward_dataset(
            X, delay_dataset
        )
        self.updated_reward_model = self._fit_reward_model(
            X_reward, y_reward, weight_reward
        )

        return self

    @staticmethod
    def _fit_time_model(
        observed_times: np.ndarray,
        rewards: np.ndarray,
        weights: np.ndarray
    ):
        delay_model = KaplanMeierWeighted()
        delay_model.fit(observed_times, rewards, weights=weights)
        return delay_model

    def _fit_reward_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray
    ):
        # lr = LogisticRegression(penalty='none', fit_intercept=self.fit_intercept)
        lr = LinearRegression(fit_intercept=self.fit_intercept)
        lr.fit(X, y, sample_weight=weight)
        return lr

    def _prepare_delay_dataset(
        self,
        X: np.ndarray,
        observed_times: np.ndarray,
        rewards: np.ndarray
    ):
        df = pd.DataFrame(columns=['observed_time', 'got_reward'])
        df['observed_time'] = observed_times
        df['got_reward'] = rewards
        df['got_reward'] = df['got_reward'].astype(bool)

        survival = self.current_delay_model.survival_function_at_times(
            df['observed_time']
        ).values

        # reward_proba_estimate = self.current_reward_model.predict_proba(X)[:, 1]
        reward_proba_estimate = self.current_reward_model.predict(X)
        reward_proba_estimate = np.clip(reward_proba_estimate, 0., 1.)

        weights = _delay_model_weights(
            survival, reward_proba_estimate, df['got_reward'].values
        )
        df["weight"] = weights

        return df

    @staticmethod
    def _prepare_reward_dataset(
        X: np.ndarray,
        delay_dataset: pd.DataFrame
    ):
        rewarded_mask = delay_dataset["got_reward"].astype(int) == 1

        X_rewarded, df_rewarded = (
            X[rewarded_mask].copy(), delay_dataset[rewarded_mask].copy()
        )
        X_not_rewarded_pos, df_not_rewarded_pos = (
            X[~rewarded_mask].copy(), delay_dataset[~rewarded_mask].copy()
        )
        X_not_rewarded_neg, df_not_rewarded_neg = (
            X_not_rewarded_pos.copy(), df_not_rewarded_pos.copy()
        )

        df_rewarded["y"] = 1
        df_not_rewarded_pos["y"] = 1
        df_not_rewarded_neg["y"] = 0
        df_not_rewarded_neg["weight"] = 1 - df_not_rewarded_neg["weight"]

        X_full = np.concatenate([X_rewarded, X_not_rewarded_pos, X_not_rewarded_neg])
        df_full = pd.concat(
            [df_rewarded, df_not_rewarded_pos, df_not_rewarded_neg],
            ignore_index=True
        )

        return X_full, df_full["y"].values, np.clip(df_full["weight"].values, 0, 1)



