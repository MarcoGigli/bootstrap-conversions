import numpy as np
import pandas as pd

from delay_mle.em_linear_model import EMLinearModel

from linear_delayed.policy import Policy


class PolicyBootstrapDeLinTS(Policy):

    def __init__(
        self,
        T: int,
        m: int,
        dim: int,
        prior_num_points: int = 1,
        is_censored: bool = True,
        prior_method: str = 'interval',
        how: str = 'linear'
    ):
        super().__init__(T, m)
        self.dim = dim  # TODO: in super init
        self.t = 0  # TODO: in super init
        self.prior_num_points = prior_num_points
        self.is_censored = is_censored

        assert prior_method in ['interval', 'sphere']
        self.prior_method = prior_method

        assert how in ['linear', 'unit_sphere']
        self.how = how

        self.db: list[dict] = []

    def init(self, dim: int):
        self.t = 0
        self.db = []

    def extract_prior_delays(self):
        possible_delays = np.arange(0, self.m)
        # TODO: seed
        delays = np.random.choice(possible_delays, size=self.prior_num_points)
        return delays

    def extract_uniform_interval(self):
        shape = (self.prior_num_points, self.dim)
        a = np.random.uniform(0, 1, size=shape)
        theta = np.random.uniform(0, 1, size=shape)

        a = a / np.linalg.norm(a, axis=1).reshape(-1, 1)
        theta = theta / np.linalg.norm(theta, axis=1).reshape(-1, 1)

        return a, theta

    def extract_uniform_sphere(self):
        shape = (self.prior_num_points, self.dim)
        a = np.random.normal(size=shape)
        theta = np.random.normal(size=shape)

        a = a / np.linalg.norm(a, axis=1).reshape(-1, 1)
        theta = theta / np.linalg.norm(theta, axis=1).reshape(-1, 1)

        a, theta = np.abs(a), np.abs(theta)

        return a, theta

    def extract_prior_data(self):
        delays = self.extract_prior_delays()

        if self.prior_method == 'interval':
            a, theta = self.extract_uniform_interval()
        else:
            a, theta = self.extract_uniform_sphere()

        means = (a * theta).sum(axis=1)

        assert all(means <= 1)
        assert all(means >= 0)
        assert all(delays >= 0)

        rewards = np.random.binomial(1, means)

        return delays, a, rewards

    def build_dataset(self):
        arms, df = self.build_played_dataset()

        prior_delays, prior_a, prior_rewards = self.extract_prior_data()
        prior_array = np.concatenate(
            [prior_delays.reshape(-1, 1), prior_rewards.reshape(-1, 1)],
            axis=1
        )
        prior_df = pd.DataFrame(prior_array, columns=["delay", "reward"])

        df = pd.concat([prior_df, df], axis=0, ignore_index=True)
        arms = np.concatenate([prior_a, arms], axis=0)

        return df["delay"].values, arms, df["reward"].values

    def build_played_dataset(self):
        arms = []
        delays = []
        rewards = []
        times = []
        for t, row in enumerate(self.db):
            arms.append(row["features"])
            delays.append(row.get("delay"))
            rewards.append(row["reward"])
            times.append(t)

        df = pd.DataFrame(columns=["time", "delay", "reward"])
        df["time"] = times
        df["delay"] = delays
        df["reward"] = rewards

        if self.is_censored:
            df["delay"].fillna(np.minimum(self.t - df["time"], self.m), inplace=True)
        else:
            df["delay"].fillna(self.t - df["time"], inplace=True)
        df = df[["delay", "reward"]]

        arms = np.array(arms)
        return arms, df

    @staticmethod
    def simple_bootstrap(arrays: list[np.ndarray]) -> list[np.ndarray]:
        assert all([arr.shape[0] == arrays[0].shape[0] for arr in arrays])

        n = arrays[0].shape[0]

        original_indices = np.arange(0, n)
        # TODO: seed
        sampled_indices = np.random.choice(original_indices, size=n)

        return [arr[sampled_indices] for arr in arrays]

    def selectArm(self, arms):
        best_arm_index = self._select_arm(arms)
        self._update_db(arms, best_arm_index)
        return best_arm_index

    def _select_arm(self, arms):
        if self.t == 0:
            return np.random.choice(np.arange(arms.shape[0]))

        played_arms, delays, rewards = self.build_bootstrapped_dataset()

        # If every reward is 0 or 1, there's nothing to learn: all actions seem equivalent
        if all([reward == rewards[0] for reward in rewards]):
            return np.random.choice(np.arange(arms.shape[0]))

        model = EMLinearModel(fit_intercept=False, how=self.how)
        model.fit(played_arms, delays, rewards)

        evaluation_times = np.array([min(self.T, self.m)] * arms.shape[0])
        scores = model.predict_proba(arms, evaluation_times)
        best_arm_index = np.argmax(scores)
        return best_arm_index

    def build_bootstrapped_dataset(self):
        delays, played_arms, rewards = self.build_dataset()
        delays, played_arms, rewards = self.simple_bootstrap(
            [delays, played_arms, rewards]
        )
        return played_arms, delays, rewards

    def _update_db(self, arms, best_arm_index):
        self.db.append(
            {"features": arms[best_arm_index], "reward": 0}
        )

    def updateState(self, disclosure):
        for feedback in disclosure:
            delay, reward, features = feedback

            assert reward == 1

            corresponding_row = self.db[self.t - delay]
            assert np.abs((corresponding_row["features"] - features)).max() < 10**-5

            corresponding_row["reward"] = reward
            corresponding_row["delay"] = delay

        self.t += 1

    @staticmethod
    def id():
        return 'BootstrapLinTS'

    def __str__(self):
        return 'BootstrapLinTS'

