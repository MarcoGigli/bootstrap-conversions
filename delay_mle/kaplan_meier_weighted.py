import pandas as pd
import numpy as np
import numpy.typing as npt

from lifelines.fitters import UnivariateFitter as LifelinesFitter


class KaplanMeierWeighted:
    series_name = 'KM_estimate'

    def __init__(self):
        self.durations = None
        self.event_observed = None
        self.survival_function_ = None
        self.cumulative_density_ = None

    @staticmethod
    def convert_to_arrays(
        durations: npt.ArrayLike,
        event_observed: npt.ArrayLike,
        weights: npt.ArrayLike | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        durations = np.array(durations)

        event_observed = np.array(event_observed)
        event_observed = event_observed.astype(bool)

        if weights is None:
            weights = np.ones(durations.shape[0])
        weights = np.array(weights)

        return durations, event_observed, weights

    @staticmethod
    def unique_sorted_times(durations: np.ndarray) -> np.ndarray:
        times = np.unique(durations)
        times.sort()
        return times

    @staticmethod
    def calculate_at_risk(
        durations: np.ndarray,
        weights: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        at_risk_numbers = []
        for time in times:
            is_still_alive = (durations >= time).astype(int)
            weighted_still_alive = is_still_alive * weights
            at_risk_number = weighted_still_alive.sum()
            at_risk_numbers.append(at_risk_number)
        return np.array(at_risk_numbers)

    @staticmethod
    def calculate_deaths(
        durations: np.ndarray,
        event_observed: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        death_numbers = []
        for time in times:
            just_dead = event_observed & (durations == time)
            just_dead = just_dead.astype(int)
            death_numbers.append(just_dead.sum())
        return np.array(death_numbers)

    @staticmethod
    def calculate_hazards(
        durations: np.ndarray,
        event_observed: np.ndarray,
        weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        times = KaplanMeierWeighted.unique_sorted_times(durations)

        at_risk = KaplanMeierWeighted.calculate_at_risk(durations, weights, times)
        deaths = KaplanMeierWeighted.calculate_deaths(durations, event_observed, times)

        hazards = deaths / at_risk
        return times, hazards

    @staticmethod
    def calculate_survival(
        hazards: np.ndarray
    ) -> np.ndarray:
        print(1 - hazards)
        survival = np.exp(np.log(1 - hazards).cumsum())
        return survival

    def save_survival_and_cumulative(self, times: np.ndarray, survival: np.ndarray):
        survival = np.array([1.0] + list(survival))
        idx = np.array([0.0] + list(times))

        survival_df = pd.DataFrame(columns=[self.series_name])
        survival_df[self.series_name] = survival
        survival_df.index = idx.astype(float)
        survival_df.index.name = 'timeline'

        self.survival_function_ = survival_df
        self.cumulative_density_ = 1 - self.survival_function_

    def fit(
        self,
        durations: npt.ArrayLike,
        event_observed: npt.ArrayLike,
        weights: npt.ArrayLike | None = None
    ) -> 'KaplanMeierWeighted':
        durations, event_observed, weights = self.convert_to_arrays(
            durations, event_observed, weights
        )
        self.event_observed = event_observed
        self.durations = durations
        times, hazards = self.calculate_hazards(
            durations, event_observed, weights
        )
        survival = self.calculate_survival(hazards)
        self.save_survival_and_cumulative(times, survival)
        return self

    @staticmethod
    def interpolate(time: float, time_series: pd.Series) -> float:
        unique_times = time_series.index.values
        insertion_index = np.searchsorted(unique_times, time)
        if insertion_index == 0:
            return time_series.values[0]
        if insertion_index == time_series.shape[0]:
            return time_series.values[-1]

        left = time_series.values[insertion_index - 1]
        left_time = unique_times[insertion_index - 1]
        right = time_series.values[insertion_index]
        right_time = unique_times[insertion_index]
        prefactor = (time - left_time) / (right_time - left_time)
        return left + prefactor * (right - left)

    @staticmethod
    def left_value(time: float, time_series: pd.Series) -> float:
        unique_times = time_series.index.values
        insertion_index = np.searchsorted(unique_times, time)
        # assert insertion_index > 0
        left = time_series.values[max(insertion_index - 1, 0)]
        return left

    def survival_function_at_times(
        self,
        times: npt.ArrayLike
    ) -> pd.Series:
        survivals = []
        for time in times:
            survival = self.left_value(time, self.survival_function_[self.series_name])
            survivals.append(survival)
        return pd.Series(survivals, index=list(times))

    def cumulative_density_at_times(
        self,
        times: npt.ArrayLike
    ) -> pd.Series:
        return 1 - self.survival_function_at_times(times)


def check_times_for_density(times: npt.ArrayLike, ref_times: np.ndarray) -> bool:
    is_close_matrix = np.isclose(times, ref_times.reshape(-1, 1))
    at_least_one_close = is_close_matrix.max(axis=0)
    return all(at_least_one_close)


def density_at_times(
    time_model: LifelinesFitter | KaplanMeierWeighted,
    times: npt.ArrayLike,
    series_name: str
) -> pd.Series:
    try:
        return time_model.density_at_times(times)
    except AttributeError:
        unique_times = time_model.cumulative_density_.index.values
        assert check_times_for_density(times, unique_times)
        right_value = time_model.cumulative_density_at_times(times + 10 ** -6).values
        left_value = time_model.cumulative_density_at_times(times - 10 ** -6).values
        density = right_value - left_value
        return pd.Series(density, index=list(times), name=series_name)