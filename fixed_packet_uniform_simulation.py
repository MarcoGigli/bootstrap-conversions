import numpy as np

from linear_delayed.de_lin_ts import PolicyDeLinTS
from linear_delayed.de_lin_ucb import PolicyDeLinUCB
from linear_delayed.de_bootstrap_lin_ts import PolicyBootstrapDeLinTS
from linear_delayed.simulator_delayed_reward import SimulatorDelayedReward
from linear_delayed.plot import plot_regret

import warnings
from lifelines.exceptions import StatisticalWarning


# Filter out a lifelines warning regarding weights:
# lifelines expects integer weights, but it works
# with float weights too
warnings.filterwarnings("ignore", category=StatisticalWarning)

# Filter out a lifelines warning: it doesn't matter,
# as the warning is spawned by calculation of confidence
# bounds, which I don't use.
warnings.filterwarnings(
    action='ignore',
    message='overflow encountered in exp'
)


d = 5  # dimension of R^n
K = 10  # number of arms

theta = np.ones(d)
theta = theta / np.linalg.norm(theta)

# Confidence shrinkage (unfair comparison with TS if <1)
alpha = 1.

T = 3000  # Finite Horizon
N = 20  # 100  # Monte Carlo simulations

delta = 0.1
lambda_reg = 1.

# save subsampled points for Figures
Nsub = 100
tsav = range(2, T, Nsub)
L = len(tsav)

verbose = True

## PARAMETERS
fixed_delays_list = [250, 500, 1000]
packet_loss_prob_list = [0.25, 0.5, 0.75]
uniform_bounds_list = [[150, 300]]


class FixedSampler:  # T = 3000 (it's 20K in Wu et al, but seems overkill)
    def __init__(self, delay):
        self.delay = delay

    def rvs(self, size):
        return np.array([self.delay] * size)


from scipy.stats import binom
class PacketLossSampler:  # T = 3000 (it's 10K in Wu et al, but seems overkill)
    def __init__(self, prob, max_time=T):
        self.prob = prob
        self.max_time = max_time

    def rvs(self, size):
        is_lost = binom.rvs(1, self.prob, size=size)
        return is_lost * 10 * self.max_time


class UniformSampler:  # T = 3000 (it's 20K in Wu et al, but seems overkill)
    def __init__(self, bounds):
        left_bound, right_bound = bounds
        self.left_bound = int(left_bound)
        self.right_bound = int(round(right_bound))

    def rvs(self, size):
        possible_delays = np.arange(self.left_bound, self.right_bound)
        return np.random.choice(possible_delays, size=size)


delay_distributions = {
    "fixed": {
        "distribution": FixedSampler,
        "parameter_list": fixed_delays_list
    },
    "packet_loss": {
        "distribution": PacketLossSampler,
        "parameter_list": packet_loss_prob_list
    },
    "uniform": {
        "distribution": UniformSampler,
        "parameter_list": uniform_bounds_list
    }
}

timeout_list = [100, 500]  # limit for accepting observations

for distribution_name, distribution in delay_distributions.items():
    for is_censored in [True, False]:
        for param in distribution["parameter_list"]:
            for timeout in timeout_list:
                # Policies to evaluate

                bootstrap = PolicyBootstrapDeLinTS(
                    T, timeout, d, prior_num_points=10, is_censored=is_censored
                )

                policies = [
                    bootstrap,
                    PolicyDeLinUCB(
                        T, timeout, lambda_reg, delta, alpha, bias_term=True
                    ),
                    PolicyDeLinTS(T, timeout, 0.5)
                ]

                try:
                    param_str = f"{param[0]}-{param[1]}"
                except TypeError:
                    param_str = str(param)

                folder_name = f"{distribution_name}_{param_str}_{timeout}"

                distribution_instance = distribution["distribution"](param)

                simulator = SimulatorDelayedReward(
                    theta, policies, K, d, timeout,
                    distribution_instance, "placeholder", empirical=True,
                    folder_name=folder_name
                )

                if is_censored:
                    results = simulator.run_cens(T, N, tsav, verbose)
                else:
                    results = simulator.run_uncens(T, N, tsav, verbose)

                cens_fname = "cens" if is_censored else "uncens"
                regret_filename = f"cumreg_{distribution_name}_{cens_fname}_"
                regret_filename += f"N{N}_param{param_str}_timeout{timeout}"

                plot_regret(results, fname=regret_filename)
