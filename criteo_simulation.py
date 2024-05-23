import numpy as np
import pandas as pd

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


class EmpiricalSampler:

    def __init__(self, delays, probs):
        self.delays = delays
        self.probs = probs

    def rvs(self, size):
        return np.random.choice(a=self.delays, p=self.probs, size=size)


criteo_folder = "criteo_distribution"
log_delay_values = pd.read_csv(f'{criteo_folder}/emp_dist_delay_log_values', header=None)
probabilities = pd.read_csv(f'{criteo_folder}/emp_dist_delay', header=None)

delay_values = log_delay_values.apply(lambda x: 10**x)

delays = delay_values.values.squeeze().astype(int)
probs = probabilities.values.squeeze()
probs = probs/np.sum(probs)

delays = delays / 100  # rescaling for experiment
cdf = np.cumsum(probs)

emp_pdf = EmpiricalSampler(delays.astype(int), probs)


d = 5  # dimension of R^n
K = 10  # number of arms

theta = np.ones(d)
theta = theta / np.linalg.norm(theta)

# Confidence shrinkage (unfair comparison with TS if <1)
alpha = 1.

T = 10000  # Finite Horizon
N = 20  # 100  # Monte Carlo simulations

delta = 0.1
lambda_reg = 1.

# save subsampled points for Figures
Nsub = 100
tsav = range(2, T, Nsub)
L = len(tsav)

verbose = True

timeout_list = [2000]  # limit for accepting observations

for is_censored in [True, False]:
    for timeout in timeout_list:
        # Policies to evaluate

        bootstrap = PolicyBootstrapDeLinTS(
            T, timeout, d, prior_num_points=10, is_censored=is_censored
        )

        policies = [
            bootstrap,
            PolicyDeLinUCB(T, timeout, lambda_reg, delta, alpha, bias_term=True),
            PolicyDeLinTS(T, timeout, 1)
        ]

        folder_name = f"criteo_{timeout}"

        simulator = SimulatorDelayedReward(
            theta, policies, K, d, timeout, emp_pdf, 1, empirical=True,
            folder_name=folder_name
        )

        if is_censored:
            results = simulator.run_cens(T, N, tsav, verbose)
        else:
            results = simulator.run_uncens(T, N, tsav, verbose)

        cens_fname = "cens" if is_censored else "uncens"
        regret_filename = f"cumreg_criteo_{cens_fname}_"
        regret_filename += f"N{N}_timeout{timeout}"

        plot_regret(results, fname=regret_filename)
