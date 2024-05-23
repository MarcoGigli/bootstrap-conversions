import numpy as np
from scipy.stats import t as student_t

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
scales = [
    1,
    100,
    500
]
degrees_of_freedom_list = [
    1,
    10,
    100,
    1000
]


class AbsStudentSampler:
    def __init__(self, degrees_of_freedom, scale):
        self.degrees_of_freedom = degrees_of_freedom
        self.scale = scale

    def rvs(self, size):
        sample = student_t.rvs(self.degrees_of_freedom, scale=scale, size=size)
        sample = np.abs(sample)
        sample = np.round(sample).astype(int)
        return sample


timeout_list = [100, 500]  # limit for accepting observations

for is_censored in [True, False]:
    for dof in degrees_of_freedom_list:
        for scale in scales:
            for timeout in timeout_list:
                # Policies to evaluate

                bootstrap = PolicyBootstrapDeLinTS(
                    T, timeout, d, prior_num_points=10, is_censored=is_censored
                )

                policies = [
                    bootstrap,
                    PolicyDeLinUCB(T, timeout, lambda_reg, delta, alpha,
                                   bias_term=True),
                    PolicyDeLinTS(T, timeout, 0.5)
                ]

                folder_name = f"student_{dof}_{scale}_{timeout}"

                distribution_instance = AbsStudentSampler(dof, scale)

                simulator = SimulatorDelayedReward(
                    theta, policies, K, d, timeout,
                    distribution_instance, "placeholder", empirical=True,
                    folder_name=folder_name)

                if is_censored:
                    results = simulator.run_cens(T, N, tsav, verbose)
                else:
                    results = simulator.run_uncens(T, N, tsav, verbose)

                cens_fname = "cens" if is_censored else "uncens"
                regret_filename = f"cumreg_student_{cens_fname}_"
                regret_filename += f"N{N}_param_{dof}_{scale}_timeout{timeout}"

                plot_regret(results, fname=regret_filename)
