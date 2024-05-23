import numpy as np
from scipy.stats import pareto
from scipy.stats import geom

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
tau_list = [
    100,
    # 500,
    # 1000
]  # Averaged delay (geometric law)
pareto_alpha_list = [0.2, 0.5, 0.8, 1.0]

artificial_history_sizes = [
    1,
    10,
    100,
    1000
]

delay_distributions = {
    "geom": {
        "distribution": geom,
        "parameter_list": tau_list
    },
    # "pareto": {
    #     "distribution": pareto,
    #     "parameter_list": pareto_alpha_list
    # }
}

timeout_list = [
    # 100,
    500
]  # limit for accepting observations

for artificial_history_size in artificial_history_sizes:
    for distribution_name, distribution in delay_distributions.items():
        for is_censored in [True, False]:
            for param in distribution["parameter_list"]:
                for timeout in timeout_list:
                    # Policies to evaluate

                    bootstrap = PolicyBootstrapDeLinTS(
                        T, timeout, d, prior_num_points=artificial_history_size,
                        is_censored=is_censored
                    )

                    policies = [
                        bootstrap,
                        PolicyDeLinUCB(T, timeout, lambda_reg, delta, alpha,
                                       bias_term=True),
                        PolicyDeLinTS(T, timeout, 0.5)
                    ]

                    folder_name = f"randomtheta_{distribution_name}_{param}_{timeout}_ah{artificial_history_size}"

                    if distribution_name == "geom":
                        actual_param = 1 / param
                    else:
                        actual_param = param

                    simulator = SimulatorDelayedReward(
                        None, policies, K, d, timeout, distribution["distribution"],
                        actual_param, folder_name=folder_name,
                        reset_theta=True
                    )

                    if is_censored:
                        results = simulator.run_cens(T, N, tsav, verbose)
                    else:
                        results = simulator.run_uncens(T, N, tsav, verbose)

                    cens_fname = "cens" if is_censored else "uncens"
                    regret_filename = f"cumreg_randomtheta_{distribution_name}_{cens_fname}_"
                    regret_filename += f"N{N}_param{param}_timeout{timeout}_ah{artificial_history_size}"

                    plot_regret(results, fname=regret_filename)
