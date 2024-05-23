###### IMPORTANT ######
# This code comes from
# Vernade et al. - Linear bandits with Stochastic Delayed Feedback - 2020
# with minimal refactoring.
# It is attached for reproducibility of the comparison

import numpy as np
from numpy.linalg import norm
from scipy.stats import bernoulli
import pandas as pd
import os
import sys
import tqdm
import pickle

verbose = True

folder = 'data_and_figures'
if not os.path.exists(folder):
    os.mkdir(folder)
subfolder = 'regret'
if not os.path.exists(f'{folder}/{subfolder}'):
    os.mkdir(f'{folder}/{subfolder}')

subfolder_pickle = 'pickle'
if not os.path.exists(f'{folder}/{subfolder_pickle}'):
    os.mkdir(f'{folder}/{subfolder_pickle}')


class SimulatorDelayedReward():
    """
    Simulator of stochastic delayed games.

    Params:
    -------
    MAB: list
        List of arms.

    policies: list
        List of policies to test.

    K: int
        Number of items (arms) in the pool.

    d: int
        Dimesion of the problem

    tau: integer
        Expected delay (geometric distribution)

    delays : an array of size T containing the independent delays

    timeout: integer
        longest time the learner accepts to wait for feedback
    """

    def __init__(
        self, theta, policies, arms_number, action_space_dimension,
        timeout, delay_pdf, param, empirical=False, folder_name=None,
        reset_theta=False
    ):
        self.theta = theta
        self.policies = policies
        self.m = timeout
        self.action_space_dimension = action_space_dimension
        self.arms_number = arms_number
        self.delay_pdf = delay_pdf
        self.param = param
        self.verbose = verbose
        self.empirical = empirical
        self.folder_name = folder_name
        self.pickle_base_path = f'{folder}/{subfolder_pickle}/'
        if self.folder_name is not None:
            self.pickle_base_path += self.folder_name + '/'

        if not os.path.exists(self.pickle_base_path):
            os.mkdir(self.pickle_base_path)

        self.reset_theta = reset_theta

        # self.tau_est = dict()

    def get_gaussian_arms(self):
        """returns K normalised vectors sampled uniformly at random in the unit ball"""

        arms = np.random.multivariate_normal(
            np.zeros(self.action_space_dimension),
            np.identity(self.action_space_dimension),
            size=self.arms_number
        )
        # arms has shape (self.arms_number, self.action_space_dimension)

        normalized_arms = self.normalize(arms)

        return normalized_arms

    def get_uniform_arms(self):
        arms = np.random.uniform(
            0, 1,
            size=(self.arms_number, self.action_space_dimension)
        )
        normalized_arms = self.normalize(arms)
        return normalized_arms

    @staticmethod
    def normalize(arms):
        # each arm gets divided by its norm:
        normalized_arms = arms / norm(arms, axis=1)[:, np.newaxis]
        return normalized_arms

    def get_best_arm_reward(self, arms):
        """Return the indices of the best arm"""
        means = self.get_expected_reward(arms)
        best_arm = np.argmax(means)
        best_reward = means[best_arm]
        return best_arm, best_reward

    def get_expected_reward(self, arms):
        """returns the expected payoff of the contextualized arm armIndex"""
        return np.inner(arms, self.theta)

    def play(self, arm):
        """Play arms and return the corresponding rewards """
        mean = np.dot(self.theta, arm)
        reward = bernoulli(mean).rvs()
        return reward, arm

    def run(self, time_horizon, num_simulations, times_to_save, verbose, is_censored):

        policy_to_results = dict()

        for policy in self.policies:
            cum_regret, policy_name = self.run_policy(num_simulations, time_horizon, is_censored, policy)

            subsampled_cum_regret = cum_regret[times_to_save, :]

            policy_to_results[policy_name] = pd.DataFrame(
                subsampled_cum_regret, index=times_to_save
            )

            policy_to_results[policy_name]['avg'] = np.mean(subsampled_cum_regret, axis=1)
            policy_to_results[policy_name]['qregret'] = np.percentile(subsampled_cum_regret, 5, axis=1)
            policy_to_results[policy_name]['Qregret'] = np.percentile(subsampled_cum_regret, 95, axis=1)

        return policy_to_results

    def run_policy(self, num_simulations, time_horizon, is_censored, policy):
        policy_name = policy.__str__()

        regret = np.zeros((time_horizon, num_simulations))
        cum_regret = np.zeros((time_horizon, num_simulations))

        for sim_index in range(num_simulations):

            if self.reset_theta:
                theta = np.random.normal(size=self.action_space_dimension)
                theta = theta / np.linalg.norm(theta)
                self.theta = np.abs(theta)

            print("real theta is ", self.theta)

            pickle_fn = f"{policy_name}_{time_horizon}_{is_censored}_{sim_index}.pkl"
            pickle_path = self.pickle_base_path + pickle_fn
            if os.path.exists(pickle_path):
                print(f"Skipping {pickle_fn}")
                with open(pickle_path, 'rb') as f:
                    sim_regret = pickle.load(f)
                regret[:, sim_index] = sim_regret

            else:

                if self.verbose and sim_index % (num_simulations / 4) == 0:
                    print("experiments executed: %d", sim_index)
                    sys.stdout.flush()

                # Reinitialize the policy
                policy.init(self.action_space_dimension)

                delays = self.generate_delays(time_horizon)

                disclosure_history = [
                    [] for t in range(
                        time_horizon
                        # self.disclosure_size(time_horizon, is_censored)
                    )
                ]  # a list of size T (?) to store delays, rewards and played vectors

                for t in tqdm.tqdm(range(time_horizon), total=time_horizon):

                    action_played, round_reward, instant_regret = self.play_single_round(
                        disclosure_history, policy, t)

                    delay = delays[t]

                    # bestDelay, bestReward, bestFeatures = self.MAB.play(instantBestArm) #get the oracle (best arm)
                    # Rq : ^ that's regret, not pseudo-regret ! We want pseudo regret

                    regret[t, sim_index] = instant_regret

                    if (
                        round_reward != 0
                        and self.delay_not_too_high(delay, is_censored)
                        and t + delay < time_horizon
                    ):
                        # add to history (possibly in the future) delay, reward,
                        # action played
                        instant_disclosure = (delay, round_reward, action_played)
                        disclosure_time = t + delay  # t + delay - 1
                        disclosure_history[disclosure_time].append(instant_disclosure)

                with open(pickle_path, 'wb') as f:
                    pickle.dump(regret[:, sim_index], f)

            cum_regret[:, sim_index] = np.cumsum(regret[:, sim_index])

        # write a big csv table containing the results and the related infos
        # ('timesteps', 'policy', cumregret , 'm', 'tau')
        np.savetxt(f'{folder}/{subfolder}/{policy_name}.csv', cum_regret)  # ,sep=',')

        return cum_regret, policy_name

    def play_single_round(self, disclosure_history, policy, t):
        available_arms = self.get_uniform_arms()  # all arms always available
        # This would be for pseudo-regret:
        instant_best_arm, instant_best_reward = self.get_best_arm_reward(available_arms)
        if t == 0 and policy.id() != 'BootstrapLinTS':
            # choose an arm uniformly at random
            chosen_arm_index = np.random.choice(range(self.arms_number))
        elif t == 0:
            chosen_arm_index = policy.selectArm(available_arms)
        else:
            # show current history to policy
            policy.updateState(disclosure_history[t - 1])
            policy.initialized = True
            # choose an arm according to policy
            chosen_arm_index = policy.selectArm(available_arms)
        chosen_arm = available_arms[chosen_arm_index]
        # get the feedback and update
        round_reward, action_played = self.play(chosen_arm)
        instant_regret = instant_best_reward - round_reward
        return action_played, round_reward, instant_regret

    def generate_delays(self, time_horizon):
        if self.empirical:
            delays = self.delay_pdf.rvs(size=time_horizon)
        else:
            delays = self.delay_pdf.rvs(self.param, size=time_horizon)
            delays = np.minimum(delays, 2 * time_horizon)  # Avoid overflow before casting
            delays = delays.astype(int)
        return delays

    def run_cens(self, T, N, tsav, verbose):
        """Runs an experiment with parameters T and N.

        It returns a dictionary whose keys are policies and whose values
        are the regret obtained by these policies over the experiments and
        averaged over N runs.

        Parameters
        ----------
        T: int
            Length of the sequential allocations.

        N: int
            Number of Monte Carlo repetitions.

        tsav: numpy array (ndim = 1)
            Points to save on each trajectory.
        """
        return self.run(T, N, tsav, verbose, True)

    def run_uncens(self, T, N, tsav, verbose):
        """Runs an experiment with parameters T and N.

        It returns a dictionary whose keys are policies and whose values
        are the regret obtained by these policies over the experiments and
        averaged over N runs.

        Parameters
        ----------
        T: int
            Length of the sequential allocations.

        N: int
            Number of Monte Carlo repetitions.

        tsav: numpy array (ndim = 1)
            Points to save on each trajectory.
        """
        return self.run(T, N, tsav, verbose, False)

    def delay_not_too_high(self, delay, has_censorship):
        return not has_censorship or delay <= self.m

    def disclosure_size(self, T, is_censored):
        if is_censored:
            return T + self.m
        return T
