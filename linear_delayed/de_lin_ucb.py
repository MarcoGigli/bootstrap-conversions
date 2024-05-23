###### IMPORTANT ######
# This code comes from
# Vernade et al. - Linear bandits with Stochastic Delayed Feedback - 2020
# with minimal refactoring.
# It is attached for reproducibility of the comparison

import numpy as np
from numpy.linalg import pinv
#use buffer to store last 'threshold' actions
from collections import deque

from linear_delayed.policy import Policy


class PolicyDeLinUCB(Policy):

    def __init__(self, T, m, lambda_reg, delta, alpha, bias_term=True):
        super().__init__(T, m)

        self.alpha = alpha
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.initialized = False
        self.bias_term = bias_term
        # self.Delta = 1
        # dropping the loglog(t) term for now


    def selectArm(self, arms):
        """
        This function implements the randomized LinUCB algorithm in delayed environment.
        It discards all observations received within the last m time steps.
        Input:
        -------
        arms : (K x d) array containing K arms in dimension d

        Output:
        -------
        chosen_arm : index of the pulled arm
        """
        if not self.initialized:
            return None  # Better raise error

        K = len(arms)
        self.UCBs = np.zeros(K)

        for i in range(K):
            a = arms[i, :]
            covxa = np.inner(self.invcov, a.T)

            if self.bias_term:
                self.UCBs[i] = np.dot(self.hat_theta, a) \
                               + self.alpha \
                               * (np.sqrt(
                    self.beta[self.t - 1] + self.lambda_reg) + np.sum(self.bias)) \
                               * (np.dot(a, covxa))
            else:
                # neglecting the bias term (wrong confidence regions)
                self.UCBs[i] = np.dot(self.hat_theta, a) \
                               + self.alpha * (np.sqrt(
                    self.beta[self.t - 1] + np.sqrt(self.lambda_reg)) * (
                                                   np.dot(a, covxa)))

        # print(self.bias)
        # print(np.sum(self.bias))
        # print(self.UCBs)
        mixer = np.random.random(
            self.UCBs.size)  # Shuffle to avoid always pulling the same arm when ties
        UCB_indices = list(np.lexsort((mixer, self.UCBs)))  # Sort the indices
        output = UCB_indices[::-1]  # Reverse list
        chosen_arm = output[0]

        # bias update (exact elliptical norm)
        A_norm = np.dot(arms[chosen_arm, :],
                        np.inner(self.invcov, arms[chosen_arm, :].T))
        self.bias.append(A_norm)  # automatically remove overflow

        xxt = np.outer(arms[chosen_arm, :], arms[chosen_arm, :].T)
        self.cov += xxt
        self.invcov = pinv(self.cov)
        # self.Delta = min(self.m, self.Delta +1)

        return chosen_arm

    def updateState(self, disclosure):
        "disclosure is a list of T lists containing all data to be displayed at time t"

        for feedback in disclosure:
            delay, reward, features = feedback
            self.xy += reward * features

        self.hat_theta = np.inner(self.invcov, self.xy)
        self.t += 1

    def init(self, dim):

        self.t = 0
        self.dim = dim
        self.hat_theta = np.zeros(self.dim)
        self.cov = self.lambda_reg * np.identity(self.dim)
        self.invcov = np.identity(self.dim)
        self.bias = deque(maxlen=self.m)  # buffer of size m
        self.xy = np.zeros(self.dim)
        self.beta = [2 * np.log(1 / self.delta) + self.dim * (
            np.log(1 + t / (self.lambda_reg * self.dim))) for t in range(1, self.T)]

    def __str__(self):
        if self.bias_term:

            return 'OTFLinUCB'
        else:
            return 'OTFLinUCB-wrong'

    @staticmethod
    def id():
        return 'OTFLinUCB'

    @staticmethod
    def recquiresInit():
        return True