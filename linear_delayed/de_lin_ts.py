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


class PolicyDeLinTS(Policy):

    def __init__(self, T, m, sigma_0):

        super().__init__(T, m)
        self.sigma_0 = sigma_0
        self.Delta = 1  # missing observations m - nb already received obs

    def selectArm(self, arms):
        """
        LinTS with approximate prior including an upper bound on the extra variance coming from the delays
        Input:
        -------
        arms : list of objects Arm with contextualized features

        Output:
        -------
        chosen_arm : index of the pulled arm
        """
        K = len(arms)

        theta_tilde = np.random.multivariate_normal(self.hat_theta, self.invcov)
        # print(theta_tilde)
        estimated_means = np.zeros(K)

        for i in range(K):
            a = arms[i, :]
            estimated_means[i] = np.dot(theta_tilde.T, a)

        # print(estimated_means)
        chosen_arm = np.argmax(estimated_means)

        xxt = np.outer(arms[chosen_arm, :], arms[chosen_arm, :].T)
        self.cov += xxt
        self.invcov = pinv((self.sigma_0 + np.sum(self.bias)) * self.cov)
        # self.Delta = min(self.Delta+1,self.m)
        # print(self.Delta)

        # bias update (exact elliptical norm)
        A_norm = np.dot(arms[chosen_arm, :],
                        np.inner(self.invcov, arms[chosen_arm, :].T))
        self.bias.append(A_norm)

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
        # the regularization is increased by the equivalent of m/lambda to cover for the additive variance due to the delays
        self.cov = (self.sigma_0 ** 2) * np.identity(self.dim)
        self.invcov = np.identity(self.dim)
        self.bias = deque(maxlen=self.m)
        self.xy = np.zeros(self.dim)

    def __str__(self):
        return 'OTFLinTS'

    @staticmethod
    def id():
        return 'OTFLinTS'

    @staticmethod
    def recquiresInit():
        return True