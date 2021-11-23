import numpy as np
from sklearn.naive_bayes import GaussianNB

from typing import Iterable


class GaussianSub(GaussianNB):
    def __init__(self, *, priors=None, var_smoothing=1e-9):
        super().__init__(priors=priors, var_smoothing=var_smoothing)
        self.class_log_prior_: Iterable[float] = []

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = self.class_log_prior_[i]
            n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) / (self.var_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood
