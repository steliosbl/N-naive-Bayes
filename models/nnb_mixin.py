import numpy as np

from dataclasses import dataclass
from typing import Optional
from sklearn.naive_bayes import _BaseNB

from gaussian_sub import GaussianSub


@dataclass
class NNBMixin:
    # Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)
    alpha: float = 1.0

    # Whether to learn class prior probabilities or not. If false, a uniform prior will be used
    fit_prior: bool = True

    # Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    class_prior: Optional[np.ndarray] = None

    # N-Naive Bayes heuristic probability adjustment factor
    delta: float = 0.01

    # Type of classifier to use as sub-estimator
    sub_estimator_factory: _BaseNB = GaussianSub

    # Max number of iterations of the heuristic
    max_iter: int = 1000

    # Value of discrimination score that has to be reached for the heuristic to terminate
    disc_threshold: float = 1e-2

    # Whether to print the probabilities with each iteration of the heuristic (for debugging)
    print_probas: bool = False

    # Whether to run the heuristic
    enable_heuristic: bool = True

    # Whether to reset the probability of S with each call to `partial_fit`
    reset_s_proba: bool = True

    # Whether to run the heuristic when calling `partial_fit`
    partial_balance: bool = True

    # The name of this estimator
    estimator_name: str = "BaseNNB"
