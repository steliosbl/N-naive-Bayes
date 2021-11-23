import numpy as np
import warnings
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, NamedTuple, Set, Iterable, Callable, Optional

from sklearn.naive_bayes import _BaseNB, GaussianNB
from sklearn.utils.multiclass import unique_labels, _check_partial_fit_first_call
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.preprocessing import (
    LabelBinarizer,
    label_binarize,
    OrdinalEncoder,
    LabelEncoder,
)

from gaussian_sub import GaussianSub

logger = logging.getLogger(__name__)


@dataclass
class BaseNNBMixin:
    alpha: float = (
        1.0  # Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)
    )
    fit_prior: bool = True  # Whether to learn class prior probabilities or not. If false, a uniform prior will be used
    class_prior: Optional[
        np.ndarray
    ] = None  # Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    delta: float = 0.01  # N-Naive Bayes heuristic probability adjustment factor
    sub_estimator_factory: _BaseNB = (
        GaussianSub  # Type of classifier to use as sub-estimator
    )
    max_iter: int = 1000  # Max number of iterations of the heuristic
    disc_threshold: float = 1e-2  # Value of discrimination score that has to be reached for the heuristic to terminate
    print_probas: bool = False  # Whether to print the probabilities with each iteration of the heuristic (for debugging)
    enable_heuristic: bool = True  # Whether to run the heuristic
    reset_s_proba: bool = (
        True  # Whether to reset the probability of S with each call to `partial_fit`
    )
    partial_balance: bool = (
        True  # Whether to run the heuristic when calling `partial_fit`
    )
    estimator_name: str = "BaseNNB"  # The name of this estimator


class BaseNNB(ABC, _BaseNB, BaseNNBMixin):
    def _check_X(self, X):
        X = np.array(X)
        return self._validate_data(X, reset=False)

    def _check_X_y(self, X, y):
        X, y = np.array(X), np.array(y)
        if X.shape[0] == y.shape[0]:
            return self._validate_data(X, y, reset=False)
        raise ValueError(f"X shape {X.shape} and y shape {y.shape} are incompatible")

    def _update_class_log_prior(self, class_prior=None, class_count=None):
        if class_prior is not None:
            if len(class_prior) != self.n_classes_:
                raise ValueError("Number of priors must match number of classes.")
            return np.log(class_prior)
        elif self.fit_prior:
            with warnings.catch_warnings():
                # silence the warning when count is 0 because class was not yet
                # observed
                warnings.simplefilter("ignore", RuntimeWarning)
                log_class_count = np.log(class_count)

            # empirical prior, with sample_weight taken into account
            return log_class_count - np.log(class_count.sum())
        else:
            return np.full(self.n_classes_, -np.log(self.n_classes_))

    def _split_S(self, X, s_index: int = -1):
        S, X_rest = X[:, s_index], np.delete(X, s_index, axis=1)

        return S, X_rest

    def _encode_S_priv(self, s_priv, s_encoder):
        if not len(s_priv):
            raise ValueError("At least one privileged group must be specified")

        s_priv = (
            s_encoder.transform(np.array(s_priv).reshape(-1, 1))
            .reshape(1, -1)[0]
            .astype(int)
        )

        return s_priv

    def _init_counters(self, n_classes, n_s_groups):
        self.class_count_ = np.zeros(n_classes)
        self.s_counts_ = np.zeros((n_classes, n_s_groups)) + 1
        self.s_proba_ = np.zeros((n_classes, n_s_groups))

    def _count(self, X, y, S):
        numpos = y.sum(axis=0)
        self.class_count_ += np.array([len(y) - numpos, numpos])
        hist = self._histogram(y, S)
        self.s_counts_ += hist
        if self.reset_s_proba:
            self.s_proba_ = self.s_counts_.copy()
        else:
            self.s_proba_ += hist

    def _encode_S(self, S, s_encoder):
        S = s_encoder.transform(S.reshape(-1, 1)).reshape(1, -1)[0].astype(int)
        s_groups = self.s_encoder_.categories_[0]
        n_s_groups = len(s_groups)
        s_group_labels = np.array(range(n_s_groups))

        return S, s_groups, n_s_groups, s_group_labels

    def _predict_unlabeled(self, X):
        check_is_fitted(self)
        jll = self._joint_log_likelihood(X)
        return np.argmax(jll, axis=1)

    def _check_partial_fit_first_call(self, classes, s_priv):
        if _check_partial_fit_first_call(self, classes):
            if not len(s_priv):
                raise ValueError("Must provide s_priv on first call to partial_fit")
            return True
        return False

    def partial_fit(
        self,
        X,
        y,
        s_priv: np.ndarray = None,
        classes: np.ndarray = None,
        s_index: int = -1,
    ):
        # If this is the first call to partial_fit,
        # verify that classes and s_priv has been passed
        # then call regular fit
        if (
            self._check_partial_fit_first_call(classes, s_priv)
            and s_priv is not None
            and classes is not None
        ):
            return self.fit(X, y, s_priv, classes, s_index)

        # Else, do a partial fit
        # Validate the given data, compute $n_features_
        X, y = self._check_X_y(X, y)

        # Split the sensitive features from X
        S, X_rest = self._split_S(X, self.s_index_)

        # Label binarize y
        # (use label_binarize function instead of binarizer to avoid unnecessary computation)
        y = label_binarize(y, classes=self.classes_).reshape(1, -1)[0]
        numpos_init = np.sum(y)

        # Ordinal encode S
        # Verify that no previously unseen values exist in S
        S, s_groups_new, _, _ = self._encode_S(S, self.s_encoder_)
        if set(s_groups_new) != set(self.s_groups_):
            raise ValueError("Unrecognised value in S feature vector")

        # Count the number of positive labels
        # Then update the class prior probability
        self._count(X, y, S)
        self.class_log_prior_ = self._update_class_log_prior(
            class_prior=self.class_prior, class_count=self.class_count_
        )

        # TODO: Validate sample_weight

        # Train the naive bayes sub-estimators
        logger.info(
            f"Partial fitting {self.n_s_groups_} sub-estimators of type {self.sub_estimator_factory.__name__}"
        )
        for s in self.s_group_labels_:
            mask = S == s
            if X_rest[mask].shape[0]:
                self.estimators_[s].partial_fit(X_rest[mask, :], y[mask])
                self.estimators_[s].class_log_prior_ = np.array([0, 0])

        if self.partial_balance:
            self._balance(X_rest, S, numpos_init)

    def fit(
        self,
        X,
        y,
        s_priv: np.ndarray,
        classes: Optional[np.ndarray] = None,
        s_index: int = -1,
    ):
        # Validate the given data, compute $n_features_
        X, y = self._check_X_y(X, y)

        # Split the sensitive features from X
        self.s_index_ = s_index
        S, X_rest = self._split_S(X, self.s_index_)

        # Label binarize y
        # Record the classes and their number
        if classes is not None:
            self.classes_ = np.array(classes)
            y = label_binarize(y, classes=self.classes_).reshape(1, -1)[0]
        else:
            label_binarizer = LabelBinarizer()
            y = label_binarizer.fit_transform(y).reshape(1, -1)[0]
            self.classes_ = np.array(label_binarizer.classes_)
        self.n_classes_ = len(self.classes_)
        self.class_labels_ = np.array(range(self.n_classes_))

        # Ordinal encode S
        # Then record the groups and their number
        self.s_encoder_ = OrdinalEncoder().fit(S.reshape(-1, 1))
        S, self.s_groups_, self.n_s_groups_, self.s_group_labels_ = self._encode_S(
            S, self.s_encoder_
        )

        # Count the number of positive labels
        # Then update the class prior probability
        self._init_counters(self.n_classes_, self.n_s_groups_)
        self._count(X, y, S)
        self.class_log_prior_ = self._update_class_log_prior(
            class_prior=self.class_prior, class_count=self.class_count_
        )

        # Validate s_priv and compute s_non_priv_ by removing the privileged groups
        # from the array of all groups
        self.s_priv_ = self._encode_S_priv(s_priv, self.s_encoder_)
        self.s_non_priv_ = np.delete(self.s_group_labels_, self.s_priv_)

        # TODO: Validate sample_weight

        # Train the naive bayes sub-estimators
        logger.info(
            f"Fitting {self.n_s_groups_} sub-estimators of type {self.sub_estimator_factory.__name__}"
        )
        self.estimators_: Dict[int, _BaseNB] = {}

        for s in self.s_group_labels_:
            mask = S == s
            est = self.sub_estimator_factory().partial_fit(
                X_rest[mask, :], y[mask], self.class_labels_
            )
            est.class_log_prior_ = np.array([0, 0])
            self.estimators_[s] = est

        self._balance(X_rest, S, self.class_count_[1])

        return self

    @abstractmethod
    def _balance(self, X_rest, S, numpos_init):
        pass

    def _predict(self, log_proba, y_pred_class_log_prior, masks):
        s_proba = np.log(self.s_proba_ + self.alpha)
        log_proba_pred = log_proba.copy()  # + y_pred_class_log_prior
        for s in self.s_group_labels_:
            log_proba_pred[masks[s], :] += s_proba[:, s]
        y_pred = np.argmax(log_proba_pred, axis=1)
        numpos = y_pred.sum()

        y_pred_class_count = np.array([len(y_pred) - numpos, numpos])
        y_pred_class_log_prior = self._update_class_log_prior(
            class_prior=self.class_prior, class_count=y_pred_class_count
        )

        return y_pred, y_pred_class_log_prior, numpos

    @abstractmethod
    def _recalculate(self, y_pred, S):
        pass

    def _histogram(self, y_pred, S):
        return np.histogram2d(y_pred, S, (self.n_classes_, self.n_s_groups_))[0]

    def _joint_log_likelihood(self, X):
        # Split the sensitive features from `X`
        S, X_rest = self._split_S(X, self.s_index_)

        # Ordinal encode S
        S, _, _, _ = self._encode_S(S, self.s_encoder_)

        # Store the results here
        log_proba = np.empty((X.shape[0], self.n_classes_))

        s_proba = self.s_proba_ + self.alpha

        # Iterate sensitive feature values
        for s in self.s_group_labels_:
            # Take the samples with this value for the sensitive feature
            mask = S == s

            # Compute the log probabilities for that subset of samples
            #   - Log probability of all features except S plus
            #   - Log of adjusted probability for S plus
            #   - Log of prior class probability
            log_proba[mask, :] = self.estimators_[s]._joint_log_likelihood(
                X_rest[mask, :]
            ) + np.log(s_proba[:, s])

        return log_proba
