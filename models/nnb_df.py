import numpy as np
import logging
from dataclasses import dataclass

from nnb_base import BaseNNB

logger = logging.getLogger(__name__)


@dataclass
class NNB_DF(BaseNNB):
    estimator_name: str = "NNB-DF"

    def _balance(self, X_rest, S, numpos_init):
        # Store the results here
        log_proba = np.empty((X_rest.shape[0], self.n_classes_))

        masks = {s: S == s for s in self.s_group_labels_}
        for s in self.s_group_labels_:
            log_proba[masks[s], :] = self.estimators_[s]._joint_log_likelihood(
                X_rest[masks[s], :]
            )

        y_pred, y_pred_class_log_prior, numpos = self._predict(
            log_proba, self.class_log_prior_, masks
        )

        (
            max_p,
            min_np,
            ratio_up,
            ratio_down,
            log_probas,
        ) = self._recalculate(y_pred, S)

        logger.debug(f"Initial probas: {np.exp(log_probas)}")

        iter_count = 0

        # See paper, Algorithm 2
        while ratio_down > self.disc_threshold and self.enable_heuristic:
            # Counting
            iter_count += 1
            if iter_count == self.max_iter:
                logger.error("Heuristic reached max iterations")
                break

            if ratio_up > ratio_down:
                self.s_proba_[1, max_p] -= self.delta * self.s_proba_[1, max_p]
                self.s_proba_[0, max_p] += self.delta * self.s_proba_[0, max_p]
            else:
                self.s_proba_[0, min_np] -= self.delta * self.s_proba_[0, min_np]
                self.s_proba_[1, min_np] += self.delta * self.s_proba_[1, min_np]

            y_pred, y_pred_class_log_prior, numpos = self._predict(
                log_proba, y_pred_class_log_prior, masks
            )

            (
                max_p,
                min_np,
                ratio_up,
                ratio_down,
                log_probas,
            ) = self._recalculate(y_pred, S)

        # self.class_log_prior_ = y_pred_class_log_prior

        logger.info(f"Heuristic took {iter_count} iterations")
        logger.debug(f"Final probas: {np.exp(log_probas)}")

    def _recalculate(self, y_pred, S):
        # Get the histogram of y_pred vs S
        m = self._histogram(y_pred, S)

        # Get the conditional probabilities P(Y=1|S=s) as
        # (n(Y=1 ∩ S=s) + a) / (n(S=s) + a)
        log_probas = np.log(m[1] + self.alpha) - np.log(
            np.sum(m, axis=0) + self.alpha * self.n_classes_
        )

        if self.print_probas:
            print(np.exp(log_probas))

        # Get the s values to modify:
        # Probability will be added to the non-privileged s with lowest probability
        # Probability will be subtracted from the privileged s with highest probability
        min_p = self.s_priv_[np.argmin(log_probas[self.s_priv_])]
        max_p = self.s_priv_[np.argmax(log_probas[self.s_priv_])]
        min_np = self.s_non_priv_[np.argmin(log_probas[self.s_non_priv_])]
        max_np = self.s_non_priv_[np.argmax(log_probas[self.s_non_priv_])]

        ratio_up = log_probas[max_np] - log_probas[min_p]
        ratio_down = log_probas[max_p] - log_probas[min_np]

        return (
            max_p,
            min_np,
            ratio_up,
            ratio_down,
            log_probas,
        )
