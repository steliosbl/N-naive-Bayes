import numpy as np
import logging
from dataclasses import dataclass

from nnb_base import BaseNNB

logger = logging.getLogger(__name__)


@dataclass
class TwoNaiveBayes(BaseNNB):
    estimator_name: str = "2NB"

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
        disc, s_min, s_max, probas = self._recalculate(y_pred, S)

        logger.debug(f"Initial probas: {probas}")
        logger.debug(f"Numpos = {numpos}/{numpos_init}")
        logger.debug(f"Disc = {disc}")

        iter_count = 0
        while disc > self.disc_threshold and self.enable_heuristic:

            # Counting
            iter_count += 1
            if iter_count == self.max_iter:
                logger.error("Heuristic reached max iterations")
                break

            if numpos < numpos_init:
                new = self.s_proba_[0, s_min] - self.delta * self.s_proba_[0, s_max]
                if new < 0:
                    break
                self.s_proba_[1, s_min] += self.delta * self.s_proba_[0, s_max]
                self.s_proba_[0, s_min] = new
            else:
                new = self.s_proba_[1, s_max] - self.delta * self.s_proba_[1, s_min]
                if new < 0:
                    break
                self.s_proba_[0, s_max] += self.delta * self.s_proba_[1, s_min]
                self.s_proba_[1, s_max] = new

            y_pred, y_pred_class_log_prior, numpos = self._predict(
                log_proba, y_pred_class_log_prior, masks
            )
            disc, s_min, s_max, probas = self._recalculate(y_pred, S)

        if self.add_prior:
            self.class_log_prior_ = y_pred_class_log_prior

        logger.info(f"Heuristic took {iter_count} iterations")
        logger.debug(f"Final probas: {probas}")

    def _recalculate(self, y_pred, S):
        # Get the histogram of y_pred vs S
        m = self._histogram(y_pred, S)

        # Get the conditional probabilities P(Y=1|S=s) as
        # (n(Y=1 ∩ S=s) + a) / (n(S=s) + a)
        probas = (m[1] + self.alpha) / (np.sum(m, axis=0) + self.alpha)

        if self.print_probas:
            print(probas)

        # Get the s values to modify:
        # Probability will be added to the non-privileged s with lowest probability
        # Probability will be subtracted from the privileged s with highest probability
        s_min = self.s_non_priv_[np.argmin(probas[self.s_non_priv_])]
        s_max = self.s_priv_[np.argmax(probas[self.s_priv_])]

        # Calculate the cv score as P(Y=1|S=s_max) - P(Y=1|S=s_min)
        disc = probas[s_max] - probas[s_min]

        return (disc, s_min, s_max, probas)
