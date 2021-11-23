import dataclasses

import numpy as np
import pandas as pd

from IPython.display import clear_output

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from dataset import ProcessedDataset
from typing import Dict, Tuple, Iterable, List, Union


def tryfit(c, train):
    try:
        return c.fit(train.X, train.y, s_priv=train.priv)
    except TypeError:
        return c.fit(train.X, train.y)


def binarizeX(X, priv):
    X = X.copy()
    X[:, -1] = X[:, -1] == priv[0]
    priv = np.array([True])
    return X, priv


def binarize(ds):
    X = ds.X.copy()
    X[:, -1] = (X[:, -1] == ds.priv[0]).astype(int)
    priv, non_priv = np.array([1]), np.array([0])
    return dataclasses.replace(ds, X=X, priv=priv, non_priv=non_priv)


def histogram(y, S, n_classes: int, n_s_groups: int) -> np.ndarray:
    """Given vectors y and S, gives the histogram"""
    return np.histogram2d(y, S, (n_classes, n_s_groups))[0]


def probas_from_histogram(m, alpha: float = 1.0, n_classes: int = 2) -> np.ndarray:
    """Given the histogram of y and S, returns the smoothed conditional probabilities P(y=1|S=s)
    as (N(y=1 ∩ S=s)+a)/(N(S=s) + |y|*a)
    """
    return (m[1] + alpha) / (np.sum(m, axis=0) + alpha * n_classes)


def sensitive_probas(
    y, S, n_classes: int, n_s_groups: int, alpha: float = 1.0
) -> np.ndarray:
    """Given vectors y and S, gives the smoothed conditional probabilities"""
    return probas_from_histogram(
        histogram(y, S, n_classes, n_s_groups), alpha, n_classes
    )


def diavgall(probas, s_priv: Iterable[int], s_non_priv: Iterable[int]) -> float:
    """Given the conditional probabilities, and the priviliged and non-priviliged groups
    returns the average disparate impact for all non-priviliged vs privilged groups
    """
    return (probas[s_non_priv].mean() / probas[s_priv]).mean()


def diavgall_twosided(
    probas, s_priv: Iterable[int], s_non_priv: Iterable[int]
) -> float:
    """Given the conditional probabilities, and the priviliged and non-priviliged groups
    returns the two-sided average disparate impact i.e. the distance of the avg. DI from 1 on either side
    """
    di = diavgall(probas, s_priv, s_non_priv)
    return abs(1 - di)


def parity(probas, s_priv: Iterable[int], s_non_priv: Iterable[int]) -> float:
    """Given the conditional probabilities, and the priviliged and non-priviliged groups
    returns the indepencence/demographic parity score
    """
    return np.max(probas) - np.min(probas)


def parity_restricted(
    probas, s_priv: Iterable[int], s_non_priv: Iterable[int]
) -> float:
    """Given the conditional probabilities, and the priviliged and non-priviliged groups
    returns the indepencence/demographic parity score
    """
    return np.max(probas[s_priv]) - np.min(probas[s_non_priv])


def diff_ratio(probas, s_priv: Iterable[int], s_non_priv: Iterable[int]) -> float:
    """Given the conditional probabilities, and the priviliged and non-priviliged groups
    returns the differential fairness ratio
    """
    return np.min(probas) / np.max(probas)


def diff_ratio_restricted(
    probas, s_priv: Iterable[int], s_non_priv: Iterable[int]
) -> float:
    """Given the conditional probabilities, and the priviliged and non-priviliged groups
    returns the differential fairness ratio
    """
    return np.min(probas[s_non_priv]) / np.max(probas[s_priv])


def diff_epsilon(probas, s_priv: Iterable[int], s_non_priv: Iterable[int]) -> float:
    """Given the conditional probabilities, and the priviliged and non-priviliged groups
    returns the differential fairness epsilon
    """
    log_probas = np.log(probas)

    return np.max(log_probas) - np.min(log_probas)


def diff_epsilon_restricted(
    probas, s_priv: Iterable[int], s_non_priv: Iterable[int]
) -> float:
    """Given the conditional probabilities, and the priviliged and non-priviliged groups
    returns the differential fairness epsilon for fractions of non_priv over priv probabilities
    """
    log_probas = np.log(probas)
    min_log_proba = np.max(log_probas[s_priv]) - np.min(log_probas[s_non_priv])
    max_log_proba = np.max(log_probas[s_non_priv]) - np.min(log_probas[s_priv])

    return max(min_log_proba, max_log_proba)


def diff_bias_amp(true_probas, pred_probas, s_priv, s_non_priv) -> float:
    return diff_epsilon(pred_probas, s_priv, s_non_priv) - diff_epsilon(
        true_probas, s_priv, s_non_priv
    )


def diff_bias_amp_restricted(true_probas, pred_probas, s_priv, s_non_priv) -> float:
    return diff_epsilon_restricted(
        pred_probas, s_priv, s_non_priv
    ) - diff_epsilon_restricted(true_probas, s_priv, s_non_priv)


def scores(y_pred_probas, classes, ds, name: str = "Classifier"):
    n_s_groups = ds.s_groups.shape[0]
    y_pred, y_pred_proba = (
        classes[np.argmax(y_pred_probas, axis=1)],
        y_pred_probas[:, 1],
    )

    m_pred = histogram(y_pred, ds.S, n_classes=2, n_s_groups=n_s_groups)
    probas_pred = probas_from_histogram(m_pred, alpha=1.0, n_classes=2)

    m_true = histogram(ds.y, ds.S, n_classes=2, n_s_groups=n_s_groups)
    probas_true = probas_from_histogram(m_true, alpha=1.0, n_classes=2)

    return [
        dict(Classifier=name, Metric=metric, Score=score)
        for metric, score in [
            ("Accuracy", accuracy_score(y_pred=y_pred, y_true=ds.y)),
            ("AUC", roc_auc_score(y_true=ds.y, y_score=y_pred_proba)),
            ("DIAvgAll", diavgall(probas_pred, ds.priv, ds.non_priv)),
            ("Parity", parity(probas_pred, ds.priv, ds.non_priv)),
            ("Parity-R", parity_restricted(probas_pred, ds.priv, ds.non_priv)),
            ("EDF-ratio", diff_ratio(probas_pred, ds.priv, ds.non_priv)),
            ("EDF-ratio-R", diff_ratio_restricted(probas_pred, ds.priv, ds.non_priv)),
            ("EDF-ε", diff_epsilon(probas_pred, ds.priv, ds.non_priv)),
            ("EDF-ε-R", diff_epsilon_restricted(probas_pred, ds.priv, ds.non_priv)),
            (
                "EDF-amp",
                diff_bias_amp(probas_true, probas_pred, ds.priv, ds.non_priv),
            ),
            (
                "EDF-amp-R",
                diff_bias_amp_restricted(
                    probas_true, probas_pred, ds.priv, ds.non_priv
                ),
            ),
        ]
    ]


def score_table(
    c: BaseEstimator,
    train: ProcessedDataset,
    test: ProcessedDataset,
    name: str = "Classifier",
    display: bool = True,
    refit: bool = True,
) -> pd.DataFrame:
    if refit:
        c = tryfit(c, train)

    y_pred_probas = c.predict_proba(test.X)
    result = scores(y_pred_probas, c.classes_, test)

    return (
        pd.DataFrame(result).drop("Classifier", axis=1).set_index("Metric")
        if display
        else result
    )


def split_ds(
    data: ProcessedDataset, test_size: float = 0.25
) -> Tuple[ProcessedDataset, ProcessedDataset]:
    (
        X_train,
        X_test,
        y_train,
        y_test,
        S_train,
        S_test,
        S_raw_train,
        S_raw_test,
    ) = train_test_split(data.X, data.y, data.S, data.S_raw, test_size=test_size)

    train, test = dataclasses.replace(
        data, X=X_train, y=y_train, S=S_train, S_raw=S_raw_train
    ), dataclasses.replace(data, X=X_test, y=y_test, S=S_test, S_raw=S_raw_test)

    return train, test


def split_preserve_groups(
    data: ProcessedDataset, test_size: float = 0.25
) -> Tuple[ProcessedDataset, ProcessedDataset]:
    splits = []
    for s in data.s_groups:
        mask = data.S == s
        splits.append(
            train_test_split(data.X[mask], data.y[mask], data.S[mask], data.S_raw[mask])
        )

    X_train, X_test, y_train, y_test, S_train, S_test, S_raw_train, S_raw_test = (
        np.concatenate([t[_] for t in splits]) for _ in range(8)
    )

    train, test = dataclasses.replace(
        data, X=X_train, y=y_train, S=S_train, S_raw=S_raw_train
    ), dataclasses.replace(data, X=X_test, y=y_test, S=S_test, S_raw=S_raw_test)

    return train, test


def score_means(
    data: ProcessedDataset,
    classifiers_cat={},
    classifiers_bin={},
    n_splits: int = 1,
    test_size: float = 0.25,
    output_order=None,
    include_perfect: bool = False,
):
    results = []
    for _ in range(n_splits):
        clear_output(wait=True)
        print(f"Split {_+1} of {n_splits}")
        train, test = split_preserve_groups(data, test_size)
        classifiers_cat = (
            {"Classifier": classifiers_cat}
            if isinstance(classifiers_cat, BaseEstimator)
            else classifiers_cat
        )

        for name, c in classifiers_cat.items():
            c = tryfit(c, train)
            y_pred_probas = c.predict_proba(test.X)
            results += scores(y_pred_probas, c.classes_, test, name=name)

        if include_perfect:
            y_pred_probas = np.array(
                [test.y == c.classes_[0], test.y == c.classes_[1]]
            ).T
            results += scores(y_pred_probas, c.classes_, test, name="Perfect")

        if classifiers_bin:
            train_bin, test_bin = binarize(train), binarize(test)

            for name, c in classifiers_bin.items():
                c = tryfit(c, train_bin)
                y_pred_probas = c.predict_proba(test_bin.X)
                results += scores(y_pred_probas, c.classes_, test, name=name)
    clear_output(wait=True)

    iterable = dict(list(pd.DataFrame(results).groupby(["Classifier"])))
    if output_order:
        iterable = {name: iterable[name] for name in output_order}
    d = {}
    for c, g2 in iterable.items():
        g2 = g2.groupby(["Metric"])
        df = pd.DataFrame(dict(Mean=g2.Score.mean(), Var=g2.Score.var()))
        d[c] = pd.DataFrame(
            index=df.index, columns=[[c] * 2, ["Mean", "Var"]], data=df.values
        )

    return pd.concat(list(d.values()), axis=1).round(5)
