import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dataclasses

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from typing import Dict, Tuple, Iterable, List
from IPython.display import clear_output

from dataset import ProcessedDataset
from scoring import tryfit, binarizeX


def group_comparison_barplot(y: np.ndarray, S: np.ndarray, priv: np.ndarray, ax=None):
    no_axis_provided = ax is None
    drop_label = next(_ for _ in np.unique(y) if not _)
    df = (
        pd.DataFrame({"Sensitive": S, "Label": y})
        .pivot_table(columns="Label", index="Sensitive", aggfunc=len, fill_value=0)
        .apply(lambda x: x / sum(x), axis=1)
        .drop(drop_label, axis=1)
    )
    priv_proba = df.loc[priv[0]].values[0]

    if no_axis_provided:
        fig, ax = plt.subplots()

    sns.barplot(x=np.ones(df.shape[0]), y=df.index, label="y=0", color="#a1c9f4", ax=ax)
    sns.barplot(
        x=np.ones(df.shape[0]) * priv_proba,
        y=df.index,
        label="y'-y",
        color="#c44e52",
        ax=ax,
    )
    sns.barplot(x=df.values.T[0], y=df.index, label="y=1", color="#4878d0", ax=ax)

    if no_axis_provided:
        ax.set(ylabel="Sensitive", xlabel="Proportion")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(reversed(handles), reversed(labels), frameon=True, ncol=2)


def group_comparison_multiplot(
    plots: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    rows: int,
    cols: int,
    title: str = None,
    axis_lim: Tuple[float, float] = (0.0, 1.0),
):
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))
    axes = np.array(axes).flatten()

    for i, (ax, (label, (y, S, priv))) in enumerate(zip(axes, plots.items())):
        group_comparison_barplot(y, S, priv, ax)
        ax.set(title=label, xlabel="Proportion", ylabel="")
        handles, labels = ax.get_legend_handles_labels()
        if i % cols:
            ax.yaxis.set_ticklabels([])

    plt.setp(axes, xlim=axis_lim)

    fig.legend(
        reversed(handles), reversed(labels), loc="lower left", frameon=True, ncol=2
    )
    fig.suptitle(title)
    plt.tight_layout()


def compare_groups(
    train: ProcessedDataset,
    test: ProcessedDataset,
    classifiers_cat: Dict[str, BaseEstimator] = {},
    classifiers_bin: Dict[str, BaseEstimator] = {},
    rows: int = 0,
    cols: int = 0,
    include_actual: bool = True,
    axis_lim: Tuple[float, float] = (0.0, 1.0),
    alternate: bool = False,
):
    (
        pred_cat_train,
        pred_cat_test,
        pred_bin_train,
        pred_bin_test,
        pred_actual_train,
        pred_actual_test,
    ) = ({}, {}, {}, {}, {}, {})

    for name, c in classifiers_cat.items():
        clear_output(wait=True)
        print(f"Fitting: {name}")
        c = tryfit(c, train)
        pred_cat_train[f"{name} Training"] = c.predict(train.X)
        pred_cat_test[f"{name} Testing"] = c.predict(test.X)

    if include_actual:
        clear_output(wait=True)
        print(f"Fitting: Actual")
        pred_actual_train["Training Actual"] = train.y
        pred_actual_test["Testing Actual"] = test.y

    if classifiers_bin:
        (X_train_bin, priv_bin), (X_test_bin, _) = binarizeX(
            train.X, train.priv
        ), binarizeX(test.X, train.priv)
        for name, c in classifiers_bin.items():
            clear_output(wait=True)
            print(f"Fitting: {name}")
            c = tryfit(c, dataclasses.replace(train, X=X_train_bin, priv=priv_bin))
            pred_bin_train[f"{name} Training"] = c.predict(X_train_bin)
            pred_bin_test[f"{name} Testing"] = c.predict(X_test_bin)
    clear_output(wait=True)

    pred_train = {
        name: (y_pred, train.S_raw, train.priv_raw)
        for name, y_pred in dict(
            pred_actual_train, **pred_bin_train, **pred_cat_train
        ).items()
    }

    pred_test = {
        name: (y_pred, test.S_raw, train.priv_raw)
        for name, y_pred in dict(
            pred_actual_test, **pred_bin_test, **pred_cat_test
        ).items()
    }

    if alternate:
        l = sum(zip(pred_train.items(), pred_test.items()), ())
        predictions = {key: value for (key, value) in l}
    else:
        predictions = {**pred_train, **pred_test}

    group_comparison_multiplot(predictions, rows=rows, cols=cols, axis_lim=axis_lim)
