import numpy as np

from enum import Enum
from dataclasses import dataclass
from typing import Optional

from sklearn.preprocessing import LabelEncoder

import folktables
from folktables import adult_filter, ACSDataSource


class SensitiveAttr(Enum):
    RACE = "RAC1P"
    SEX = "SEX"
    RACESEX = ["RAC1P", "SEX"]


@dataclass
class ProcessedDataset:
    X: np.ndarray
    y: np.ndarray
    S: np.ndarray
    S_raw: np.ndarray
    priv: np.ndarray
    non_priv: np.ndarray
    priv_raw: np.ndarray
    s_groups: np.ndarray
    s_groups_raw: np.ndarray
    s_encoder: LabelEncoder


def filter_sensitive(S, other_value, threshold):
    values, counts = np.unique(S, return_counts=True)
    for value, count in zip(values, counts):
        if count < threshold:
            S[S == value] = other_value
    return S


@dataclass
class Dataset:
    RACES = {
        1: "White",
        2: "Black",
        3: "Amer-Indian",
        4: "Alaskan",
        5: "AI_AN_S",
        6: "Asian",
        7: "HW-PI",
        8: "Other",
        9: "MR",
    }

    RACES_OTHER = 8

    SEXES = {1: "Male", 2: "Female"}

    PRIVILEGED = {
        SensitiveAttr.RACE: [1],
        SensitiveAttr.SEX: [1],
        SensitiveAttr.RACESEX: ["11"],
    }

    PRIVILEGED_MULTI = {
        SensitiveAttr.RACE: [1],
        SensitiveAttr.SEX: [1],
        SensitiveAttr.RACESEX: ["11", "12"],
    }

    s_encoder: LabelEncoder = None
    sensitive: SensitiveAttr = SensitiveAttr.RACE
    sensitive_multi: bool = False
    binsensitive: bool = False

    def process_ds(self, X, y, S):
        priv = (
            Dataset.PRIVILEGED_MULTI[self.sensitive]
            if self.sensitive_multi
            else Dataset.PRIVILEGED[self.sensitive]
        )
        if self.sensitive == SensitiveAttr.RACESEX:
            S[:, 0] = filter_sensitive(S[:, 0], Dataset.RACES_OTHER, 100)
            S_raw = np.array(
                [
                    Dataset.RACES[x1] + Dataset.SEXES[x2]
                    for x1, x2 in zip(S[:, 0], S[:, 1])
                ]
            )
            S = np.apply_along_axis(lambda x: "".join(x), 1, S.astype(str))
            priv_raw = [
                Dataset.RACES[int(_[0])] + Dataset.SEXES[int(_[1])] for _ in priv
            ]
        else:
            if self.sensitive == SensitiveAttr.RACE:
                S = filter_sensitive(S, Dataset.RACES_OTHER, 100)
                S_raw = np.array([Dataset.RACES[x] for x in S])
                priv_raw = [Dataset.RACES[_] for _ in priv]
            elif self.sensitive == SensitiveAttr.SEX:
                S_raw = np.array([Dataset.SEXES[x] for x in S])
                priv_raw = [Dataset.SEXES[_] for _ in priv]

        if not self.s_encoder:
            self.s_encoder = LabelEncoder().fit(S)
        S = self.s_encoder.transform(S)
        priv = self.s_encoder.transform(priv)

        if self.binsensitive:
            S = np.isin(S, priv).astype(int)
            priv = np.array([1])

        s_groups = np.unique(S)
        non_priv = s_groups[s_groups != priv]

        return ProcessedDataset(
            X=np.c_[X, S],
            y=y,
            S=S,
            S_raw=S_raw,
            priv=np.array(priv),
            priv_raw=np.array([priv_raw]),
            non_priv=non_priv,
            s_groups=s_groups,
            s_groups_raw=np.unique(S_raw),
            s_encoder=self.s_encoder,
        )


@dataclass
class Income(Dataset):
    FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP"]

    def load(self, data_src):
        spec = folktables.BasicProblem(
            features=Income.FEATURES,
            target="PINCP",
            target_transform=lambda x: x > 50000,
            group=self.sensitive.value,
            preprocess=adult_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )

        X, y, S = spec.df_to_numpy(data_src)
        return self.process_ds(X, y, S)


@dataclass
class Employment(Dataset):
    FEATURES = [
        "AGEP",
        "SCHL",
        "MAR",
        "RELP",
        "DIS",
        "ESP",
        "CIT",
        "MIG",
        "MIL",
        "ANC",
        "NATIVITY",
        "DEAR",
        "DEYE",
        "DREM",
    ]

    def load(self, data_src):
        spec = folktables.BasicProblem(
            features=Employment.FEATURES,
            target="ESR",
            target_transform=lambda x: x == 1,
            group=self.sensitive.value,
            preprocess=lambda x: x,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )

        X, y, S = spec.df_to_numpy(data_src)
        return self.process_ds(X, y, S)


def init_data_src(year: str = "2018", download: bool = True):
    data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
    return data_source.get_data(states=["CA"], download=download)
