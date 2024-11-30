from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from config import make_dotdict_recursive

datasets_config = {
    "titanic": {
        "splits": {
            "train": "/train.csv",
        },
        "y_column": "Survived",
        "x_columns": {
            "Pclass": {
                "type": "int",
            },
            "Sex": {
                "type": "string",
                "map": {"male": "-1", "female": "1"},
            },
            "Age": {"type": "int"},
            "SibSp": {"type": "int"},
            "Parch": {"type": "int"},
            "Fare": {"type": "int"},
        },
    },
}


def load_and_pad(dataset: pd.DataFrame, columns: str | list[str], features_max: int):
    """Pad feature vectors with 0s."""
    data = torch.Tensor(dataset[columns].to_numpy())
    if isinstance(columns, list):
        data = torch.nn.functional.pad(
            data,
            pad=(0, features_max - data.shape[-1]),
            mode="constant",
            value=0,
        )
    return data


def load_dataset(dataset_str: str, features_max: int):
    """Return relevant columns and normalized values of dataset."""
    assert dataset_str in datasets_config
    base_path = "datasets/"
    dataset_config = make_dotdict_recursive(datasets_config[dataset_str])
    dataset_train = pd.read_csv(
        base_path + dataset_str + dataset_config.splits.train,
    ).dropna()
    if "test" in dataset_config.splits:
        dataset_test = pd.read_csv(
            base_path + dataset_str + dataset_config.splits.test,
        ).dropna()
        dataset_full = pd.concat(dataset_train, dataset_test)
    else:
        ds_len = len(dataset_train)
        dataset_full = dataset_train.copy()
        dataset_train_shuffled = dataset_train.iloc[np.random.permutation(ds_len)]
        dataset_train = dataset_train_shuffled.iloc[: int(ds_len / 2)].copy()
        dataset_test = dataset_train_shuffled.iloc[int(ds_len / 2) :].copy()

    x_columns = list(dataset_config.x_columns.keys())
    y_column = dataset_config.y_column

    # replace and standardize
    for cname, cspec in dataset_config.x_columns.items():
        if "map" in cspec:
            for find, replace in cspec.map.items():
                dataset_train[cname] = dataset_train[cname].replace(
                    to_replace=find,
                    value=replace,
                )
                dataset_test[cname] = dataset_test[cname].replace(
                    to_replace=find,
                    value=replace,
                )
            dataset_train[cname] = pd.to_numeric(dataset_train[cname])
            dataset_test[cname] = pd.to_numeric(dataset_test[cname])
        else:
            mean = dataset_full[cname].mean()
            std = dataset_full[cname].std()
            dataset_train[cname] = (dataset_train[cname] - mean) / std
            dataset_test[cname] = (dataset_test[cname] - mean) / std
    x_train = load_and_pad(dataset_train, x_columns, features_max)
    y_train = load_and_pad(dataset_train, y_column, 1)
    x_test = load_and_pad(dataset_test, x_columns, features_max)
    y_test = load_and_pad(dataset_test, y_column, 1)

    return x_train, y_train, x_test, y_test
