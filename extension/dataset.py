# Expand this for the other tasks

import gzip
import json
from pathlib import Path
from typing import Optional, Tuple
import time 
from copy import deepcopy

import numpy as np 
from sklearn.preprocessing import LabelEncoder
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        cat_idx: list,
        target_type: str,
        num_classes: int,
        num_features: Optional[int] = None,
        num_instances: Optional[int] = None,
        cat_dims: Optional[list] = None,
        split_indeces: Optional[list] = None,
        split_source: Optional[str] = None,
    ) -> None:
        assert (
            X.shape[0] == y.shape[0]
        ), "X and y must match along their 0-th dimensions"
        assert len(X.shape) == 2, "X must be 2-dimensional"
        assert len(y.shape) == 1, "y must be 1-dimensional"

        if num_features is not None:
            assert (
                X.shape[1] == num_features
            ), f"second dimension of X must be equal to num_features. X has shape {X.shape}"
        else:
            num_features = X.shape[1]

        if len(cat_idx) > 0:
            assert (
                max(cat_idx) <= num_features - 1
            ), f"max index in cat_idx is {max(cat_idx)}, but num_features is {num_features}"
        assert target_type in ["regression", "classification", "binary"]

        if target_type == "regression":
            assert num_classes == 1
        elif target_type == "binary":
            assert num_classes == 1
        elif target_type == "classification":
            assert num_classes > 2

        self.name = name
        self.cat_idx = cat_idx
        num_mask = np.ones(X.shape[1], dtype=int)
        num_mask[cat_idx] = 0
        self.num_idx = np.where(num_mask)[0]
        self.target_type = target_type
        if not isinstance(self.cat_idx, np.ndarray):
            self.cat_idx = np.array(self.cat_idx)
        self.num_idx = self.num_idx.astype(np.uint)
        self.cat_idx = self.cat_idx.astype(np.uint)

        # X and y are of type np.ndarray; data and target are of type torch.Tensor
        # pred is the prediction (both numpy array and tensor are fine)
        self.X = X
        self.y = y
        self.data = None
        self.target = None
        self.data_dim = None
        self.pred = None
        self.pred_proba = None

        self.num_classes = num_classes
        self.num_features = num_features
        self.cat_dims = cat_dims
        self.split_indeces = split_indeces
        self.split_source = split_source
        self.fold = -1
        self.device = 'cpu'

    def target_encode(self):
        # print("target_encode...")
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)

        # Sanity check
        if self.target_type == "classification":
            assert self.num_classes == len(
                le.classes_
            ), "num_classes was set incorrectly."

    def cat_feature_encode(self):
        # print("cat_feature_encode...")
        if not self.cat_dims is None:
            raise RuntimeError(
                "cat_dims is already set. Categorical feature encoding might be running twice."
            )
        self.cat_dims = []

        # Preprocess data
        for i in range(self.num_features):
            if self.cat_idx and i in self.cat_idx:
                le = LabelEncoder()
                self.X[:, i] = le.fit_transform(self.X[:, i])

                # Setting this?
                self.cat_dims.append(len(le.classes_))

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "cat_idx": self.cat_idx,
            "cat_dims": self.cat_dims,
            "target_type": self.target_type,
            "num_classes": self.num_classes,
            "num_features": self.num_features,
            "num_instances": len(self.X),
            "split_source": self.split_source,
        }
    
    @property
    def features_low(self):
        return self.data.min(axis=0).values

    @property
    def features_high(self):
        return self.data.max(axis=0).values

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def num_instances(self):
        if self.data is None:
            return len(self.X)
        else:
            return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)
    
    def to_string(self):
        return f'{self.name}_fold_{self.fold}'

    @classmethod
    def read(cls, p: Path):
        """read a dataset from a folder"""

        # make sure that all required files exist in the directory
        X_path = p.joinpath("X.npy.gz")
        y_path = p.joinpath("y.npy.gz")
        metadata_path = p.joinpath("metadata.json")
        split_indeces_path = p / "split_indeces.npy.gz"

        assert X_path.exists(), f"path to X does not exist: {X_path}"
        assert y_path.exists(), f"path to y does not exist: {y_path}"
        assert (
            metadata_path.exists()
        ), f"path to metadata does not exist: {metadata_path}"
        assert (
            split_indeces_path.exists()
        ), f"path to split indeces does not exist: {split_indeces_path}"

        # read data
        with gzip.GzipFile(X_path, "r") as f:
            X = np.load(f, allow_pickle=True)
        with gzip.GzipFile(y_path, "r") as f:
            y = np.load(f)
        with gzip.GzipFile(split_indeces_path, "rb") as f:
            split_indeces = np.load(f, allow_pickle=True)

        # read metadata
        with open(metadata_path, "r") as f:
            kwargs = json.load(f)

        kwargs["X"], kwargs["y"], kwargs["split_indeces"] = X, y, split_indeces
        return cls(**kwargs)

    def write(self, p: Path, overwrite=False) -> None:
        """write the dataset to a new folder. this folder cannot already exist"""

        if not overwrite:
            assert ~p.exists(), f"the path {p} already exists."

        # create the folder
        p.mkdir(parents=True, exist_ok=overwrite)

        # write data
        with gzip.GzipFile(p.joinpath("X.npy.gz"), "w") as f:
            np.save(f, self.X)
        with gzip.GzipFile(p.joinpath("y.npy.gz"), "w") as f:
            np.save(f, self.y)
        with gzip.GzipFile(p.joinpath("split_indeces.npy.gz"), "wb") as f:
            np.save(f, self.split_indeces)

        # write metadata
        with open(p.joinpath("metadata.json"), "w") as f:
            metadata = self.get_metadata()
            json.dump(self.get_metadata(), f, indent=4)


def train_test_split_openml(dataset: TabularDataset, fold) -> Tuple[TabularDataset, TabularDataset, TabularDataset]:
    """
    Split the dataset into train, val, test
    Using the official splits provided by openml
    """
    train_idx, val_idx, test_idx = dataset.split_indeces[fold]['train'], dataset.split_indeces[fold]['val'], dataset.split_indeces[fold]['test']
    dataset_train = deepcopy(dataset)
    dataset_train.X = dataset.X[train_idx]
    dataset_train.y = dataset.y[train_idx]
    dataset_train.fold = fold
    dataset_val = deepcopy(dataset)
    dataset_val.X = dataset.X[val_idx]
    dataset_val.y = dataset.y[val_idx]
    dataset_val.fold = fold
    dataset_test = deepcopy(dataset)
    dataset_test.X = dataset.X[test_idx]
    dataset_test.y = dataset.y[test_idx]
    dataset_test.fold = fold
    return dataset_train, dataset_val, dataset_test
