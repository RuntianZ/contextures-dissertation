import numpy as np
import time 
from copy import deepcopy
from typing import Tuple

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import torch
from torch import Tensor
from torch.utils.data import Dataset

from framework.dataset import TabularDataset
from framework.base import StandaloneModule


class TabzillaPreprocessor(StandaloneModule):
    """
    This preprocessor also creates the tensors in data and target
    Hyperparameters:
      - scaler_type: If set to Quantile, will use Quantile Transformer for numerical columns
    """
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        self.scaler = None
        scaler_type = self.config.get('scaler_type', None)
        if scaler_type == 'Quantile':
            self.scaler = QuantileTransformer(n_quantiles=min(len(dataset.X), 1000))
        self.numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer())])
        self.logger.debug(f'dataset.num_idx = {dataset.num_idx}')
        self.logger.debug(f'dataset.cat_idx = {dataset.cat_idx}')
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", self.numeric_transformer, dataset.num_idx),
                ("pass", "passthrough", dataset.cat_idx),
                # ("cat", categorical_transformer, categorical_features),
            ],
            # remainder="passthrough",
        )
        self.fully_nan_num_idcs = None
        return dataset

    def reorder_columns(self, dataset: TabularDataset, X: np.ndarray) -> np.ndarray:
        """Re-order columns (ColumnTransformer permutes them)"""
        num_mask = np.ones(X.shape[1], dtype=int)
        num_mask[dataset.cat_idx] = 0
        perm_idx = []
        running_num_idx = 0
        running_cat_idx = 0
        for is_num in num_mask:
            if is_num > 0:
                perm_idx.append(running_num_idx)
                running_num_idx += 1
            else:
                perm_idx.append(running_cat_idx + len(dataset.num_idx))
                running_cat_idx += 1
        assert running_num_idx == len(dataset.num_idx)
        assert running_cat_idx == len(dataset.cat_idx)
        X = X[:, perm_idx]
        return X

    def fit(self, dataset: TabularDataset) -> None:
        # The imputer drops columns that are fully NaN. So, we first identify columns that are fully NaN and set them to
        # zero. This will effectively drop the columns without changing the column indexing and ordering that many of
        # the functions in this repository rely upon.
        self.fully_nan_num_idcs = np.nonzero(
            (~np.isnan(dataset.X[:, dataset.num_idx].astype("float"))).sum(axis=0) == 0
        )[0]
        X = deepcopy(dataset.X)
        self.logger.debug(f'X.dtype = {X.dtype}')
        if self.fully_nan_num_idcs.size > 0:
            X[:, dataset.num_idx[self.fully_nan_num_idcs]] = 0
        X = self.column_transformer.fit_transform(X)
        X = self.reorder_columns(dataset, X)

        # Apply scaler to the numerical columns
        if self.scaler is not None:
            X[:, dataset.num_idx] = self.scaler.fit_transform(X[:, dataset.num_idx])

        # Save this X to the dataset for future use
        setattr(dataset, '_tabzilla_preprocessed_X', X)

    def transform(self, dataset: TabularDataset) -> TabularDataset:
        X = getattr(dataset, '_tabzilla_preprocessed_X', None)
        if X is None:
            X = deepcopy(dataset.X)
            if self.fully_nan_num_idcs.size > 0:
                X[:, dataset.num_idx[self.fully_nan_num_idcs]] = 0
            X = self.column_transformer.transform(X)
            X = self.reorder_columns(dataset, X)
            if self.scaler is not None:
                X[:, dataset.num_idx] = self.scaler.transform(X[:, dataset.num_idx])
        else:
            delattr(dataset, '_tabzilla_preprocessed_X')
        y_type = torch.float32 if dataset.target_type == "regression" else torch.long
        dataset.data = torch.tensor(X.astype('float'), dtype=torch.float32).to(dataset.device)
        dataset.target = torch.tensor(dataset.y, dtype=y_type).to(dataset.device)
        dataset.data_dim = dataset.data.shape[1]
        # dataset.data_raw = torch.clone(dataset.data).detach()
        return dataset
