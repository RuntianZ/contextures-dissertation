import numpy as np
# import time
from copy import deepcopy
from math import ceil

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
from framework.utils import to_numpy

from anomaly.dataset_anomaly import AnomalyDetectionDataset
from framework.base import StandaloneModule


class MinMaxScalerModel(StandaloneModule):
    def init_module(self, dataset: AnomalyDetectionDataset) -> AnomalyDetectionDataset:
        self.scaler = MinMaxScaler()
        return dataset

    def fit(self, dataset: AnomalyDetectionDataset) -> None:
        # self.logger.debug(f'Calling standard_scaler.fit, dataset size = {len(dataset.data)}')
        X = to_numpy(dataset.data)
        # self.logger.debug(f'X.shape = {X.shape}')
        self.scaler.fit(X)
        # self.logger.debug(f'scaler.mean = {self.scaler.mean_}')
        # self.logger.debug(f'scaler.var = {self.scaler.var_}')

    def transform(self, dataset: AnomalyDetectionDataset) -> AnomalyDetectionDataset:
        # self.logger.debug(f'Calling standard_scaler.transform, dataset size = {len(dataset.data)}')
        X = to_numpy(dataset.data)
        X = self.scaler.transform(X)
        dataset.data = torch.tensor(X).float().to(self.device)
        return dataset


class AnomalyDetectionPreprocessor(StandaloneModule):
    """
    This preprocessor also creates the tensors in data and target
    Hyperparameters:
      - scaler_type: If set to Minmax, will use Minmax Transformer for numerical columns
    """

    def init_module(self, dataset: AnomalyDetectionDataset) -> AnomalyDetectionDataset:
        self.scaler = None
        scaler_type = self.config.get('scaler_type', None)
        # TODO: changed here
        # if scaler_type == 'MinMax':
        #     self.scaler = MinMaxScaler()
        self.logger.debug(f'dataset.num_idx = {dataset.num_idx}')
        self.logger.debug(f'dataset.cat_idx = {dataset.cat_idx}')
        # TODO: changed here
        # self.column_transformer = ColumnTransformer(
        #     transformers=[
        #         ("num", self.numeric_transformer, dataset.num_idx),
        #         ("pass", "passthrough", dataset.cat_idx),
        #     ],
        # )
        self.fully_nan_num_idcs = None
        return dataset

    def reorder_columns(self, dataset: AnomalyDetectionDataset, X: np.ndarray) -> np.ndarray:
        """ No columns transformer here, so columns are not permuted here """
        raise NotImplementedError

    def fit(self, dataset: AnomalyDetectionDataset) -> None:
        # note that this is only called for train dataset
        # TODO: don't know if we need imputer here...
        # The imputer drops columns that are fully NaN. So, we first identify columns that are fully NaN and set them to
        # zero. This will effectively drop the columns without changing the column indexing and ordering that many of
        # the functions in this repository rely upon.
        # TODO: changed here
        self.fully_nan_num_idcs = np.nonzero(
            (~np.isnan(dataset.X[:, dataset.num_idx].astype("float"))).sum(axis=0) == 0
        )[0]
        X = deepcopy(dataset.X)
        y = deepcopy(dataset.y)
        self.logger.debug(f'X.dtype = {X.dtype}')
        # TODO: changed here
        if self.fully_nan_num_idcs.size > 0:
            X[:, dataset.num_idx[self.fully_nan_num_idcs]] = 0
        # X = self.column_transformer.fit_transform(X)
        # X = self.reorder_columns(dataset, X)

        # Apply scaler to the numerical columns
        # all columns are numeric for anomaly dataset
        # if self.scaler is not None:
        #     X = self.scaler.fit_transform(X)

        y = self._process_anomaly_label(y, la=dataset.la, at_least_one_labeled=dataset.at_least_one_labeled)

        # Save this X to the dataset for future use
        setattr(dataset, '_anomaly_preprocessed_X', X)
        setattr(dataset, '_anomaly_preprocessed_y', y)

    def transform(self, dataset: AnomalyDetectionDataset) -> AnomalyDetectionDataset:
        X = getattr(dataset, '_anomaly_preprocessed_X', None)
        y = getattr(dataset, '_anomaly_preprocessed_y', None)
        # TODO: only test and val go to X is None, otherwise is train
        if X is None or y is None:
            X = deepcopy(dataset.X)
            y = deepcopy(dataset.y)
            # TODO: changed here
            if self.fully_nan_num_idcs.size > 0:
                X[:, dataset.num_idx[self.fully_nan_num_idcs]] = 0
            # X = self.column_transformer.transform(X)
            # X = self.reorder_columns(dataset, X)
            # Apply scaler to the numerical columns
            # all columns are numeric for anomaly dataset
            # if self.scaler is not None:
            #     X = self.scaler.fit_transform(X)
            # IMPORTANT: we don't need la for test and val dataset
            # y = self._process_anomaly_label(y, la=dataset.la, at_least_one_labeled=dataset.at_least_one_labeled)
        else:
            delattr(dataset, '_anomaly_preprocessed_X')
            delattr(dataset, '_anomaly_preprocessed_y')
        # it only has binary, so y_type is always long
        y_type = torch.long
        dataset.data = torch.tensor(X.astype('float'), dtype=torch.float32).to(self.device)
        dataset.target = torch.tensor(y, dtype=y_type).to(self.device)
        dataset.data_dim = dataset.data.shape[1]
        # dataset.data_raw = torch.clone(dataset.data).detach()
        return dataset

    def _process_anomaly_label(self, y, la, at_least_one_labeled):
        idx_normal = np.where(y == 0)[0]
        idx_anomaly = np.where(y == 1)[0]

        # TODO: changed here
        # calculate the index of anomaly data that we want to keep as anomaly
        # however, we don't allow "changed" data, so add a new option here
        # if la is None, then we don't process the data
        if la is not None:
            self.logger.warning("Don't set 'la' because it modifies the training data")
            if type(la) == float:
                if at_least_one_labeled:
                    idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(la * len(idx_anomaly)), replace=False)
                else:
                    idx_labeled_anomaly = np.random.choice(idx_anomaly, int(la * len(idx_anomaly)), replace=False)
            elif type(la) == int:
                if la > len(idx_anomaly):
                    raise AssertionError(
                        f'the number of labeled anomalies are greater than the total anomalies: {len(idx_anomaly)} !')
                else:
                    idx_labeled_anomaly = np.random.choice(idx_anomaly, la, replace=False)
            else:
                raise NotImplementedError

            idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)

            # unlabel data = normal data + unlabeled anomalies (which is considered as contamination)
            idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

            del idx_anomaly, idx_unlabeled_anomaly

            # the label of unlabeled data is 0, and that of labeled anomalies is 1
            y[idx_unlabeled] = 0
            y[idx_labeled_anomaly] = 1
        return y
