from logging import Logger
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple
import numpy as np
import torch
from torch import Tensor

from framework.base import StandaloneModule, LinkedModule
from framework.dataset import TabularDataset
from framework.utils import to_numpy


class DummyStandaloneModule(StandaloneModule):
    """Do nothing"""
    def __init__(self, config: dict, default_config: dict, logger: Logger, name: str, ckpt_name: str = None, seed: int = -1) -> None:
        super().__init__(config, default_config, logger, name, ckpt_name, seed)

    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        return dataset
    
    def to_string(self, ckpt: bool = False, include_fit: bool = False) -> str:
        return self.ckpt_name if ckpt else self.name
    
    def transform(self, dataset: TabularDataset) -> TabularDataset:
        dataset.data = dataset.data.to(self.device)
        dataset.target = dataset.target.to(self.device)
        return dataset

    def fit(self, dataset: TabularDataset) -> None:
        pass


class DummyLinkedModule(LinkedModule):
    """Do nothing"""
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        return dataset
    
    def to_string(self, ckpt: bool = False, include_fit: bool = False) -> str:
        return 'dummy-linked'

    def forward(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        return X, y


class DummyModule(DummyStandaloneModule):
    """Used for mixture modules in save and load path"""
    def __init__(self, name: str, ckpt_name: str) -> None:
        super().__init__({}, {}, None, name, ckpt_name)

    def to_string(self, ckpt: bool = False, include_fit: bool = False) -> str:
        return self.ckpt_name


class LinearModel(StandaloneModule):
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        all_model_types = {
            "regression": "ridge",
            "classification": "logistic",
            "binary": "logistic",
        }
        model_type = all_model_types[dataset.target_type]
        match model_type:
            case "ridge":
                self.model = Ridge(alpha=self.config['ridge'], fit_intercept=self.config['fit_intercept'])
            case "logistic":
                self.model = LogisticRegression(max_iter=self.config['max_iter'], fit_intercept=self.config['fit_intercept'])
            case _:
                raise NotImplementedError('Linear model type {} is not implemented'.format(model_type))
        self.loadable_items = ['model']
        return dataset

    def fit(self, dataset: TabularDataset) -> None:
        self.logger.debug('Calling linear_model.fit')
        if self.has_loaded:
            self.logger.debug('linear_model has been loaded, skip fit')
        else:
            X = to_numpy(dataset.data)
            y = to_numpy(dataset.target)
            self.target_type = dataset.target_type
            self.model.fit(X, y)

    def transform(self, dataset: TabularDataset) -> TabularDataset:
        # self.logger.debug('Calling linear_model.transform')
        X = to_numpy(dataset.data)
        self.logger.debug('Linear model data shape = {}'.format(X.shape))
        dataset.pred = self.model.predict(X)
        if dataset.target_type == "regression":
            dataset.pred_proba = dataset.pred
        else:
            dataset.pred_proba = self.model.predict_proba(X)
            if dataset.target_type == "binary":
                dataset.pred_proba = dataset.pred_proba[:, 1]
            elif dataset.num_classes > 0:
                # Handling labels that didn't appear in the train set
                all_classes = np.arange(dataset.num_classes)
                pred_old = dataset.pred_proba
                dataset.pred_proba = np.zeros((pred_old.shape[0], all_classes.size))
                dataset.pred_proba[:, all_classes.searchsorted(self.model.classes_)] = pred_old
        return dataset


class StandardScalerModel(StandaloneModule):
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        self.scaler = StandardScaler()
        self.loadable_items = ['scaler']
        return dataset
    
    def fit(self, dataset: TabularDataset) -> None:
        # self.logger.debug(f'Calling standard_scaler.fit, dataset size = {len(dataset.data)}')
        X = to_numpy(dataset.data).astype('float')
        # self.logger.debug(f'X.shape = {X.shape}')
        self.scaler.fit(X)
        # self.logger.debug(f'scaler.mean = {self.scaler.mean_}')
        # self.logger.debug(f'scaler.var = {self.scaler.var_}')

    def transform(self, dataset: TabularDataset) -> TabularDataset:
        # self.logger.debug(f'Calling standard_scaler.transform, dataset size = {len(dataset.data)}')
        X = to_numpy(dataset.data).astype('float')
        X = self.scaler.transform(X)
        dataset.data = torch.tensor(X).float().to(self.device)
        return dataset


class NumericalStandardScalerModel(StandaloneModule):
    """Only apply standard scaler to numerical columns"""
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        self.num_idx = dataset.num_idx
        self.scaler = StandardScaler()
        self.loadable_items = ['scaler']
        return dataset
    
    def fit(self, dataset: TabularDataset) -> None:
        if len(self.num_idx) > 0:
            X = to_numpy(dataset.data)
            X_num = X[:, self.num_idx]
            self.scaler.fit(X_num)

    def transform(self, dataset: TabularDataset) -> TabularDataset:
        if len(self.num_idx) > 0:
            X = to_numpy(dataset.data)
            X_num = X[:, self.num_idx]
            X_num = self.scaler.transform(X_num)
            X[:, self.num_idx] = X_num
            dataset.data = torch.tensor(X).float().to(self.device)
        return dataset


class TargetStandardScalerModel(StandaloneModule):
    """Apply standard scaler to target for regression datasets"""
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        self.mu = 0.0
        self.std = 0.0
        return dataset
    
    def fit(self, dataset: TabularDataset) -> None:
        if dataset.target_type != 'regression':
            return 
        assert len(dataset.target.shape) == 1
        self.mu = dataset.target.mean().item()
        y = dataset.target - self.mu
        self.std = y.std().item()
        eps = 1e-8
        self.std = max(self.std, eps)
        self.logger.debug('target scaler: mu = {}, std = {}'.format(self.mu, self.std))

    def transform(self, dataset: TabularDataset) -> TabularDataset:
        if dataset.target_type == 'regression':
            dataset.target = (dataset.target - self.mu) / self.std
            def pred_transform(y: Tensor) -> Tensor:
                y = y * self.std + self.mu
                return y
            dataset.register_pred_transform(pred_transform)
            dataset.register_pred_proba_transform(pred_transform)
        return dataset


class PCAModel(StandaloneModule):
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        n_components = min([dataset.num_instances, dataset.num_features, self.config['n_components']])
        self.pca = PCA(n_components)
        self.loadable_items = ['pca']
        return dataset

    def fit(self, dataset: TabularDataset) -> None:
        X = to_numpy(dataset.data).astype('float')
        self.pca.fit(X)

    def transform(self, dataset: TabularDataset) -> TabularDataset:
        X = to_numpy(dataset.data).astype('float')
        X = self.pca.transform(X)
        dataset.data = torch.tensor(X).float().to(self.device)
        return dataset

