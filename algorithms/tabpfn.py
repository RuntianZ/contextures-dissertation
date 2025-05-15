import numpy as np
import torch
import torch.nn.functional as F
from framework.dataset import TabularDataset
from framework.base import StandaloneModule
from framework.utils import to_numpy
from tabpfn import TabPFNClassifier, TabPFNRegressor  
from copy import deepcopy
import warnings

warnings.simplefilter("ignore")


class TabPFNModel(StandaloneModule):
    """
    TabPFN v2
    https://github.com/PriorLabs/TabPFN
    """
    def build_model(self, train_set: TabularDataset) -> None:
        self.num_classes = train_set.num_classes
        self.logger.debug('tabpfn.device =', self.device)
        n_estimators = self.config["n_estimators"]
        self.model = TabPFNRegressor(device=str(self.device), n_estimators=n_estimators) if train_set.target_type == "regression" else TabPFNClassifier(device=str(self.device), n_estimators=n_estimators)
        self.target_type = train_set.target_type
        self.logger.info(f'TabPFN device = {str(self.device)}')

    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        self.build_model(dataset)
        max_features = self.config["max_features"]
        self.selected_columns = None
        if dataset.num_features > max_features:
            self.logger.info('There are {} features, and will be reduced to {} features'.format(dataset.num_features, max_features))
            self.selected_columns = np.random.choice(dataset.num_features, max_features, replace=False)
        return dataset

    def fit(self, dataset: TabularDataset) -> None:
        # Create a smaller train set if dataset has too many samples
        train_set = deepcopy(dataset)
        subset_rows = self.config["subset_rows"]
        n = len(train_set.data)
        data = train_set.data 
        target = train_set.target
        if n > subset_rows:
            ids = np.random.choice(n, subset_rows, replace=False)
            data = train_set.data[ids]
            target = train_set.target[ids]
        if self.selected_columns is not None:
            data = train_set.data[:, self.selected_columns]
        data = to_numpy(data)
        target = to_numpy(target)
        self.model.fit(data, target)

    def transform(self, dataset: TabularDataset) -> TabularDataset:
        with torch.no_grad():
            X = dataset.data if self.selected_columns is None else dataset.data[:, self.selected_columns]
            X = to_numpy(X)
            dataset.pred = self.model.predict(X)
            dataset.pred_proba = dataset.pred if self.target_type == "regression" else self.model.predict_proba(X)
        return dataset


class TabPFNEmbedding(TabPFNModel):
    def fit(self, dataset: TabularDataset) -> None:
        # Create a smaller train set if dataset has too many samples
        train_set = deepcopy(dataset)
        subset_rows = self.config["subset_rows"]
        train_ratio = self.config["train_ratio"]
        n = len(train_set.data)
        if n > subset_rows:
            ids = np.random.choice(n, subset_rows, replace=False)
        else:
            ids = np.arange(n)

        ids = np.random.permutation(ids)
        n_train = int(n * train_ratio)
        train_ids = ids[:n_train]
        downstream_ids = ids[n_train:]
        dataset.tabpfn_downstream_ids = downstream_ids 

        data = train_set.data[train_ids]
        target = train_set.target[train_ids]
        if self.selected_columns is not None:
            data = train_set.data[:, self.selected_columns]
        data = to_numpy(data)
        target = to_numpy(target)
        self.model.fit(data, target)

    def transform(self, dataset: TabularDataset) -> TabularDataset:
        with torch.no_grad():
            if hasattr(dataset, 'tabpfn_downstream_ids'):
                X = dataset.data[dataset.tabpfn_downstream_ids]
                dataset.target = dataset.target[dataset.tabpfn_downstream_ids]
                dataset.y = dataset.y[dataset.tabpfn_downstream_ids]
                delattr(dataset, 'tabpfn_downstream_ids')
            else:
                X = dataset.data
            if self.selected_columns is not None:
                X = X[:, self.selected_columns]
            X = to_numpy(X)
            self.logger.info('X.shape = {}'.format(X.shape))
            z = torch.tensor(self.model.get_embeddings(X, "test")).float().to(self.device)
            d = z.shape[1]
            dataset.data = z.transpose(0, 1).reshape(d, -1)
            self.logger.info('Embedding.shape = {}'.format(dataset.data.shape))
        return dataset

