import numpy as np
import torch
import torch.nn.functional as F
from framework.dataset import TabularDataset
from framework.base import StandaloneModule
from models.tabpfn import TabPFNClassifier
from copy import deepcopy
from framework.utils import fix_seed
from framework.files import load_ckpt
import warnings

warnings.simplefilter("ignore")


class TabPFNModel(StandaloneModule):
    def build_model(self, train_set: TabularDataset) -> None:
        if train_set.target_type == "regression":
            raise NotImplementedError("TabPFN doesn't support regression!")
        self.num_classes = train_set.num_classes
        self.logger.debug('tabpfn.device =', self.device)
        self.model = TabPFNClassifier(device=self.device,
                                      N_ensemble_configurations=self.config["n_ensemble"])

    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        self.build_model(dataset)
        max_features = self.config.get('max_features', 100)
        self.selected_columns = None
        if dataset.num_features > max_features:
            self.logger.info('There are {} features, and will be reduced to {} features'.format(dataset.num_features, max_features))
            self.selected_columns = np.random.choice(dataset.num_features, max_features, replace=False)
        return dataset

    def fit(self, dataset: TabularDataset) -> None:
        # Create a smaller train set if dataset has too many samples
        train_set = deepcopy(dataset)
        subset_rows = self.config.get('subset_rows', 1500)
        n = len(train_set.data)
        if n > subset_rows:
            ids = np.random.choice(n, subset_rows, replace=False)
            train_set.data = train_set.data[ids]
            train_set.target = train_set.target[ids]
        if self.selected_columns is not None:
            train_set.data = train_set.data[:, self.selected_columns]
        self.model.fit(train_set.data, train_set.target, overwrite_warning=True)

    def transform(self, dataset: TabularDataset) -> TabularDataset:
        with torch.no_grad():
            X = dataset.data if self.selected_columns is None else dataset.data[:, self.selected_columns]
            dataset.pred = self.model.predict(X, return_winning_probability=False)
            dataset.pred_proba = self.get_logits(dataset)
        return dataset

    def get_logits(self, dataset: TabularDataset) -> torch.Tensor:
        X = dataset.data if self.selected_columns is None else dataset.data[:, self.selected_columns]
        y_pred = self.model.predict_proba(X, return_logits=False)
        if len(self.model.classes_) < self.num_classes and dataset.target_type == "classification":
            all_class_pred = torch.zeros((len(y_pred), self.num_classes)).to(y_pred.dtype)
            all_class_pred[:, self.model.classes_] = y_pred
            y_pred = all_class_pred
        if dataset.target_type == "binary":
            y_pred = y_pred[:, 1]  
        return y_pred
