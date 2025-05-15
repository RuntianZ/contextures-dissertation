import os
import numpy as np
import xgboost as xgb
import torch
from copy import deepcopy
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from framework.dataset import TabularDataset
from framework.base import StandaloneModule
from framework.utils import fix_seed, proba_to_prediction, to_numpy


class BoostingModel(StandaloneModule):
    def build_model(self, dataset: TabularDataset, model_params: dict) -> None:
        """Implement this: self.model = ..."""
        raise NotImplementedError

    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        model_params = self._filter_model_params()
        if self.seed >= 0:
            model_params.setdefault("random_state", self.seed)
        self.build_model(dataset, model_params)
        self.loadable_items = ['model']
        self._inspect_missing_class(dataset)  # This will set self.missing_class
        return dataset

    def fit(self, dataset: TabularDataset) -> None:
        self._inspect_missing_class(dataset)  # This will set self.missing_class
        X = to_numpy(dataset.data)
        y = to_numpy(dataset.target)
        self.model.fit(X, y)

    def transform(self, dataset: TabularDataset) -> TabularDataset:
        X = to_numpy(dataset.data)
        dataset.pred = self.model.predict(X)
        match dataset.target_type:
            case "regression":
                dataset.pred_proba = dataset.pred
            case "classification":
                y_pred = self.model.predict_proba(X)
                # Handle missing classes
                if self.missing_class is not None:
                    all_class_pred = np.zeros((len(y_pred), self.num_classes), dtype=y_pred.dtype)
                    all_class_pred[:, self.model.classes_] = y_pred
                    y_pred = all_class_pred
                dataset.pred_proba = y_pred
            case "binary":
                dataset.pred_proba = self.model.predict_proba(X)
                dataset.pred_proba = dataset.pred_proba[:, 1]
            case _:
                raise RuntimeError(f"Unknown dataset target type {dataset.target_type}")
        return dataset

    def _inspect_missing_class(self, train_set: TabularDataset):
        y = train_set.target
        self.num_classes = train_set.num_classes
        if train_set.target_type == "binary":
            if y.sum() == y.shape[0]:
                self.missing_class = [0]
            elif y.sum() == 0:
                self.missing_class = [1]
        elif train_set.target_type == "classification":
            set_diff = set(list(range(train_set.num_classes))) - set(list(y))
            if len(set_diff) != 0:
                self.missing_class = list(set_diff)


class RandomForestModel(BoostingModel):
    def build_model(self, dataset: TabularDataset, model_params: dict) -> None:
        match dataset.target_type:
            case "regression":
                self.model = RandomForestRegressor(**model_params)
            case "binary":
                self.model = RandomForestClassifier(**model_params)
            case "classification":
                self.model = RandomForestClassifier(**model_params)


class CatBoostModel(BoostingModel):
    def build_model(self, dataset: TabularDataset, model_params: dict) -> None:
        match dataset.target_type:
            case "regression":
                model_params.setdefault("loss_function", "RMSE")
                self.model = CatBoostRegressor(**model_params, verbose=False)
            case "binary":
                model_params.setdefault("loss_function", "MultiClass")
                self.model = CatBoostClassifier(**model_params, verbose=False)
            case "classification":
                model_params.setdefault("loss_function", "MultiClass")
                model_params["classes_count"] = dataset.num_classes
                self.model = CatBoostClassifier(**model_params, verbose=False)


class XGBoostModel(BoostingModel):
    # Note: XGBoost 3.0.0 is buggy. Use 2.1.2 instead.
    def build_model(self, dataset: TabularDataset, model_params: dict) -> None:
        model_params.setdefault("num_class", dataset.num_classes)
        match dataset.target_type:
            case "regression":
                model_params.setdefault("objective", "reg:squarederror")
            case "classification":
                model_params.setdefault("objective", "multi:softprob")
            case "binary":
                model_params.setdefault("objective", "binary:logistic")
        self.model_params = model_params
        self.num_boost_round = self.model_params.pop("num_boost_round")
        self.n_estimators = self.num_boost_round
        if dataset.target_type == "regression":
            self.model_params.pop("num_class")
        self.model = None   # To be defined in fit

    def fit(self, dataset: TabularDataset) -> None:
        self._inspect_missing_class(dataset)  # This will set self.missing_class
        X = to_numpy(dataset.data)
        y = to_numpy(dataset.target)
        dtrain = xgb.DMatrix(X, label=y)
        params = self.model_params.copy()
        params["device"] = self.device
        # self.logger.warning("XGBoost: device = {}".format(self.device))
        self.model = xgb.train(params,
                               dtrain,
                               num_boost_round=self.num_boost_round)
        
    def transform(self, dataset: TabularDataset) -> TabularDataset:
        X = to_numpy(dataset.data)
        X = xgb.DMatrix(X)
        dataset.pred_proba = self.model.predict(X)
        if dataset.target_type == "regression":
            dataset.pred = dataset.pred_proba
        else:
            dataset.pred = proba_to_prediction(dataset.pred_proba)
        return dataset


class KNNModel(BoostingModel):
    def build_model(self, dataset: TabularDataset, model_params: dict) -> None:
        params = deepcopy(model_params)
        params.pop("random_state")
        # Handle very small datasets
        if dataset.num_instances < params['n_neighbors']:
            self.logger.error("Dataset {} too small ({} samples) for {} n_neighbors. Reducing n_neighbors...".format(dataset.name, dataset.num_instances, params['n_neighbors']))
            params['n_neighbors'] = dataset.num_instances
            self.logger.error("New n_neighbors = {}".format(params['n_neighbors']))
        match dataset.target_type:
            case "regression":
                self.model = KNeighborsRegressor(**params)
            case _:
                self.model = KNeighborsClassifier(**params)


class DecisionTreeModel(BoostingModel):
    def build_model(self, dataset: TabularDataset, model_params: dict) -> None:
        match dataset.target_type:
            case "regression":
                self.model = DecisionTreeRegressor(**model_params)
            case _:
                self.model = DecisionTreeClassifier(**model_params)


class XGBoostEmbeddingModel(XGBoostModel):
    """Embedding = [preds from all trees]"""
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        super().init_module(dataset)
        dataset.data_dim = self.n_estimators
        return dataset

    def transform(self, dataset: TabularDataset) -> TabularDataset:
        X = to_numpy(dataset.data)
        X = xgb.DMatrix(X)
        # https://stackoverflow.com/questions/43702514/how-to-get-each-individual-trees-prediction-in-xgboost
        preds = []
        for tree_ in self.model:
            ypred = np.array(tree_.predict(X))
            preds.append(ypred)
            if dataset.target_type == 'classification':
                assert len(ypred.shape) == 2
            else:
                assert len(ypred.shape) == 1
        assert len(preds) == self.n_estimators
        if dataset.target_type == 'classification':
            preds = np.hstack(preds).astype('float')
            assert preds.shape[1] == self.n_estimators * dataset.num_classes
        else:
            preds = np.array(preds).astype('float').T 
        self.logger.debug('preds.shape = {}'.format(preds.shape))
        assert dataset.data.shape[0] == preds.shape[0], f"Mismatch: dataset.data.shape[0] = {dataset.data.shape[0]}, preds.shape[0] = {preds.shape[0]}"
        dataset.data = torch.tensor(preds).float().to(self.device)
        return dataset
