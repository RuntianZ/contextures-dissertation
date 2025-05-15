import numpy as np
from typing import Tuple
import xgboost as xgb
import torch
from torch import Tensor
from typing import List, Tuple

from framework.base import LinkedModule
from framework.dataset import TabularDataset
from framework.utils import to_numpy
from knowledge.base import Knowledge
from knowledge.teacher import TeacherModelKnowledge
from knowledge.learner import ConvolutionLearner, MinmaxConvexCombinationLearner, SCARFKnowledge, YLinearKnowledge

class XGBoostEmbeddingLayer(LinkedModule):
    """
    At train mode, this module will precompute the embedding during init and use that during training
    At test mode, this module will compute the embedding on the fly
    Warnings: 
      1. This layer cannot pass gradients
    """
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        self.forward_with_id = True
        model_params = self._filter_model_params()
        if self.seed >= 0:
            model_params.setdefault("random_state", self.seed)
        self.build_model(dataset, model_params)
        self.loadable_items = ['model']
        if dataset.target_type == 'classification':
            dataset.data_dim = dataset.num_classes * self.n_estimators
        else:
            dataset.data_dim = self.n_estimators
        return dataset

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
        self.target_type = dataset.target_type
        self.num_classes = dataset.num_classes
        self.data_X = to_numpy(dataset.data)
        self.data_y = to_numpy(dataset.target)
        self.model = None
        self.precomputed_embedding = None
        self.is_training = False

    def prepare_train(self) -> None:
        self.is_training = True

    def prepare_eval(self) -> None:
        self.is_training = False

    def fit_xgboost(self, X: np.ndarray, y: np.ndarray) -> None:
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.model_params,
                               dtrain,
                               num_boost_round=self.num_boost_round)
        # self.logger.debug('get_dump = {}'.format(self.model.get_dump(with_stats=True)))
        # for tree_ in self.model:
            # self.logger.debug('Tree')
            # self.logger.debug('{}'.format(tree_.get_dump()))

    def get_embedding(self, X: np.ndarray) -> np.ndarray:
        X = xgb.DMatrix(X)
        preds = []
        for tree_ in self.model:
            ypred = np.array(tree_.predict(X))
            preds.append(ypred)
            if self.target_type == 'classification':
                assert len(ypred.shape) == 2
            else:
                assert len(ypred.shape) == 1
        assert len(preds) == self.n_estimators
        if self.target_type == 'classification':
            preds = np.hstack(preds).astype('float')
            assert preds.shape[1] == self.n_estimators * self.num_classes
        else:
            preds = np.array(preds).astype('float').T 
        return preds

    def forward_id(self, X: Tensor, y: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        assert self.is_training
        if self.model is None:
            self.logger.debug('Fitting XGBoost embedding model')
            self.fit_xgboost(self.data_X, self.data_y)
        if self.precomputed_embedding is None:
            self.logger.debug('Precomputing XGBoost embedding')
            preds = self.get_embedding(self.data_X)
            self.precomputed_embedding = torch.tensor(preds).to(self.device).float()
        if self.data_X is not None: # Save memory
            del self.data_X
            del self.data_y
            self.data_X = None
            self.data_y = None
        Xpred = self.precomputed_embedding[z]
        return Xpred, y

    def forward(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        assert not self.is_training
        X = to_numpy(X)
        preds = self.get_embedding(X)
        Xpred = torch.tensor(preds).to(self.device).float()
        return Xpred, y


class XGBoostKnowledge(TeacherModelKnowledge):
    def fit(self, dataset: TabularDataset) -> None:
        params = ['eta', 'max_depth', 'num_boost_round']
        model_params = {k: self.config[k] for k in params}
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
        self.target_type = dataset.target_type
        self.num_classes = dataset.num_classes     
        X = to_numpy(dataset.data)
        y = to_numpy(dataset.target)
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.model_params,
                               dtrain,
                               num_boost_round=self.num_boost_round)
        super().fit(dataset)

    def teacher_model(self, X: Tensor, y: Tensor) -> Tensor:
        X = to_numpy(X)
        X = xgb.DMatrix(X)
        preds = []
        for tree_ in self.model:
            ypred = np.array(tree_.predict(X))
            preds.append(ypred)
            if self.target_type == 'classification':
                assert len(ypred.shape) == 2
            else:
                assert len(ypred.shape) == 1
        assert len(preds) == self.n_estimators
        if self.target_type == 'classification':
            preds = np.hstack(preds).astype('float')
            assert preds.shape[1] == self.n_estimators * self.num_classes
        else:
            preds = np.array(preds).astype('float').T 
        Xpred = torch.tensor(preds).to(self.device).float()
        return Xpred
    

class XGBoostLearner(ConvolutionLearner):
    def get_knowledge_list(self, dataset: TabularDataset) -> List[Knowledge]:
        k1 = XGBoostKnowledge(self.logger, self.config, self.device)
        return [k1]


class XGBoostSCARFLearner(ConvolutionLearner):
    """xgboost * SCARF"""
    def get_knowledge_list(self, dataset: TabularDataset) -> List[Knowledge]:
        k1 = XGBoostKnowledge(self.logger, self.config, self.device)
        k2 = SCARFKnowledge(self.logger, self.config, self.device)
        return [k1, k2]
    

class XGBoostYLinearLearner(ConvolutionLearner):
    def get_knowledge_list(self, dataset: TabularDataset) -> List[Knowledge]:
        k1 = XGBoostKnowledge(self.logger, self.config, self.device)
        k2 = YLinearKnowledge(self.logger, self.config, self.device)
        return [k1, k2]
    
class YLinearXGBoostLearner(ConvolutionLearner):
    def get_knowledge_list(self, dataset: TabularDataset) -> List[Knowledge]:
        k1 = YLinearKnowledge(self.logger, self.config, self.device)
        k2 = XGBoostKnowledge(self.logger, self.config, self.device)
        return [k1, k2]


class XGBoostPlusYLinearMinmaxLearner(MinmaxConvexCombinationLearner):
    def get_learner_names(self, dataset: TabularDataset) -> List[Knowledge]:
        return ['xgboost_learner', 'ylinear_learner']




