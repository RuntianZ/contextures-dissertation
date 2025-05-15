from sklearn.metrics import get_scorer, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
import numpy as np 
import torch
import logging
from typing import Tuple, Callable

from framework.dataset import TabularDataset
from framework.utils import to_numpy


class Evaluator:
    def get_metrics(self, target_type: str) -> list:
        """
        Input: target_type of the dataset
        Output: A list of strings. Each string is the name of a metric
        """
        raise NotImplementedError
    
    def model_selection_metric(self, target_type: str) -> str:
        """
        Input: target_type of the dataset
        Output: Which metric should be used for model selection during validation
        """
        raise NotImplementedError
    
    def get_score_function(self, name: str) -> Tuple[Callable, bool]:
        """
        Input: The name of a metric
        Output: (score_func, pred)
          If pred = True, will call score_func(y_true, y_pred)
          If pred = False, will call score_func(y_true, y_proba)
        """
        raise NotImplementedError
    
    def metric_string(self, s: str) -> str:
        """The metric string you want in the result csv file"""
        return s
    

class PredictionEvaluator(Evaluator):
    def get_metrics(self, target_type):
        match target_type:
            case "regression":
                metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'explained_variance']
            case "binary":
                metrics = ['accuracy', 'f1', 'roc_auc', 'neg_log_loss']
            case "classification":
                metrics = ['accuracy', 'f1_macro', 'roc_auc_ovr', 'neg_log_loss']
            case _:
                metrics = []
        return metrics 

    def model_selection_metric(self, target_type):
        if target_type == "regression":
            return "neg_mean_squared_error"
            # return "r2"
        else:
            return "accuracy"

    def get_score_function(self, name: str):
        """
        Returns: scorer, bool (True if pred, False if proba)
        """
        pred = True
        if name.startswith('f1_'):
            avg = name[3:]
            def func(y_true, y_pred):
                return f1_score(y_true, y_pred, average=avg)
            return func, True
        elif name.startswith('roc_auc_'):
            mc = name[8:]
            def func(y_true, y_score):
                return roc_auc_score(y_true, y_score, multi_class=mc)
            pred = False
        else:
            score_func = get_scorer(name)._score_func
            if name in ['accuracy', 'f1']:
                # Needs prediction instead of probability
                def func(y_true, y_pred):
                    return score_func(y_true, y_pred)
                return func, True
            elif name.startswith('neg_'):
                def func(y_true, y_score):
                    return -score_func(y_true, y_score)
                pred = False
            else:
                func = score_func
                pred = False
        def func2(y_true, y_score):
            """
            Handling missing classes
            If one class is not present in y_true
            That class will be eliminated, and y_score will be normalized again
            """
            if len(y_score.shape) == 2 and y_score.shape[1] == 1:
                y_score = y_score.ravel()
            if len(y_score.shape) == 1:
                # Ignore binary cases
                return func(y_true, y_score)
            classes = unique_labels(y_true)
            y1 = y_score[:, classes]
            y1 = y1 / y1.sum(axis=1).reshape((-1, 1))
            return func(y_true, y1)
        return func2, pred

    def metric_string(self, s: str) -> str:
        if s.startswith('f1'):
            return 'f1'
        elif s.startswith('roc_auc'):
            return 'roc_auc'
        else:
            return s


def eval_on_dataset(evaluator: Evaluator, dataset: TabularDataset, split: str, logger: logging.Logger) -> dict:
    """
    Returns a dict of
      metric_name : score
    Using dataset.pred and dataset.pred_proba
    """
    logger.debug(f'Eval on Dataset {dataset.name}, fold {dataset.fold}, split {split}')
    y_real = dataset.y
    y_pred = to_numpy(dataset.get_pred_for_eval())
    y_pred_proba = to_numpy(dataset.get_pred_proba_for_eval())
    metrics = evaluator.get_metrics(dataset.target_type)
    ans = {}
    k = 0
    for m in metrics:
        scorer, is_pred = evaluator.get_score_function(m)
        try:
            result = scorer(y_real, y_pred if is_pred else y_pred_proba)
            logger.debug(f'Eval: Dataset: {dataset.name}\tFold: {dataset.fold}\tSplit: {split}\tNum classes: {dataset.fold}\tMetric: {m}\tScore: {result}')
        except ValueError:
            logger.error(f'ValueError encountered when eval on Dataset {dataset.name}, fold {dataset.fold}, with metric {m}')
            result = None
        ans[split + '_' + evaluator.metric_string(m)] = result
        if k == 0:
            k = 1
            ans[split + '_performance'] = result
    return ans

