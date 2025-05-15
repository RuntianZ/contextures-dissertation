from sklearn.metrics import get_scorer, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
import numpy as np
import torch
import logging

from anomaly.dataset_anomaly import AnomalyDetectionDataset


def get_metrics(target_type):
    return ['roc_auc', 'pr_auc']
    # return ['accuracy', 'f1', 'roc_auc', 'pr_auc', 'neg_log_loss']


def get_score_function(name: str):
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
    elif name.startswith('pr_auc'):
        def func(y_true, y_score):
            return average_precision_score(y_true, y_score, pos_label=1)

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


def model_selection_metric(target_type):
    # Only for choosing the best setup for each dataset
    return "roc_auc"


def metric_string(s: str) -> str:
    if s.startswith('roc_auc'):
        return 'roc_auc'
    elif s.startswith('pr_auc'):
        return 'pr_auc'
    else:
        return s


def eval_on_anomaly_dataset(dataset: AnomalyDetectionDataset, split: str, logger: logging.Logger) -> dict:
    """
    Returns a dict of
      metric_name : score
    Using dataset.pred and dataset.pred_proba
    """
    logger.debug(f'Eval on Dataset {dataset.name}, fold {dataset.fold}, split {split}')
    y_real = dataset.y
    y_pred = dataset.pred.detach().cpu().numpy() if isinstance(dataset.pred, torch.Tensor) else dataset.pred
    y_pred_proba = dataset.pred_proba.detach().cpu().numpy() if isinstance(dataset.pred_proba,
                                                                           torch.Tensor) else dataset.pred_proba
    metrics = get_metrics(dataset.target_type)
    ans = {}
    k = 0
    for m in metrics:
        scorer, is_pred = get_score_function(m)
        try:
            result = scorer(y_real, y_pred if is_pred else y_pred_proba)
            logger.debug(
                f'Eval: Dataset: {dataset.name}\tFold: {dataset.fold}\tSplit: {split}\tNum classes: {dataset.fold}\tMetric: {m}\tScore: {result}')
        except ValueError:
            logger.error(
                f'ValueError encountered when eval on Dataset {dataset.name}, fold {dataset.fold}, with metric {m}')
            result = None
        ans[split + '_' + metric_string(m)] = result
        if k == 0:
            k = 1
            ans[split + '_performance'] = result
    return ans
