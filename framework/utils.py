import logging
import numpy as np 
import random 
import torch
from logging import Logger
import ast 
import importlib
from pathlib import Path
from framework.files import load_all_algs
from framework.dataset import train_test_split_openml, TabularDataset


class DatasetLoader:
    def load(self, config: dict, logger: logging.Logger) -> dict:
        raise NotImplementedError


class PredictionDatasetLoader(DatasetLoader):
    def load(self, config: dict, logger: logging.Logger) -> dict:
        dataset_path = Path(config['dataset_path'])
        num_folds = config['folds']
        dataset = TabularDataset.read(dataset_path)
        logger.debug(f"dataset name: {dataset.name}")
        logger.debug(f"target type: {dataset.target_type}")
        logger.debug(f"number of target classes: {dataset.num_classes}")
        logger.debug(f"number of features: {dataset.num_features}")
        logger.debug(f"number of instances: {len(dataset.X)}")
        logger.debug(f"indeces of categorical features: {dataset.cat_idx}")
        all_folds = []
        for fold in range(num_folds):
            train, val, test = train_test_split_openml(dataset, fold)
            all_folds.append({'train_set': train, 'val_set': val, 'test_set': test})
        return {dataset.name: all_folds}


def get_device(config: dict = None):
    if config is not None:
        device = config.get("device", None)
        if device is not None:
            return device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def fix_seed(seed, offset=0):
    if seed != -1:
        random.seed(seed + offset)
        np.random.seed(seed + offset + 10)
        torch.manual_seed(seed + offset + 20)
        torch.cuda.manual_seed(seed + offset + 30)

def proba_to_prediction(proba):
    if len(proba.shape) == 1:
        a = np.sign(proba - 0.5)
        a[a == -1] = 0
        return a
    else:
        return np.argmax(proba, axis=1)
    

def to_numpy(X):
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    return X


def check_loss_integrity(x: torch.Tensor, logger: Logger, raise_error=True) -> bool:
    a = (torch.isnan(x) | torch.isinf(x)).any()
    if a:
        logger.error('Nan or inf tensor encountered when computing the loss')
        if raise_error:
            raise RuntimeError('Nan or inf tensor encountered')
    return a


def z_score(numbers):
    numbers = np.array(numbers)
    return (numbers - np.mean(numbers)) / np.std(numbers)


def find_nan_row(matrix):
    nan_mask = torch.isnan(matrix)
    nan_rows = torch.any(nan_mask, dim=1)
    return torch.where(nan_rows)[0]


def compute_gaussian_kernel(x: torch.Tensor, gauss_temp: float) -> torch.Tensor:
    """
    Computes the Gaussian kernel matrix for the input tensor x.
    G(x, x') = exp(-|x - x'|^2 / t)
    """
    n = x.shape[0]
    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
    cov = x @ x.T
    cov_diag = torch.diagonal(cov)
    cov_r = torch.matmul(cov_diag.reshape((n, 1)), torch.ones((1, n)).to(x.device))
    d_sqr = cov_r + cov_r.T - 2 * cov  # d_sqr[i, j] = | x[i] - x[j] |_2^2
    kernel_values = torch.exp(-d_sqr / gauss_temp)
    # Prevent inf
    max_value = kernel_values[~torch.isinf(kernel_values)].max()
    kernel_values[torch.isinf(kernel_values)] = max_value * 1000 + 1000
    return kernel_values


def str_to_list(x) -> list:
    if isinstance(x, list):
        return x
    assert isinstance(x, str)
    p = True
    while p:
        p = False
        if x[0] in ['"', "'"]:
            x = x[1:]
            p = True
        if x[-1] in ['"', "'"]:
            x = x[:-1]
            p = True
    x = ast.literal_eval(x)
    assert isinstance(x, list)
    return x


def import_module(class_path: str):
    a = class_path.rfind('.')
    class_name = class_path[a+1:]
    module = importlib.import_module(class_path[:a])
    method_class = getattr(module, class_name)
    return method_class


def create_module(md: dict, default_config: dict, logger: logging.Logger, seed: int):
    all_algs = load_all_algs()
    md_name = md['module']
    assert md_name in all_algs, 'Module {} is not implemented'.format(md_name)
    alg = all_algs[md_name]
    class_path = alg['class']
    hyperparameters = alg.get('args', [])
    module_config = {}
    for k in hyperparameters:
        if isinstance(k, dict):   # This arg has a default value
            item = list(k.items())[0]
            module_config[item[0]] = md.get(item[0], item[1])
        else:
            assert k in md, f"In module {md_name}, hyperparameter {k} is not specified and does not have a default value"
            module_config[k] = md[k]
    module_class = import_module(class_path)
    ckpt_name = None
    if 'ckpt_name' in alg:
        ckpt_name = alg['ckpt_name']
    md_obj = module_class(module_config, default_config, logger, md_name, ckpt_name, seed)
    md_obj.args_keys = md_obj.args_keys + list(module_config.keys())
    return md_obj


