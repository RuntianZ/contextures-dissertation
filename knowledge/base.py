import logging
from typing import Tuple
import torch
import torch.nn as nn 
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from framework.dataset import TabularDataset
from framework.utils import find_nan_row


class Knowledge:
    def __init__(self, logger: logging.Logger, config: dict, device) -> None:
        self.logger = logger
        self.config = config
        self.device = device

    def fit(self, dataset: TabularDataset) -> None:
        """
        Some knowledge needs to fit itself before use
        For example, centering and whitening can be done here
        """
        pass


class PPKnowledge(Knowledge):
    """
    Knowledge provided in the form of P+
    The transform function performs x -> a -> x', y -> b -> y'
    The transform_a function performs x -> a, y -> b
    """

    def transform(self, x: Tensor, y: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        """Implement this!"""
        raise NotImplementedError
    
    def transform_a(self, x: Tensor, y: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        """Optional"""
        raise NotImplementedError


class DualKKnowledge(Knowledge):
    """
    Knowledge provided in the form of the dual kernel kX+
    """

    def kernel(self, x1: Tensor, x2: Tensor, y1: Tensor, y2: Tensor, **kwargs) -> Tensor:
        """
        The dual kernel
        if x2 = None then x2 = x1
        if y2 = None then y2 = y1
        Output: n * n matrix, where n = #samples of the batch
        K[i, j] = k(x1[i], x2[j])
        Implement this!
        """
        raise NotImplementedError
    
    def gram_matrix(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        return self.kernel(x, None, y, None)
    

class RandomWalkKnowledge(PPKnowledge):
    """
    Convert a non-negative kernel to a conditional probability
    P(x'|x) = k(x,x') / \sum_z k(x,z)
    """
    def kernel(self, x1: Tensor, x2: Tensor, y1: Tensor, y2: Tensor, **kwargs) -> Tensor:
        """Output: n * n Gram matrix of the kernel"""
        raise NotImplementedError
    
    def gram_matrix(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        return self.kernel(x, None, y, None)
    
    def transform(self, x: Tensor, y: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        gram = self.gram_matrix(x, y, **kwargs)
        gram = gram + 1e-8
        diag_zero = self.config.get('diag_zero', False)
        if diag_zero:
            gram.fill_diagonal_(0)

        prob = gram / gram.sum(dim=1, keepdim=True)
        if torch.isnan(prob).any():
            self.logger.exception('find_nan_row(prob) = {}'.format(find_nan_row(prob)))
            raise ValueError("Probability values contain NaN")
        if torch.isinf(prob).any():
            raise ValueError("Probability values contain Inf")
        idx = torch.multinomial(prob, num_samples=1).squeeze()
        if (idx < 0).any() or (idx >= batch_size).any():
            raise ValueError(f"Sampled indices are out of bounds: {idx}")
        
        x1 = x[idx]
        return x1, y

