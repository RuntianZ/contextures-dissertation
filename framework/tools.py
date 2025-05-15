from typing import Type, List, Union
import torch


def process_config_value(x) -> str:
    if x is None:
        x = 'None'
    x = str(x)
    while x[0] == '"' or x[0] == "'":
        x = x[1:]
    while x[-1] == '"' or x[-1] == "'":
        x = x[:-1]
    x = x.replace('none', 'None')
    return x


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
