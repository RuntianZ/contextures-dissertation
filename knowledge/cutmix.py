from typing import Tuple
import torch
from torch import Tensor
from framework.dataset import TabularDataset
from knowledge.base import PPKnowledge


class CutmixKnowledge(PPKnowledge):
    def fit(self, dataset: TabularDataset) -> None:
        self.corruption_rate = self.config.get('corruption_rate', 0.5)

    def transform(self, x: Tensor, y: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        m = x.shape[0] // 2
        f = x.shape[1]
        mat1 = torch.randint(0, m, (m, f))
        mat2 = torch.randint(0, m, (m, f))

        rows = mat1.flatten()
        cols = torch.arange(f).repeat(1, m).flatten()
        x1 = x[m:2*m]
        x1 = x1[rows, cols].reshape((m, f))
        corruption_mask = torch.rand_like(x1, device=x.device) < self.corruption_rate
        x1 = torch.where(corruption_mask, x1, x[:m])  # x1 is noise, x[:m] is clean data

        rows = mat2.flatten()
        x2 = x[rows, cols].reshape((m, f))
        corruption_mask = torch.rand_like(x2, device=x.device) < self.corruption_rate
        x2 = torch.where(corruption_mask, x2, x[m:2*m])

        x_out = x.clone()
        x_out[:m] = x1
        x_out[m:2*m] = x2
        return x_out, y


class CutoutKnowledge(PPKnowledge):
    def fit(self, dataset: TabularDataset) -> None:
        self.corruption_rate = self.config.get('corruption_rate', 0.5)
        self.cutout_value = self.config.get('cutout_value', 0.0)

    def transform(self, x: Tensor, y: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        mask = torch.rand(x.shape[1]).to(x.device) < self.corruption_rate
        x_cutout = x * (~mask[None, :]) + self.cutout_value * mask[None, :]  # Choose cutout_value at rate cr
        return x_cutout, y
