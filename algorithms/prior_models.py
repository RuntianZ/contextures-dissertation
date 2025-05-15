from typing import List, Optional, Tuple
from logging import Logger
import torch
import torch.nn.functional as F
from framework.dataset import TabularDataset
from framework.base import LinkedModule
from framework.utils import str_to_list
from priors.prior import Prior
from models.mlp import MLP


class PriorTransformer(LinkedModule):
    """
    Transforms x into [x, prior(x)], and y into [y, y]
    """
    def get_prior(self, dataset: TabularDataset) -> Prior:
        """Implement this!"""
        raise NotImplementedError
    
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        self.prior = self.get_prior(dataset)
        return dataset
    
    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        X_corrupted = self.prior.transform(X, y).detach()
        X_new = torch.cat([X, X_corrupted])
        y_new = torch.cat([y, y])
        return X_new, y_new


class PriorTrainerWithProjectionHead(LinkedModule):
    """
    A trainer with a loss, and a projection head
    Note: The encoder itself is not included
    Args:
    - projection_head: A string, which is a list of dims, not including the input_dim
      * If [], then no projection_head is used
      * If [128, 128], then a input_dim -> 128 -> 128 head is used
    """
    def get_prior_loss(self, z: torch.Tensor, z_corrupt: torch.Tensor) -> torch.Tensor:
        """
        Implement this!
        z: Latent of original samples (after proj head)
        z_corrupt: Latent of corrupted samples
        """
        raise NotImplementedError

    def get_loss(self, dataset: TabularDataset, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Here, x should be [x_orig, x_corrupted]; y should be [y_orig, y_orig]
        NOTE: Do we also need to consider y_corrupted?
        """
        if self.head_model is not None:
            x = self.head_model(x)
        m = x.shape[0] // 2
        x_orig = x[:m]
        x_corrupted = x[m:]
        y_orig = y[:m]
        y_corrupted = y[m:]
        loss = self.get_prior_loss(x_orig, x_corrupted)
        return loss

    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        self.head_dims = []
        if 'projection_head' in self.config:
            self.head_dims = str_to_list(self.config['projection_head'])
        self.head_model = None
        if len(self.head_dims) > 0:
            head_config = self.config | {
                'input_size': dataset.data_dim,
                'output_size': self.head_dims[-1],
                'hidden_dims': self.head_dims[:-1],
            }
            self.head_model = MLP(head_config, self.logger).to(self.device)
            self.loadable_items = ['head_model']
        self.is_training = True
        return dataset
        
    def prepare_train(self) -> None:
        if self.head_model is not None:
            self.head_model.train()
        self.is_training = True

    def prepare_eval(self) -> None:
        if self.head_model is not None:
            self.head_model.eval()
        self.is_training = False

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        # When train, do nothing; When transform, split out the corrupted, only keeps the original's embeddings
        if self.is_training:
            return X, y 
        else:
            m = X.shape[0] // 2
            return X[:m], y[:m]


class NTXentPriorTrainer(PriorTrainerWithProjectionHead):
    """
    Contrastive learning with NTXent
    Args:
      temperature: default 1.0
      normalize: default True
    """
    def get_prior_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        temp = self.config['temperature']
        normalize = self.config['normalize']
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        if normalize:
            similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        else:
            similarity = torch.einsum('ac,bc->ab', z, z)
        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
        numerator = torch.exp(positives / temp)
        denominator = mask * torch.exp(similarity / temp)
        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss


class VICRegPriorTrainer(PriorTrainerWithProjectionHead):
    """
    Non-contrastive learning with VICReg
    """
    def get_prior_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sim_coeff = self.config['sim_coeff']
        std_coeff = self.config['std_coeff']
        cov_coeff = self.config['cov_coeff']
        batch_size = x.shape[0]
        sim_loss = F.mse_loss(x, y)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = (self.off_diagonal(cov_x).pow_(2).mean() + self.off_diagonal(cov_y).pow_(2).mean()) / 2
        loss = (
            sim_coeff * sim_loss
            + std_coeff * std_loss
            + cov_coeff * cov_loss
        ) / (sim_coeff + std_coeff + cov_coeff)
        return loss

    def off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


