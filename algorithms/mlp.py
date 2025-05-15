import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from typing import List, Optional, Tuple

from framework.dataset import TabularDataset
from framework.base import LinkedModule, StandaloneModule
from framework.utils import str_to_list
from models.mlp import MLP


class MLPLayer(LinkedModule):
    """
    Transform x into mlp(x)
    Args:
      dims: Including the output dim
      e.g. [128, 128] means input_dim -> 128 -> 128
    """
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        self.logger.debug(f'config.dims = {self.config["dims"]}')
        model_dims = str_to_list(self.config["dims"])
        self.logger.debug(f'model_dims = {model_dims}')
        output_dim = model_dims[-1]
        self.logger.debug(f'output_dim = {output_dim}')
        self.logger.debug(f'hidden_dims = {model_dims[:-1]}')
        model_config = self.config | {
            'input_size': dataset.data_dim,
            'output_size': output_dim,
            'hidden_dims': model_dims[:-1],
        }
        self.model = MLP(model_config, self.logger).to(self.device)
        self.loadable_items = ['model']
        dataset.data_dim = output_dim
        return dataset
    
    def prepare_train(self) -> None:
        self.model.train()

    def prepare_eval(self) -> None:
        self.model.eval()

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        z = self.model(X)
        return z, y
    

class LinearLayer(MLPLayer):
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        input_dim = dataset.data_dim
        output_dim = self.config['output_dim']
        bias = self.config['bias']
        self.logger.debug('Linear layer: input_dim = {}, output_dim = {}, bias = {}'.format(input_dim, output_dim, bias))
        self.model = Linear(input_dim, output_dim, bias=bias).to(self.device)
        self.loadable_items = ['model']
        dataset.data_dim = output_dim
        return dataset   



class FinalLinearLayer(LinearLayer):
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        input_dim = dataset.data_dim
        if dataset.target_type == 'classification':
            output_dim = dataset.num_classes
        else:
            output_dim = 1
        bias = self.config['bias']
        self.logger.debug('Linear layer: input_dim = {}, output_dim = {}, bias = {}'.format(input_dim, output_dim, bias))
        self.model = Linear(input_dim, output_dim, bias=bias).to(self.device)
        self.loadable_items = ['model']
        dataset.data_dim = output_dim
        self.latent = None
        return dataset   

    def prepare_train(self) -> None:
        self.model.train()
        self.latent = None

    def prepare_eval(self) -> None:
        self.model.eval()
        self.latent = []

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        if self.latent is not None:
            self.latent.append(X.detach().to(self.device))
        z = self.model(X)
        return z, y

    def transform(self, dataset: TabularDataset, **kwargs) -> TabularDataset:
        assert dataset.data.shape[0] == dataset.y.shape[0]
        match dataset.target_type:
            case 'binary':
                dataset.pred_proba = dataset.data.flatten()
                dataset.pred = (dataset.pred_proba >= 0).long()
                dataset.pred_proba = torch.sigmoid(dataset.pred_proba)
            case 'classification':
                dataset.pred = dataset.data.argmax(dim=1)
                dataset.pred_proba = F.softmax(dataset.data, dim=1)
            case 'regression':
                dataset.pred = dataset.data.flatten()
                dataset.pred_proba = dataset.pred

        assert dataset.pred.shape[0] == dataset.y.shape[0]
        dataset.final_linear_latent = torch.cat(self.latent)
        self.latent = None
        return dataset

    def get_loss(self, dataset: TabularDataset, X: Tensor, y: Tensor) -> Tensor:
        criterion = self._get_default_criterion(dataset)
        if dataset.target_type != 'classification':
            assert X.shape[1] == 1
            X = X.flatten()
        return criterion(X, y)


class FinalLinearLatentFeatures(StandaloneModule):
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        return dataset
    
    def fit(self, dataset: TabularDataset) -> None:
        pass

    def transform(self, dataset: TabularDataset) -> TabularDataset:
        assert hasattr(dataset, 'final_linear_latent')
        dataset.data = dataset.final_linear_latent
        return dataset 



