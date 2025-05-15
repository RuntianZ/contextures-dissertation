from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from framework.base import LinkedModule
from framework.dataset import TabularDataset
from models.ft_transformer import FTTransformer
from copy import deepcopy
from framework.utils import check_loss_integrity


class FTTransformerModel(LinkedModule):
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        self.forward_in_loss = True
        self.model = FTTransformer.make_default(
            n_num_features=len(dataset.num_idx) if dataset.num_idx is not None else 0,
            cat_cardinalities=dataset.cat_dims,
            n_blocks=self.config["n_blocks"],
            d_out=dataset.num_classes,
        ).to(self.device)
        self.loadable_items = ['model']
        self.num_idx = dataset.num_idx
        self.cat_idx = dataset.cat_idx
        return dataset

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        if len(self.num_idx) > 0:
            x_num = X[:, self.num_idx]
        else:
            x_num = None
        if len(self.cat_idx) > 0:
            x_cat = X[:, self.cat_idx].to(torch.int)
        else:
            x_cat = None
        y_pred = self.model(x_num, x_cat)
        return y_pred, y
    
    def prepare_train(self) -> None:
        self.model.train()

    def prepare_eval(self) -> None:
        self.model.eval()

    def get_loss(self, dataset: TabularDataset, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # self.logger.info('x.shape = {}'.format(x.shape))
        # self.logger.info('self.num_idx = {}'.format(self.num_idx))
        x_num = x[:, self.num_idx] if len(self.num_idx) > 0 else None
        x_cat = x[:, self.cat_idx].to(torch.int) if len(self.cat_idx) > 0 else None
        y_pred = self.model(x_num, x_cat)
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        criterion = self._get_default_criterion(dataset)
        loss = criterion(y_pred, y)
        return loss

    def transform(self, dataset: TabularDataset, **kwargs) -> TabularDataset:
        y_pred = dataset.data
        match dataset.target_type:
            case "binary":
                dataset.pred = (y_pred.ravel() >= 0).long()
                dataset.pred_proba = F.sigmoid(y_pred.ravel())
            case "classification":
                dataset.pred = torch.argmax(y_pred, dim=1)
                dataset.pred_proba = F.softmax(y_pred, dim=1)
            case "regression":
                dataset.pred = y_pred
                dataset.pred_proba = y_pred
            case _:
                raise NotImplementedError
        return dataset



