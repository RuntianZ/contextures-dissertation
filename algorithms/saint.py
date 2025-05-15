import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from framework.utils import fix_seed, check_loss_integrity
from framework.files import load_ckpt
from framework.optim import get_optimizer, get_scheduler
from framework.dataset import TabularDataset
from framework.predictor import Predictor
from models.saint import SAINT
from copy import deepcopy


class SAINTModel(Predictor):

    # TabZilla: add default number of epochs.
    # default_epochs = 100  # from SAINT paper. this is equal to our max-epochs

    def __init__(self, config):
        super().__init__(config)
        self.orig_config = deepcopy(self.config)

    def get_setup(self, prefix: str = None) -> str:
        s = self.orig_config['algorithm'] if prefix is None else prefix 
        for k in self.metadata_keys:
            s = s + '_' + k + f'[{self.orig_config[k]}]'
        return s

    def build_model(self, train_set: TabularDataset):
        if train_set.cat_idx:
            num_idx = list(set(range(train_set.num_features)) - set(train_set.cat_idx))
            # Appending 1 for CLS token, this is later used to generate embeddings.
            cat_dims = torch.cat([torch.tensor([1]), torch.tensor(train_set.cat_dims)]).to(int)
        else:
            num_idx = list(range(train_set.num_features))
            cat_dims = torch.tensor([1])

        model = SAINT(
            categories=tuple(cat_dims),
            num_continuous=len(num_idx),
            dim=self.config["hidden_dim"],
            dim_out=1,
            depth=self.config["depth"],  # 6
            heads=self.config["num_heads"],  # 8
            attn_dropout=self.config["dropout"],  # 0.1
            ff_dropout=self.config["dropout"],  # 0.1
            mlp_hidden_mults=(4, 2),
            cont_embeddings="MLP",
            attentiontype="colrow",
            final_mlp_style="sep",
            y_dim=train_set.num_classes,
        ).to(self.device)
        # For predictors: no projection head by default
        return model, None

    def prepare_training(self, train_set: TabularDataset):
        self.model, self.projection_head = self.build_model(train_set)

    def fit(self, train_set: TabularDataset):
        print('Running fit:')
        print(self.config)
        if 'seed' in self.config:
            fix_seed(self.config.get('seed', -1))
        should_save_model = self.config.get('save_model', False)
        should_load_model = self.config.get('load_model', False)
        should_shuffle = self.config.get('train_shuffle', True)

        # Tabzilla did something to this to avoid memory issue
        self.config["hidden_dim"] = self.config["hidden_dim"] if train_set.num_features < 50 else 8
        self.config["batch_size"] = self.config["batch_size"] if train_set.num_features < 50 else 64

        train_loader = self._build_dataloader(dataset=train_set, should_shuffle=should_shuffle)

        start_epoch = 0
        self.prepare_training(train_set)
        params = []
        for module in self.module_names:
            md = self.get_attr(module)
            if md is not None:
                params.append({'params': md.parameters()})
        self.optimizer = get_optimizer(self.config, params)

        if should_load_model:
            ckpt_folder = self.get_ckpt_folder(train_set)
            ckpt_path = os.path.join(ckpt_folder, f"{train_set.fold}.pth")
            ckpt = load_ckpt(ckpt_path, self.device, self.config)
            if ckpt is not None:
                start_epoch = self.load_state_dict(ckpt)

        if start_epoch >= self.config['epochs']:
            should_save_model = False
        else:
            num_training_steps = self.config['epochs'] * len(train_loader)
            scheduler_dict = get_scheduler(self.config, self.optimizer, num_training_steps)
            self.scheduler = scheduler_dict['scheduler']
            self.scheduler_key = scheduler_dict['key']
            if start_epoch > 0:
                if self.scheduler_key == 'step':
                    self.scheduler.step(start_epoch * len(train_loader))
                elif self.scheduler_key == 'epoch':
                    self.scheduler.step(start_epoch)

            criterion = self.get_loss(train_set)
            for epoch in range(start_epoch + 1, self.config['epochs'] + 1):
                # Prepare for train
                for module in self.module_names:
                    md = self.get_attr(module)
                    if md is not None:
                        md.train()

                epoch_loss = self.train_epoch(criterion, train_loader)
                self.epoch_scheduler(epoch_loss)

                if ('log_interval' in self.config) and (epoch % self.config['log_interval'] == 0):
                    print(f"epoch {epoch}/{self.config['epochs']} - loss: {epoch_loss:.4f}")

        if should_save_model:
            self.save(self.config['epochs'], train_set)

    def train_epoch(self, criterion, train_loader):
        epoch_loss = 0.0
        for x_categ, x_cont, y_gts, cat_mask, con_mask in train_loader:
            if x_categ.shape[0] + x_cont.shape[0] == 1:
                continue
            x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
            cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)
            y_gts = y_gts.to(self.device)
            if len(y_gts.shape) == 2 and y_gts.shape[1] == 1:
                y_gts = y_gts.ravel()
            # We are converting the data to embeddings in the next step
            _, x_categ_enc, x_cont_enc = embed_data_mask(
                x_categ, x_cont, cat_mask, con_mask, self.model
            )
            reps = self.model.transformer(x_categ_enc, x_cont_enc)

            # select only the representations corresponding to CLS token
            # and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:, 0, :]

            y_pred = self.model.mlpfory(y_reps)

            if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
                y_pred = y_pred.ravel()
            loss = criterion(y_pred, y_gts)
            check_loss_integrity(loss)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step_scheduler(loss)
        return epoch_loss / len(train_loader.dataset)

    def predict(self, dataset: TabularDataset):
        self.prepare_eval()
        y_pred = self.get_logits(dataset)
        match dataset.target_type:
            case "binary":
                return (y_pred.ravel() >= 0).long()
            case "classification":
                return torch.argmax(y_pred, dim=1)
            case "regression":
                return y_pred
            case _:
                raise NotImplementedError

    def predict_proba(self, dataset: TabularDataset):
        self.prepare_eval()
        y_pred = self.get_logits(dataset)
        match dataset.target_type:
            case "binary":
                return F.sigmoid(y_pred.ravel())
            case "classification":
                return F.softmax(y_pred, dim=1)
            case "regression":
                return y_pred
            case _:
                raise NotImplementedError

    def get_logits(self, dataset: TabularDataset) -> torch.Tensor:
        loader = self._build_dataloader(dataset=dataset, should_shuffle=False)
        y_pred = []
        with torch.no_grad():
            for x_categ, x_cont, _, cat_mask, con_mask in loader:
                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)
                _, x_categ_enc, x_cont_enc = embed_data_mask(
                    x_categ, x_cont, cat_mask, con_mask, self.model
                )
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = self.model.mlpfory(y_reps)
                y_pred.append(y_outs.cpu())
        y_pred = torch.cat(y_pred).cpu()
        return y_pred

    def _build_dataloader(self, dataset: TabularDataset, should_shuffle: bool = False):
        dataloader = DataLoader(
            DataSetCatCon(X={"data": dataset.data, "mask": torch.ones_like(dataset.data)},
                          Y={"data": dataset.target.reshape(-1, 1)},
                          cat_cols=dataset.cat_idx,
                          task=dataset.target_type),
            batch_size=self.config['batch_size'],
            shuffle=should_shuffle,
            drop_last=self.train_drop_last
        )
        return dataloader


def embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False):
    device = x_cont.device
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1, n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1, n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:, i, :] = model.simple_MLP[i](x_cont[:, i])
    else:
        raise Exception('This case should not work!')

    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)

    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    if vision_dset:
        pos = torch.tile(torch.arange(x_categ.shape[-1]), (x_categ.shape[0], 1))
        pos = torch.from_numpy(pos).to(device)
        pos_enc = model.pos_encodings(pos)
        x_categ_enc += pos_enc

    return x_categ, x_categ_enc, x_cont_enc


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, task='regression', continuous_mean_std=None):

        X_mask = X['mask'].clone()
        X = X['data'].clone()

        # Added this to handle data without categorical features
        if cat_cols is not None:
            cat_cols = list(cat_cols)
            con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        else:
            con_cols = list(np.arange(X.shape[1]))
            cat_cols = []

        self.X1 = X[:, cat_cols].clone().to(torch.int64)  # categorical columns
        self.X2 = X[:, con_cols].clone().to(torch.float32)  # numerical columns
        self.X1_mask = X_mask[:, cat_cols].clone().to(torch.int64)  # categorical columns
        self.X2_mask = X_mask[:, con_cols].clone().to(torch.int64)  # numerical columns
        if task == 'regression':
            self.y = Y['data'].to(torch.float32)
        else:
            self.y = Y['data']  # .astype(np.float32)
        self.cls = torch.zeros_like(self.y).to(int)
        self.cls_mask = torch.ones_like(self.y).to(int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return torch.cat((self.cls[idx], self.X1[idx])), self.X2[idx], self.y[idx], torch.cat(
            (self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]
