"""
The base class of a module
Also provides some useful helper functions
"""

import os
import time 
from typing import List, Optional, Tuple, Union
from logging import Logger
import numpy as np
import torch
from torch import Tensor 
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from framework.dataset import TabularDataset
from framework.utils import to_numpy, get_device, fix_seed, check_loss_integrity
from framework.tools import process_config_value
from framework.files import get_path_from_list, make_parent_dir, load_ckpt, save_ckpt
from framework.optim import get_optimizer, get_scheduler


class Module:
    """
    At init, no models are built
    Models are built when fit is called
    After models built, first load the models if needed
    save_dict is only saved in the last module (the one right before fit)
    """
    def __init__(self, config: dict, default_config: dict, logger: Logger, name: str, ckpt_name: str = None, seed: int = -1) -> None:
        self.config = default_config | config
        self.default_config = default_config
        self.name = name
        self.ckpt_name = self.name if ckpt_name is None else ckpt_name
        self.ckpt_name_use_task_id = False
        self.device = get_device(self.config)
        self.seed = seed
        self.logger = logger
        self.args_keys = []
        self.loadable_items = [] # A list of strs: names of items to be saved and loaded
        self.save_dict = {}  # This dict will be saved together with the models
        self.has_loaded = False
        self.saved_fit_config = None

    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        """
        Implement this! This function will be called before fit or train
        Warning: You should add loadable_items in this function
        If there are no loadable_items, then won't attempt save/load at all
        You can modify the dataset to record some key information
        Especially, you should change dataset.data_dim to reflect the data dim after running this module
        """
        raise NotImplementedError
    
    def get_attr(self, name: str, default=None):
        return getattr(self, name, default)
    
    def _filter_model_params(self) -> dict:
        """
        Helper function: Returns a dict that only contains the args from the config
        """
        model_params = {_k: self.config[_k] for _k in self.args_keys}
        return model_params

    def to_string(self, ckpt: bool = False, include_fit: bool = False) -> str:
        s = self.ckpt_name if ckpt else self.name
        for k in sorted(self.args_keys):
            a = process_config_value(self.config[k])
            s = s + '_' + k + f'[{a}]'
        if include_fit and self.saved_fit_config is not None:
            s = s + f'_fit_{self._fit_config_str(self.saved_fit_config)}'
        return s

    def get_ckpt_folder(self, dataset: TabularDataset, all_modules: List['Module']) -> str:
        """Caution: ckpt_folder does not contain config['ckpt_folder']"""
        # ans = [dataset.name, f'fold_{dataset.fold}'] + [md.to_string(ckpt=True, include_fit=True) for md in all_modules[:-1]] + \
            #   [all_modules[-1].to_string(ckpt=True, include_fit=False)]
        ans = [dataset.name, f'fold_{dataset.fold}'] + [md.to_string(ckpt=True, include_fit=True) for md in all_modules]
        self.logger.debug(f'ans = {ans}')
        folder = get_path_from_list(ans)
        return folder
    
    def get_ckpt_path(self, dataset: TabularDataset, all_modules: List['Module'], this_id: int, fit_config: dict) -> str:
        """Caution: ckpt_path does not contain config['ckpt_folder']"""
        assert all_modules[this_id] == self, f'all_modules = {all_modules}, this_id = {this_id}'
        self.logger.debug(f'all_modules = {all_modules}, this_id = {this_id}')
        ans = self.get_ckpt_folder(dataset, all_modules)
        if self.ckpt_name_use_task_id and dataset.task_id is not None:
            ans = os.path.join(ans, f'{self.ckpt_name}_id{this_id}_task{dataset.task_id}.pth')
        else:
            ans = os.path.join(ans, f'{self.ckpt_name}_id{this_id}.pth')
        if fit_config is not None:
            ans = ans[:-4]
            ans = f'{ans}_fit_{self._fit_config_str(fit_config)}.pth'
        return ans

    def init_and_load(self, dataset: TabularDataset, all_modules: List['Module'], this_id: int, fit_config: dict = None) -> TabularDataset:
        """
        all_modules: A list of all modules, including all linked ones before and after this module
        this_id: The id of this module in all_modules
        """
        dataset = self._init_module(dataset)
        should_load = self.config.get('load', False)
        if not self.config.get('should_load', True):
            should_load = False
        self.logger.debug('should_load = {}'.format(should_load))
        if should_load:
            self.load(dataset, all_modules, this_id, fit_config)
        return dataset

    def state_dict(self) -> dict:
        ans = {}
        for k in self.loadable_items:
            md = self.get_attr(k)
            if isinstance(md, nn.Module):
                ans[k] = md.state_dict()
            else:
                ans[k] = md
        for k in self.save_dict:
            ans['save_dict_' + k] = self.save_dict[k]
        return ans

    def _fit_config_str(self, fit_config: dict) -> str:
        ans = ''
        for item in sorted(fit_config.items()):
            if item[0] != 'module' and item[0] != 'epochs':
                # if everything other than the epochs is the same then that's fine
                a = process_config_value(item[1])
                ans += f'{item[0]}[{a}]'
        return ans

    def save(self, dataset: TabularDataset, all_modules: List['Module'], this_id: int, fit_config: dict = None) -> None:
        should_save = self.config.get('save', False)
        if not self.config.get('should_save', True):
            should_save = False
        if should_save:
            path = self.get_ckpt_path(dataset, all_modules, this_id, fit_config)
            save_ckpt(self.state_dict(), path, self.device, self.config, self.logger)

    def load(self, dataset: TabularDataset, all_modules: List['Module'], this_id: int, fit_config: dict = None) -> bool:
        if len(self.loadable_items) == 0:
            return False
        path = self.get_ckpt_path(dataset, all_modules, this_id, fit_config)
        ans = load_ckpt(path, self.device, self.config, self.logger)
        if ans is not None:
            self.save_dict = {}
            for k in ans:
                if k.startswith('save_dict_'):
                    self.save_dict[k[10:]] = ans[k]
                else:
                    md = self.get_attr(k)
                    if isinstance(md, nn.Module):
                        md.load_state_dict(ans[k])
                    else:
                        setattr(self, k, ans[k])
            self.has_loaded = True
        return (ans is not None)

    def _init_module(self, dataset: TabularDataset) -> TabularDataset:
        fix_seed(self.seed, 1)
        ans = self.init_module(dataset)
        assert ans is not None, 'Bug: You do not return dataset in init_module of {}'.format(self.name)
        return ans


class StandaloneModule(Module):
    """
    This type of module has the fit function. When called, it automatically does the following:
      - Fit itself, and transform
    """
    def transform(self, dataset: TabularDataset) -> TabularDataset:
        """Implement this!"""
        raise NotImplementedError
    
    def fit(self, dataset: TabularDataset) -> None:
        """Implement this!"""
        raise NotImplementedError
    
    def fit_transform(self, dataset: TabularDataset, all_modules: List['Module']) -> TabularDataset:
        assert all_modules[-1] == self, f'all_modules = {all_modules}'
        self._fit(dataset, all_modules)
        ans = self.transform(dataset)
        assert ans is not None, 'Bug: You do not return dataset in transform of {}'.format(self.name)
        return ans
    
    def _fit(self, dataset: TabularDataset, all_modules: List['Module']) -> None:
        this_id = len(all_modules) - 1
        if not self.has_loaded:
            fix_seed(self.seed, 2)
            self.fit(dataset)
            self.has_loaded = True
            self.save(dataset, all_modules, this_id, None)
        else:
            self.logger.debug('Module {} already loaded, skip fitting'.format(self.name))

    def save(self, dataset: TabularDataset, all_modules: List['Module'], this_id: int, fit_config: dict = None) -> None:
        if len(self.loadable_items) == 0:
            return
        super().save(dataset, all_modules, this_id, fit_config)


class LinkedModule(Module):
    """
    This type of module has the forward function, which can convert tensors to tensors and keep the gradients.
    Modules that use Pytorch + batched training should be linked modules.

    About lazy loading:
    When calling init_and_load, a linked module will only init the modules and load the checkpoint, but the 
    weights won't be loaded into the model. This facilitates unload.
    """
    def forward(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Implement this!"""
        raise NotImplementedError

    def transform(self, dataset: TabularDataset, **kwargs) -> TabularDataset:
        """
        Implement this if this module makes predictions
        Warning: Only the transform of the last module (before a transform module) will be called, after self.forward
        """
        self.logger.debug('Module {} does not have a transform method'.format(self.name))
        return dataset

    def prepare_train(self) -> None:
        """This function does something like calling model.train()"""
        pass

    def prepare_eval(self) -> None:
        """This function does something like calling model.eval()"""
        pass

    def get_loss(self, dataset: TabularDataset, X: Tensor, y: Tensor) -> Tensor:
        """
        Implement this if fit will be called after this module!
        Returns the loss tensor
        """
        raise NotImplementedError

    def after_epoch(self, dataset: TabularDataset, train_loader: DataLoader, optimizer: Optimizer, scheduler_dict: dict, all_modules: List[Module], linked_start: int, fit_config: dict) -> None:
        """
        Implement this if this module needs to do something after each training epoch
        For example, clean up, adjust learning rate, etc.
        """
        pass

    def forward_id(self, X: Tensor, y: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Implement this if self.forward_with_id is True. z = sample id. This is only useful at train time. You still need to implement forward"""
        raise NotImplementedError
    
    def _get_default_criterion(self, dataset: TabularDataset):
        """
        Util function: Default loss function
        BCE for binary, cross entropy for classification, and mse for regression
        """
        match dataset.target_type:
            case "binary":
                loss_func = nn.BCEWithLogitsLoss()
                def func(y_pred, y):
                    y = y.float()
                    return loss_func(y_pred, y)
                return func
            case "classification":
                return nn.CrossEntropyLoss()
            case "regression":
                return nn.MSELoss()
            case _:
                raise NotImplementedError("Unknown target type {}".format(dataset.target_type))    

    def __init__(self, config: dict, default_config: dict, logger: Logger, name: str, ckpt_name: str = None, seed: int = -1) -> None:
        super().__init__(config, default_config, logger, name, ckpt_name, seed)
        self.loaded_state_dict = None
        self.fit_config = None   # Only set when this is the last linked module in fit
        self.forward_in_loss = False   # If True, will forward self in get_loss
        self.forward_with_id = False   # If True, will forward self with forward_id, and id will be provided

    def load(self, dataset: TabularDataset, all_modules: List['Module'], this_id: int, fit_config: dict = None) -> bool:
        """Lazy loading. Linked module will always attempt loading."""
        path = self.get_ckpt_path(dataset, all_modules, this_id, fit_config)
        ans = load_ckpt(path, self.device, self.config, self.logger)
        self.save_dict = {}
        if ans is not None:
            for k in ans:
                if k.startswith('save_dict_'):
                    self.save_dict[k[10:]] = ans[k]
        self.loaded_state_dict = ans
        return (ans is not None)

    def load_weights(self) -> None:
        """Complete loading from lazy"""
        self.logger.debug('Loading weights of module {}'.format(self.name))
        if self.loaded_state_dict is not None:
            for k in self.loaded_state_dict:
                if not k.startswith('save_dict_'):
                    md = self.get_attr(k)
                    if isinstance(md, nn.Module):
                        md.load_state_dict(self.loaded_state_dict[k])
                    else:
                        setattr(self, k, self.loaded_state_dict[k])
        self.unload()

    def unload(self) -> None:
        """Only useful if lazy loading"""
        del self.loaded_state_dict
        self.loaded_state_dict = None

    def get_params(self) -> List[dict]:
        """
        Returns a list of dictionaries for the optimizer
        Example: [{'params': self.model.parameters(), 'lr': self.config['lr']}]
        """
        ans = []
        for mdn in self.loadable_items:
            md = self.get_attr(mdn)
            if isinstance(md, nn.Module):
                ans.append({'params': md.parameters(), 'name': '{}_{}'.format(self.name, mdn)})
        return ans
    
    def transform_all(self, dataset: TabularDataset, all_modules: List[Module], linked_start: int, transform_config: dict) -> TabularDataset:
        """
        Perform transform for all linked modules
        """
        linked_modules = all_modules[linked_start:]
        assert linked_modules[-1] == self, f'linked_modules = {linked_modules}'
        batch_size = transform_config['batch_size']
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False) 
        x_new = []
        y_new = []
        with torch.no_grad():
            for md in linked_modules:
                md.prepare_eval()
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                for md in linked_modules:
                    x, y = md.forward(x, y)
                x_new.append(x.to(self.device))
                y_new.append(y.to(self.device))
            dataset.data = torch.cat(x_new)
            dataset.target = torch.cat(y_new)
            dataset = self.transform(dataset, **transform_config)
            assert dataset is not None, 'Bug: You do not return dataset in transform of {}'.format(self.name)
        return dataset
    
    def train(self, dataset: TabularDataset, all_modules: List[Module], linked_start: int, fit_config: dict) -> None:
        """
        Train all linked_modules and self, using the loss function provided by self
        linked_start: id of the first linked module
        fit_config: A dict that contains the fit module and config
        Modules are assumed to have already loaded at this point
        """
        self.logger.debug(f'train: all_modules = {all_modules}, linked_start = {linked_start}')
        self.fit_config = fit_config
        linked_modules = all_modules[linked_start:]
        self.logger.debug(' ==> Running the train function')
        n_samples = len(dataset)
        self.logger.debug('len(dataset) = {}'.format(n_samples))
        assert linked_modules[-1] == self, f'linked_modules = {linked_modules}'
        if not self in linked_modules:
            self.logger.debug('In function train, self is not in linked_modules')
            linked_modules = linked_modules + [self]
        shuffle = fit_config.get('shuffle', True)
        drop_last = fit_config.get('drop_last', True)
        batch_size = fit_config['batch_size']
        epochs = fit_config['epochs']

        if batch_size > n_samples:
            self.logger.warning(f'Batch size is set too big. Got {batch_size}, but the dataset size is {n_samples}. Reducing batch size...')
            batch_size = n_samples

        # If all linked modules have the same epoch, start from there. Else, start from zero.
        start_epoch = None
        for md in linked_modules:
            epoch = md.save_dict.get('epoch', 0)
            if start_epoch is None:
                start_epoch = epoch
            else:
                if start_epoch != epoch:
                    start_epoch = 0
                    break
        self.logger.info('Start training from epoch {}'.format(start_epoch))
        if start_epoch > 0:
            for md in linked_modules:
                md.load_weights()
        else:
            # Train all from scratch
            for md in linked_modules:
                md.save_dict['epoch'] = 0
        params = []
        for md in linked_modules:
            params = params + md.get_params()
        self.logger.debug('optimizer params = {}'.format(params))
        # self.logger.debug('optimizer config = {}'.format(fit_config))
        optimizer = get_optimizer(fit_config, params, self.logger)

        fix_seed(self.seed, 3)
        dataset.get_id = True
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) 
        num_training_steps = epochs * len(train_loader)
        scheduler_dict = get_scheduler(fit_config, optimizer, num_training_steps)
        if start_epoch == epochs:
            self.logger.warning('Checkpoint successfully loaded. Skip training.')
        else:
            if start_epoch > 0:
                self.logger.debug('!== Starting at epoch {}'.format(start_epoch))
                if scheduler_dict['key'] == 'step':
                    scheduler_dict['scheduler'].step(start_epoch * len(train_loader))
                else:
                    scheduler_dict['scheduler'].step(start_epoch)

            save_freq = fit_config.get('save_freq', 0)
            log_freq = fit_config.get('log_freq', 0)
            self.logger.debug(' ==> Training on device {}'.format(self.device))
            self.logger.debug(' len(train_loader) = {}'.format(len(train_loader)))
            for epoch in range(start_epoch + 1, epochs + 1):
                # Note: linked_modules[-1] == self
                for md in linked_modules:
                    md.prepare_train()
                epoch_loss = self.train_epoch(dataset, train_loader, optimizer, scheduler_dict, linked_modules)
                self.epoch_scheduler(scheduler_dict, epoch_loss)
                for md in linked_modules:
                    assert isinstance(md, LinkedModule)
                    md.save_dict['epoch'] = epoch
                    md.after_epoch(dataset, train_loader, optimizer, scheduler_dict, all_modules, linked_start, fit_config)
                if log_freq > 0 and epoch % log_freq == 0:
                    self.logger.warning(f"epoch {epoch}/{epochs} - loss: {epoch_loss:.4f}")
                if save_freq > 0 and epoch % save_freq == 0:
                    self.logger.debug(f"Saving all modules at epoch {epoch}")
                    self.save_all(dataset, all_modules, linked_start, fit_config)
            # Always save the modules at the end of training
            self.save_all(dataset, all_modules, linked_start, fit_config)
        dataset.get_id = False
        self.saved_fit_config = fit_config

    def save_all(self, dataset: TabularDataset, all_modules: List[Module], linked_start: int, fit_config: dict) -> None:
        for i in range(linked_start, len(all_modules)):
            assert isinstance(all_modules[i], LinkedModule)
            all_modules[i].save(dataset, all_modules, i, fit_config)
            
    def train_epoch(self, dataset: TabularDataset, train_loader: DataLoader, optimizer: Optimizer, scheduler_dict: dict, linked_modules: List['LinkedModule']) -> float:
        epoch_loss = 0.0
        for x, y, z in train_loader:
            x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
            m = len(linked_modules)
            for i in range(m - 1):
                x, y = linked_modules[i].forward_id(x, y, z) if linked_modules[i].forward_with_id else linked_modules[i].forward(x, y)
            assert linked_modules[-1] == self
            if not self.forward_in_loss:
                x, y = self.forward_id(x, y, z) if self.forward_with_id else self.forward(x, y)
            if self.forward_with_id:
                loss = self.get_loss(dataset, x, y, z)
            else:
                loss = self.get_loss(dataset, x, y)
            check_loss_integrity(loss, self.logger)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.step_scheduler(scheduler_dict, loss)
        return epoch_loss / len(train_loader)

    def step_scheduler(self, scheduler_dict: dict, loss) -> None:
        if scheduler_dict['key'] == 'step':
            scheduler_dict['scheduler'].step()
        elif scheduler_dict['key'] == 'loss_step':
            scheduler_dict['scheduler'].step(loss)

    def epoch_scheduler(self, scheduler_dict: dict, loss) -> None:
        if scheduler_dict['key'] == 'epoch':
            scheduler_dict['scheduler'].step()
        elif scheduler_dict['key'] == 'loss_epoch':
            scheduler_dict['scheduler'].step(loss)


def summarize_recipe(all_modules: List[Union[dict, Module]]) -> str:
    """The string starts with __ (2 _)"""
    ans = ''
    for md in all_modules:
        assert isinstance(md, Module)
        ans += '__' + md.to_string()
        if isinstance(md, LinkedModule) and md.fit_config is not None:
            ans += '__fit'
            for k in sorted(md.fit_config):
                if k != 'module':
                    a = process_config_value(md.fit_config[k])
                    ans += f'_{k}[{a}]'
    return ans

