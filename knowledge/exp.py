import os
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple
from copy import deepcopy

from framework.base import LinkedModule, Module, summarize_recipe
from framework.dataset import TabularDataset
from framework.files import make_parent_dir
from framework.utils import create_module
from framework.tools import off_diagonal, process_config_value
from knowledge.base import PPKnowledge, DualKKnowledge, Knowledge
from knowledge.utils import clip_singular_values
from knowledge.scarf import SCARFKnowledge
from knowledge.cutmix import CutmixKnowledge
from knowledge.teacher import YLinearKnowledge
from knowledge.xgboost import XGBoostKnowledge


class ExperimentalConvolutionLearner(LinkedModule):
    """
    In VICReg, use z_x_centered instead of B
    This version is not good, because VICReg is not really doing anything
    If only enforces Phi(z) to be non-zero, but B could still be zero
    """
    def get_knowledge_list(self, dataset: TabularDataset) -> List[Knowledge]:
        """Return [P1, P2, ..., Pr]"""
        raise NotImplementedError
    
    def init_module(self, dataset: TabularDataset, phi_modules: list = None, phi_head_modules: list = None, knowledge_list: list = None) -> TabularDataset:
        """
        We do not set loadable_items here because we will modify the init_and_load and save methods
        """
        self.forward_in_loss = True  # Will run forward in get_loss
        self.load_mode = None
        self.logger.debug('ConvolutionLearner init_module')
        self.logger.debug('Config: {}'.format(self.config))
        self.knowledge_list = self.get_knowledge_list(dataset) if knowledge_list is None else knowledge_list
        for kn in self.knowledge_list:
            kn.fit(dataset)
        need_psi = isinstance(self.knowledge_list[-1], PPKnowledge) and self.config['svme']
        if need_psi:
            dataset_psi = deepcopy(dataset)
            psi_batch_size = self.config.get('psi_batch_size', 128)
            ds = TensorDataset(dataset_psi.data, dataset_psi.target)
            loader = DataLoader(ds, batch_size=psi_batch_size, shuffle=False, drop_last=False)
            xa = None
            ya = None
            with torch.no_grad():
                for _, (x, y) in enumerate(loader):
                    # x, y = x.to(self.device), y.to(self.device)
                    x1, y1 = self.knowledge_list[-1].transform(x, y)
                    if xa is None:
                        xa, ya = x1, y1
                    else:
                        xa, ya = torch.cat([xa, x1]), torch.cat([ya, y1])
            dataset_psi.data = xa
            dataset_psi.target = ya

        # Initialize phi and psi
        seed = self.seed
        if phi_modules is None:   
            self.phi_modules = []         
            for md in self.config['encoder']:
                if phi_modules is None:
                    if seed >= 0:
                        seed += 13
                    md_obj = create_module(md, self.default_config, self.logger, seed)
                    assert isinstance(md_obj, LinkedModule), 'Encoder and head only support linked modules'
                    dataset = md_obj._init_module(dataset)
                    self.phi_modules.append(md_obj)
        else:
            self.phi_modules = phi_modules
        
        self.psi_modules = [] if need_psi else None
        if need_psi:
            psi_mds = self.config['psi_encoder'] if isinstance(self.config['psi_encoder'], dict) else self.config['encoder']
            for md in psi_mds:
                if seed >= 0:
                    seed += 13
                md_obj_2 = create_module(md, self.default_config, self.logger, seed)
                dataset_psi = md_obj_2._init_module(dataset_psi)
                self.psi_modules.append(md_obj_2)

        # Initialize phi_head and psi_head
        # TODO
        self.phi_head_modules = None
        self.psi_head_modules = None

        return dataset

    def to_string(self, ckpt: bool = False) -> str:
        s = self.ckpt_name if ckpt else self.name
        for k in sorted(self.args_keys):
            if not k in ['encoder', 'head']:
                a = a = process_config_value(self.config[k])
                s = s + '_' + k + f'[{a}]'
        assert self.phi_modules is not None, 'phi_modules not built yet'
        a = summarize_recipe(self.phi_modules)
        a = a[2:]
        s = s + '_encoder' + f'[{a}]'
        if self.phi_head_modules is not None:
            a = summarize_recipe(self.phi_head_modules)
            a = a[2:]
            s = s + '_head' + f'[{a}]'
        else:
            s = s + '_head[None]'
        if self.psi_modules is not None:
            if isinstance(self.config['psi_encoder'], dict):
                a = summarize_recipe(self.psi_modules)
                a = a[2:]
                s = s + '_psi_encoder' + f'[{a}]'
            else:
                s = s + '_psi_encoder[Same]'
        if self.load_mode is not None:
            s = s + f'_loading[{self.load_mode}]'
        return s

    def load(self, dataset: TabularDataset, all_modules: List['Module'], this_id: int, fit_config: dict = None) -> bool:
        assert all_modules[this_id] == self, f'all_modules = {all_modules}'
        self.save_dict['epoch'] = -1
        assert self.phi_modules is not None, 'phi_modules not built yet'
        self.load_mode = 'phi'
        self.group_load(dataset, self.phi_modules, all_modules, fit_config)
        self.logger.debug('After load, phi_modules = {}'.format(self.phi_modules))
        assert self.save_dict['epoch'] != -1
        if self.phi_head_modules is not None and self.save_dict['epoch'] != 0:
            self.load_mode = 'phi_head'
            self.group_load(dataset, self.phi_head_modules, all_modules, fit_config)
        self.load_psi(dataset, all_modules, this_id, fit_config)
        if self.save_dict['epoch'] == 0:
            self.unload_phi()
            self.unload_psi()
        self.load_mode = None
        return (self.save_dict['epoch'] != 0)

    def load_psi(self, dataset: TabularDataset, all_modules: List['Module'], this_id: int, fit_config: dict) -> None:
        if self.psi_modules is not None and self.save_dict['epoch'] != 0:
            self.load_mode = 'psi'
            self.group_load(dataset, self.psi_modules, all_modules, fit_config)
        if self.psi_head_modules is not None and self.save_dict['epoch'] != 0:
            self.load_mode = 'psi_head'
            self.group_load(dataset, self.psi_head_modules, all_modules, fit_config)

    def group_load(self, dataset: TabularDataset, submodules: List[Module], all_modules: List['Module'], fit_config: dict) -> None:
        # The load_mode is set so we know which module is currently loading
        assert self.load_mode is not None
        mds = deepcopy(all_modules)
        for md in submodules:
            mds.append(md)
            success = md.load(dataset, mds, len(mds) - 1, fit_config)
            if not success or (self.save_dict['epoch'] != -1 and md.save_dict['epoch'] != self.save_dict['epoch']):
                self.save_dict['epoch'] = 0
                return
            self.save_dict['epoch'] = md.save_dict['epoch']

    def load_weights(self) -> None:
        # This will be called right before training, in base.LinkedModule
        self.logger.debug('convolution learner load_weights(epoch={})'.format(self.save_dict['epoch']))
        if self.save_dict['epoch'] > 0:
            ans = self.get_all_modules()
            for md in ans:
                md.load_weights()

    def unload_phi(self) -> None:
        for md in self.phi_modules:
            md.save_dict['epoch'] = 0
        if self.phi_head_modules is not None:
            for md in self.phi_head_modules:
                md.save_dict['epoch'] = 0

    def unload_psi(self) -> None:
        if self.psi_modules is not None:
            for md in self.psi_modules:
                md.save_dict['epoch'] = 0
        if self.psi_head_modules is not None:
            for md in self.psi_head_modules:
                md.save_dict['epoch'] = 0

    def save(self, dataset: TabularDataset, all_modules: List['Module'], this_id: int, fit_config: dict) -> None:
        should_save = self.config.get('save', False)
        assert all_modules[this_id] == self, f'all_modules = {all_modules}'
        if should_save:
            assert self.phi_modules is not None, 'phi_modules not built yet'
            self.load_mode = 'phi'
            self.group_save(dataset, self.phi_modules, all_modules, fit_config)
            if self.phi_head_modules is not None:
                self.load_mode = 'phi_head'
                self.group_save(dataset, self.phi_head_modules, all_modules, fit_config)
            self.save_psi(dataset, all_modules, fit_config)
            self.load_mode = None

    def save_psi(self, dataset: TabularDataset, all_modules: List[Module], fit_config: dict) -> None:
        if self.psi_modules is not None:
            self.load_mode = 'psi'   # For setting the right ckpt name
            self.group_save(dataset, self.psi_modules, all_modules, fit_config)
        if self.psi_head_modules is not None:
            self.load_mode = 'psi_head'
            self.group_save(dataset, self.psi_head_modules, all_modules, fit_config)
            
    def group_save(self, dataset: TabularDataset, submodules: List[Module], all_modules: List['Module'], fit_config: dict) -> None:
        assert self.load_mode is not None
        mds = deepcopy(all_modules)
        for md in submodules:
            mds.append(md)
            md.save_dict['epoch'] = self.save_dict['epoch']
            assert md.save_dict['epoch'] > 0
            md.save(dataset, mds, len(mds) - 1, fit_config)

    def get_all_modules(self) -> List[LinkedModule]:
        ans = self.phi_modules
        if self.phi_head_modules is not None:
            ans = ans + self.phi_head_modules
        ans = ans + self.get_psi_modules()
        return ans

    def get_psi_modules(self) -> List[LinkedModule]:
        ans = []
        if self.psi_modules is not None:
            ans = ans + self.psi_modules
        if self.psi_head_modules is not None:
            ans = ans + self.psi_head_modules
        return ans
    
    def get_params(self) -> List[dict]:
        mds = self.get_all_modules()
        ans = [] 
        for md in mds:
            ans = ans + md.get_params()
        return ans

    def prepare_train(self) -> None:
        ans = self.get_all_modules()
        for md in ans:
            md.prepare_train()

    def prepare_eval(self) -> None:
        ans = self.get_all_modules()
        for md in ans:
            md.prepare_eval()

    def forward(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        for md in self.phi_modules:
            X, y = md.forward(X, y)
        return X, y
    
    def forward_psi(self, Xa: Tensor, ya: Tensor) -> Tuple[Tensor, Tensor]:
        if self.psi_modules is None:
            self.logger.debug('Not using SVME, using phi instead')
            mds = self.phi_modules 
        else:
            mds = self.psi_modules
        for md in mds:
            Xa, ya = md.forward(Xa, ya)
        return Xa, ya
    
    def get_loss(self, dataset: TabularDataset, X: Tensor, y: Tensor, regularization: bool = True) -> Tensor:
        # Algorithm 1 in the paper
        m = X.shape[0]
        X0 = X.clone()   # Original inputs
        y0 = y.clone()
        r = len(self.knowledge_list)
        assert r > 0, "Empty knowledge list"
        # with torch.no_grad():   # Is torch.no_grad safe here?
        z_X, z_y = self.forward(X, y)
        mu_X = z_X.mean(axis=0)
        z_X_centered = z_X - mu_X
        B = None
        ki = -1  # The id of the stored k
        for j in range(r - 1):
            if isinstance(self.knowledge_list[j], DualKKnowledge):
                if ki == -1:
                    assert B is None
                    z_X, z_y = self.forward(X, y)
                    B = z_X - mu_X  # Center Phi
                    B = B.T
                    assert B.shape[1] == m
                else:
                    assert B is not None
                    G = self.knowledge_list[ki].kernel(X0, X, y0, y)
                    B = B @ G / m 
                ki = j
                X = X0.clone()  # Reset to original inputs
                y = y0.clone()
            else:
                X, y = self.knowledge_list[j].transform(X, y)

        if ki == -1:
            assert B is None
            z_X, z_y = self.forward(X, y)
            B = z_X - mu_X
            B = B.T
            assert B.shape[1] == m 
        else:
            assert B is not None
            G = self.knowledge_list[ki].kernel(X0, X, y0, y)
            B = B @ G / m
        
        if isinstance(self.knowledge_list[-1], DualKKnowledge):
            Gr = self.knowledge_list[-1].gram_matrix(X0, y0)
            # Gr = Gr / m 
            # Gr = clip_singular_values(Gr)  # Make sure that no singular value > 1, for numerical stability
            C = B @ Gr / m

            # loss = (torch.norm(z_X_centered, p="fro") ** 2 - torch.linalg.vecdot(B, C).sum()) / m  # Original. Numerically unstable
            loss = (torch.norm(B, p="fro") ** 2 - torch.linalg.vecdot(B, C).sum()) / m

            # U, S, Vh = torch.linalg.svd(z_X_centered)
            # Q = U.T @ B @ C.T @ U 
            # Q = torch.diagonal(Q)
            # S = F.relu(S - Q)
            # loss = (S ** 2).sum() / m 

        else:
            Xa, ya = self.knowledge_list[-1].transform(X0, y0)
            z_Xa, z_ya = self.forward_psi(Xa, ya)
            z_Xa -= z_Xa.mean(axis=0)
            # loss = (torch.norm(z_X_centered, p="fro") ** 2 + torch.norm(z_Xa, p="fro") ** 2 - 2 * torch.linalg.vecdot(B.T, z_Xa).sum()) / m   # Original
            loss = (torch.norm(B, p="fro") ** 2 + torch.norm(z_Xa, p="fro") ** 2 - 2 * torch.linalg.vecdot(B.T, z_Xa).sum()) / m

        zx = z_X_centered  # Original. Numerically unstable
        # zx = B.T
        d = zx.shape[1]
        loss = loss / d          # Divided by d, since all other losses are divded by d
        # loss2 = F.mse_loss(zx, z_Xa)   # Should have loss = loss2 for 1 knowledge
        # self.logger.warning('loss - loss2 = {}'.format(torch.norm(loss - loss2)))

        if regularization:
            # VICReg
            std_x = torch.sqrt(zx.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_x))    # Divided by d here
            cov_x = (zx.T @ zx) / (m - 1)
            cov_loss = off_diagonal(cov_x).pow_(2).mean()   # Divided by d(d-1) here
            std_coeff = self.config['std_coeff']
            cov_coeff = self.config['cov_coeff']
            loss = (loss + std_coeff * std_loss + cov_coeff * cov_loss) / (1 + std_coeff + cov_coeff)
        return loss
    

class ExperimentalYLinearSCARFLearner(ExperimentalConvolutionLearner):
    """y_linear * SCARF"""
    def get_knowledge_list(self, dataset: TabularDataset) -> List[Knowledge]:
        k1 = YLinearKnowledge(self.logger, self.config, self.device)
        k2 = SCARFKnowledge(self.logger, self.config, self.device)
        return [k1, k2]

