import torch
import torch.nn.functional as F
import numpy as np

from framework.dataset import TabularDataset
from framework.encoder import Encoder
from framework.utils import check_loss_integrity, z_score
from algorithms.vicreg import VICRegLoss
from priors.prior import SCARFPrior, GaussianKernelPrior, ConvolutionPrior
from priors.cutmix import CutMixPrior


class VICRegMultiPriorLoss(VICRegLoss):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x is the embedding of the original samples
        y is the embedding of the corrupted samples
        """
        batch_size = x.shape[0]
        sim_loss = F.mse_loss(x, y)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x))
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).mean()
        return sim_loss, std_loss, cov_loss
    

class VICRegMultiPriorEncoder(Encoder):
    """
    VMPE - VICReg Multi Prior Encoder
    config:
      - eta: Learning rate of w
      - tau: epoch (update every epoch) or step-t (update every t steps)
          Example: step-1 means update every step
    """
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.prior_list = []
        self.min_samples_per_prior = 8
        self.w = None
        self.train_step_count = 0
        self.tau_type = 'epoch' if self.config['tau'] == 'epoch' else 'step'
        self.tau = 0 if self.tau_type == 'epoch' else int(self.config['tau'][self.config['tau'].rfind('-')+1:])

    def get_loss(self, train_set: TabularDataset):
        return VICRegMultiPriorLoss(self.config)
    
    def fit(self, train_set: TabularDataset):
        self.set_prior_list(train_set)
        self.w = np.ones((len(self.prior_list),)).astype('float')
        self.w /= sum(self.w)
        super().fit(train_set)

    def train_epoch(self, criterion, train_loader):
        epoch_loss = 0.0
        n_priors = len(self.prior_list)
        sim_losses_sum = [0] * n_priors
        batch_cnt = 0

        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            batch_size = len(x)
            batch_cnt += 1
            
            # 1. Divide the batch into n_priors minibatches
            if n_priors * self.min_samples_per_prior > batch_size: 
                print('Batch size {} too small, skipping...'.format(batch_size))
                continue
            x_list = []
            y_list = []
            for i in range(n_priors):
                left = i * batch_size // n_priors
                right = (i + 1) * batch_size // n_priors
                x_list.append(x[left:right])
                y_list.append(y[left:right])

            # 2. Update the models
            sim_losses, avg_std_loss, avg_cov_loss = self.compute_loss(x_list, y_list, criterion)
            loss = avg_std_loss + avg_cov_loss
            sim_losses_arr = [x.item() for x in sim_losses]
            sim_losses_arr = z_score(sim_losses_arr)
            for i in range(n_priors):
                loss += self.w[i] * sim_losses[i]
                sim_losses_sum[i] += sim_losses_arr[i]
            
            check_loss_integrity(loss)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step_scheduler(loss)
            self.train_step_count += 1

            # 3. Update w for every tau steps
            if self.tau_type == 'step' and self.train_step_count == self.tau:
                self.train_step_count = 0
                sim_losses, avg_std_loss, avg_cov_loss = self.compute_loss(x_list, y_list, criterion)
                sim_losses_arr = [x.item() for x in sim_losses]
                sim_losses_arr = z_score(sim_losses_arr)
                for i in range(n_priors):
                    self.w[i] *= np.exp(sim_losses_arr[i] * self.config['eta'])
                self.w /= sum(self.w)

        # Update w at the end of epoch if needed
        if self.tau_type == 'epoch':
            print_w = self.config.get('print_w', False)
            if print_w:
                print(sim_losses_sum)
                print(batch_cnt)
            for i in range(n_priors):
                self.w[i] *= np.exp(sim_losses_sum[i] / batch_cnt * self.config['eta'])
            self.w /= sum(self.w)
            if print_w:
                print(self.w)

        return epoch_loss / batch_cnt
    
    def compute_loss(self, x_list: list, y_list: list, criterion):
        n_priors = len(x_list)
        sim_losses = []
        avg_std_loss = 0
        avg_cov_loss = 0
        for i in range(n_priors):
            xi, yi = x_list[i], y_list[i]
            x_corrupted = self.prior_list[i].transform(xi, input_x=xi, target_y=yi)
            embeddings = self.get_latent(xi, False)   
            embeddings_corrupted = self.get_latent(x_corrupted, self.svde)
            sim_loss, std_loss, cov_loss = criterion(embeddings, embeddings_corrupted)
            sim_losses.append(sim_loss)
            avg_std_loss += std_loss / n_priors
            avg_cov_loss += cov_loss / n_priors
        return sim_losses, avg_std_loss, avg_cov_loss

    # Implement this
    def set_prior_list(self, train_set: TabularDataset):
        # self.prior_list = [...]
        pass


class SCARFTargetVMPE(VICRegMultiPriorEncoder):
    def set_prior_list(self, train_set: TabularDataset):
        cfg = self.config | {'train_set': train_set}
        self.prior_list = [SCARFPrior(cfg),
                           GaussianKernelPrior(cfg, key='target_y')]
        

class SCARFTargetWithRawVMPE(SCARFTargetVMPE):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.with_raw = True 
        self.ckpt_prefix = 'scarf_target_vmpe'


class CutMixTargetVMPE(VICRegMultiPriorEncoder):
    def set_prior_list(self, train_set: TabularDataset):
        cfg = self.config | {'train_set': train_set}
        self.prior_list = [CutMixPrior(cfg),
                           GaussianKernelPrior(cfg, key='target_y')]
        
class CutMixTargetWithRawVMPE(CutMixTargetVMPE):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.with_raw = True 
        self.ckpt_prefix = 'cutmix_target_vmpe'


# SFT means SCARF * Target
class SCARFSFTVMPE(VICRegMultiPriorEncoder):
    """
    corruption_rate1 is for the single SCARF prior
    corruption_rate2 is for the SFT
    """
    def set_prior_list(self, train_set: TabularDataset):
        cfg = self.config | {'train_set': train_set}
        if 'corruption_rate1' in cfg and 'corruption_rate2' in cfg:
            crate1 = cfg['corruption_rate1']
            crate2 = cfg['corruption_rate2']
        else:
            crate1 = cfg['corruption_rate']
            crate2 = cfg['corruption_rate']

        self.prior_list = [SCARFPrior(cfg | {'corruption_rate': crate1}),
                           ConvolutionPrior(cfg, [GaussianKernelPrior(cfg, key='target_y'), SCARFPrior(cfg | {'corruption_rate': crate2})])]
        


class SCARFSFTWithRawVMPE(SCARFSFTVMPE):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.with_raw = True 
        self.ckpt_prefix = 'scarf_sft_vmpe'



class SFTCMTVMPE(VICRegMultiPriorEncoder):
    """
    corruption_rate1 is for the single SCARF prior
    corruption_rate2 is for the SFT
    """
    def set_prior_list(self, train_set: TabularDataset):
        cfg = self.config | {'train_set': train_set}
        if 'corruption_rate1' in cfg and 'corruption_rate2' in cfg:
            crate1 = cfg['corruption_rate1']
            crate2 = cfg['corruption_rate2']
        else:
            crate1 = cfg['corruption_rate']
            crate2 = cfg['corruption_rate']

        self.prior_list = [ConvolutionPrior(cfg, [GaussianKernelPrior(cfg, key='target_y'), SCARFPrior(cfg | {'corruption_rate': crate1})]),
                           ConvolutionPrior(cfg, [GaussianKernelPrior(cfg, key='target_y'), CutMixPrior(cfg | {'corruption_rate': crate2})])]


class SFTCMTWithRawVMPE(SFTCMTVMPE):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.with_raw = True 
        self.ckpt_prefix = 'sft_cmt_vmpe'



