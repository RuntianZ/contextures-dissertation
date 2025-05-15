
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from framework.dataset import TabularDataset
from framework.utils import to_numpy
from knowledge.base import DualKKnowledge

class TeacherModelKnowledge(DualKKnowledge):
    def teacher_model(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Return Phi(x) by default
        """
        raise NotImplementedError
    
    def fit(self, dataset: TabularDataset) -> None:
        teacher_batch_size = self.config.get('teacher_batch_size', 128)
        with torch.no_grad():
            if teacher_batch_size == 0:
                x, y = dataset.data.to(self.device), dataset.target.to(self.device)
                z = self.teacher_model(x, y)
            else:
                ds = TensorDataset(dataset.data, dataset.target)
                loader = DataLoader(ds, batch_size=teacher_batch_size, shuffle=False, drop_last=False)
                z = None
                for _, (x, y) in enumerate(loader):
                    x, y = x.to(self.device), y.to(self.device)
                    z0 = self.teacher_model(x, y)
                    z = z0 if z is None else torch.cat([z, z0])

        self.mu = z.mean(axis=0).to(self.device)  # The mean of the teacher model
        z = z - self.mu 
        U, S, Vh = np.linalg.svd(to_numpy(z))
        threshold = self.config['teacher_threshold']

        if threshold == 0:
            D = np.eye(Vh.shape[0]) / S[0]
        else:
            # STK
            # S0: original singular values; S: target singular values
            # Multiplying z by S / S0 makes the singular values -> S
            S0 = S.copy()   
            S[S < threshold] = 0
            S[S >= threshold] = 1
            ids = (S0 == 0)
            S0[ids] = 1
            R = S / S0
            R[ids] = 0
            if len(S) == Vh.shape[0]:
                D = np.diag(R)
            else:
                D = np.zeros((Vh.shape[0], Vh.shape[0]))
                D[:len(S), :len(S)] = np.diag(R)
        self.proj = Vh.T @ D @ Vh
        self.proj = torch.tensor(self.proj).float().to(self.device)

    def kernel(self, x1: Tensor, x2: Tensor, y1: Tensor, y2: Tensor, **kwargs) -> Tensor:
        """
        k = n * (x x.T) + 1, where x is centered and whitened
        so that 1/n \sum_i k(x, x_i) f(x_i) = lambda_i f(x)
        Empirical distribution: Uniform over the batch
        """
        n = x1.shape[0]
        if x2 is not None:
            assert x2.shape[0] == n
        with torch.no_grad():
            z1 = self.teacher_model(x1, y1)
            z1 = z1 - self.mu 
            z1 = z1 @ self.proj
            if x2 is None and y2 is None:
                z2 = z1
            else:
                if x2 is None:
                    x2 = x1
                if y2 is None:
                    y2 = y1
                z2 = self.teacher_model(x2, y2)
                z2 = z2 - self.mu 
                z2 = z2 @ self.proj
            k = n * z1 @ z2.T + 1
        return k


class RawFeatureKnowledge(TeacherModelKnowledge):
    """
    Test ok
    Regard the raw features X as a teacher model
    """
    def teacher_model(self, x: Tensor, y: Tensor) -> Tensor:
        return x
    

class YLinearKnowledge(TeacherModelKnowledge):
    """
    Train only
    Linear kernel of the target y
      - For classification, the cls kernel: k_value(y) * I[y = y']; k_value(y) = n_all / n_y
      - For regression, the centered and normalized version of <y, y'>

    About k_value:
      For v = [1, 1, ..., 1], we want k @ v / n = v, so that v has eigenvalue 1
      w.r.t. the empirical distribution (uniform over the n samples)
    """
    def fit(self, dataset: TabularDataset) -> None:
        self.target_type = dataset.target_type
        if self.target_type == 'regression':
            super().fit(dataset)
        else:
            n = len(dataset.target)
            self.unique_y = torch.unique(dataset.target).to(self.device)
            self.k_value = torch.zeros(len(self.unique_y)).float().to(self.device)
            y = dataset.target.to(self.device)
            for i in range(len(self.unique_y)):
                ny = (y == self.unique_y[i]).sum()
                self.k_value[i] = n / ny 

    def teacher_model(self, x: Tensor, y: Tensor) -> Tensor:
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        return y
        
    def kernel(self, x1: Tensor, x2: Tensor, y1: Tensor, y2: Tensor, **kwargs) -> Tensor:
        if self.target_type == 'regression':
            return super().kernel(x1, x2, y1, y2, **kwargs)
        else:
            n = x1.shape[0]
            k = torch.zeros((n, n)).float().to(self.device)
            for i in range(len(self.unique_y)):
                id1 = (y1 == self.unique_y[i]).view(-1, 1)
                if y2 is None:
                    id2 = id1
                else:
                    id2 = (y2 == self.unique_y[i]).view(-1, 1)
                # self.logger.debug(f'id1 = {id1}, id2 = {id2}')
                ids = id1 & id2.T  # ids[i, j] == True if y1[i] == y2[j] == y
                k[ids] = self.k_value[i]
            return k 


