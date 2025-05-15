from typing import Tuple
import torch
from torch import Tensor
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial

from framework.dataset import TabularDataset
from knowledge.base import PPKnowledge


class SCARFKnowledge(PPKnowledge):
    def fit(self, dataset: TabularDataset) -> None:
        features_low = dataset.features_low
        features_high = dataset.features_high
        self.distribution = self.config.get('distribution', 'uniform')
        self.corruption_rate = self.config.get('corruption_rate', 0.5)
        self.uniform_eps = self.config.get('uniform_eps', 1e-6)

        if self.distribution == 'uniform':
            self.marginals = Uniform(torch.Tensor(features_low) - self.uniform_eps, 
                                     torch.Tensor(features_high) + self.uniform_eps)
        elif self.distribution == 'gaussian':
            self.marginals = Normal(torch.Tensor((features_high + features_low) / 2), 
                                    torch.Tensor((features_high - features_low) / 4))
        elif self.distribution == 'bimodal':
            self.marginals_low = Normal(torch.Tensor(features_low), torch.Tensor((features_high - features_low) / 8))
            self.marginals_high = Normal(torch.Tensor(features_high), torch.Tensor((features_high - features_low) / 8))
        else:
            raise NotImplementedError(f"Unsupported prior distribution: {self.distribution}")

    def transform(self, x: Tensor, y: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        batch_size, _ = x.size()
        corruption_mask = torch.rand_like(x, device=x.device) < self.corruption_rate # cr of the entries are 1
        if self.distribution in ['uniform', 'gaussian']:
            x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        elif self.distribution == 'bimodal':
            x_random_low = self.marginals_low.sample(torch.Size((batch_size,))).to(x.device)
            x_random_high = self.marginals_high.sample(torch.Size((batch_size,))).to(x.device)
            x_random = torch.where(torch.rand(batch_size, device=x.device) > 0.5, x_random_low, x_random_high)
        # self.logger.debug('corruption_mask = {}, x_random = {}, x = {}'.format(corruption_mask.shape, x_random.shape, x.shape))
        x_corrupted = torch.where(corruption_mask, x_random, x)  # Choose x_random at rate cr
        return x_corrupted, y