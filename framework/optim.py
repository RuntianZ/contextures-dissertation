import logging
import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, MultiStepLR
from torch.optim.optimizer import Optimizer
from functools import partial
from framework.utils import str_to_list


class Lars(Optimizer):
    """ https://huggingface.co/spaces/Roll20/pet_score/blob/main/lib/timm/optim/lars.py
    PyTorch LARS / LARC Optimizer
    An implementation of LARS (SGD) + LARC in PyTorch
    Based on:
    * PyTorch SGD: https://github.com/pytorch/pytorch/blob/1.7/torch/optim/sgd.py#L100
    * NVIDIA APEX LARC: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
    Additional cleanup and modifications to properly support PyTorch XLA.
    Copyright 2021 Ross Wightman

    LARS for PyTorch
    Paper: `Large batch training of Convolutional Networks` - https://arxiv.org/pdf/1708.03888.pdf
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1.0).
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        trust_coeff (float): trust coefficient for computing adaptive lr / trust_ratio (default: 0.001)
        eps (float): eps for division denominator (default: 1e-8)
        trust_clip (bool): enable LARC trust ratio clipping (default: False)
        always_adapt (bool): always apply LARS LR adapt, otherwise only when group weight_decay != 0 (default: False)
    """

    def __init__(
        self,
        params,
        lr=1.0,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        trust_coeff=0.001,
        eps=1e-8,
        trust_clip=False,
        always_adapt=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coeff=trust_coeff,
            eps=eps,
            trust_clip=trust_clip,
            always_adapt=always_adapt,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]['params'][0].device
        one_tensor = torch.tensor(1.0, device=device)  # because torch.where doesn't handle scalars correctly

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            trust_coeff = group['trust_coeff']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # apply LARS LR adaptation, LARC clipping, weight decay
                # ref: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
                if weight_decay != 0 or group['always_adapt']:
                    w_norm = p.norm(2.0)
                    g_norm = grad.norm(2.0)
                    trust_ratio = trust_coeff * w_norm / (g_norm + w_norm * weight_decay + eps)
                    # FIXME nested where required since logical and/or not working in PT XLA
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, trust_ratio, one_tensor),
                        one_tensor,
                    )
                    if group['trust_clip']:
                        trust_ratio = torch.minimum(trust_ratio / group['lr'], one_tensor)
                    grad.add(p, alpha=weight_decay)
                    grad.mul_(trust_ratio)

                # apply SGD update https://github.com/pytorch/pytorch/blob/1.7/torch/optim/sgd.py#L100
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1. - dampening)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf

                p.add_(grad, alpha=-group['lr'])

        return loss



def get_optimizer(config: dict, parameters: list, logger: logging.Logger):
    optimizer = config['optimizer']
    logger.debug('get_optimizer: Number of learnable parameters = {}'.format(len(parameters)))
    learning_rate = float(config['lr'])
    weight_decay = float(config['wd'])

    if optimizer == 'adam':
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'adamw':
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer.startswith('sgd'):   # sgd-0.1
        momentum = float(optimizer[optimizer.rfind('-')+1:])
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == 'lars':
        return Lars(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} is not supported")
        


###########################################
def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class DummyScheduler(LRScheduler):
    def step(self, epoch=None):
        pass


def get_scheduler(config: dict, optimizer, num_training_steps: int) -> dict:
    """
    scheduler key:
      - Step: Run at every step
      - Epoch: Run at the end of every epoch
      - Loss: Use training loss
    Supported:
      - warmup-0.1
      - reduce_on_plateau
    """
    scheduler = config.get('scheduler', None)
    if scheduler == 'None' or scheduler == 'none':
        scheduler = None
    config['scheduler'] = scheduler

    if scheduler is None:
        sch = DummyScheduler(optimizer)
        key = 'none'
    elif scheduler.startswith('warmup'):
        fraction = float(scheduler[scheduler.rfind('-')+1:])
        num_warmup_steps = int(fraction * num_training_steps)
        sch = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        key = 'step'
    elif scheduler == 'reduce_on_plateau':
        factor = config.get('scheduler_factor', 0.1)
        patience = config.get('scheduler_patience', 10)
        threshold = config.get('scheduler_threshold', 0.0001)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, threshold=threshold)
        key = 'loss_epoch'
    elif scheduler.startswith('multistep'):
        s = scheduler.split('_')
        key = s[1]
        gamma = float(s[2])
        steps = '[' + s[3] + ']'
        steps = str_to_list(steps)
        sch = MultiStepLR(optimizer, steps, gamma=gamma)
    else:
        raise NotImplementedError(f"Scheduler {scheduler} is not supported")

    return {'scheduler': sch, 'key': key}