import logging
import torch
import torch.nn as nn


class MLP(torch.nn.Sequential):
    def __init__(self, config: dict, logger: logging.Logger) -> None:
        self.config = config
        self.residual = self.config['residual']
        self.residual_project = self.config['residual_project']
        self.logger = logger

        self.logger.debug(' ==> MLP config:')
        self.logger.debug(f"{self.config}")
        self.logger.debug('MLP.dropout = {}'.format(self.config['dropout']))
        self.logger.debug('MLP.residual = {}'.format(self.residual))

        layer_dims = [self.config['input_size']] + self.config['hidden_dims']
        self.logger.debug('MLP.layer_dims = {}'.format(layer_dims))
        self.logger.debug('MLP.all_layer_dims = {}'.format(layer_dims + [self.config['output_size']]))
        layers = []
        self.layer_is_linear = []   # 0 - not linear, 1 - linear, 2 - linear, dimension mismatch with projection, 3 - linear but dimension mismatch
        residual_cnt = 0

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=self.config['bias']))
            if self.residual and i > 0:
                if layer_dims[i] == layer_dims[i+1]:
                    self.layer_is_linear.append(1)
                    residual_cnt += 1
                else:
                    if self.residual_project:
                        layers = layers[:-1] + [nn.Linear(layer_dims[i], layer_dims[i+1], bias=self.config['bias'])] + layers[-1]
                        self.layer_is_linear.append(2)
                        residual_cnt += 1
                    else:
                        # If the dimensions don't match and projection not used, then no residual
                        self.layer_is_linear.append(3)
            else:
                self.layer_is_linear.append(0)

            if self.config['batch_norm']:
                layers.append(nn.BatchNorm1d(layer_dims[i+1]))
                self.layer_is_linear.append(0)

            match self.config['activation']:
                case 'relu':
                    layers.append(nn.ReLU(inplace=True))
                case 'tanh':
                    layers.append(nn.Tanh())
                case 'leaky_relu':
                    layers.append(nn.LeakyReLU(self.config.get('leaky_relu_slope', 0.01)))
                case 'sigmoid':
                    layers.append(nn.Sigmoid())
                case 'softmax':
                    layers.append(nn.Softmax(dim=self.config.get('softmax_dim', 1)))
                case 'elu':
                    layers.append(nn.ELU(alpha=self.config.get('elu_alpha', 1.0)))
                case 'selu':
                    layers.append(nn.SELU())
                case 'gelu':
                    layers.append(nn.GELU())
                case 'swish' | 'silu':
                    layers.append(nn.SiLU())
                case _:
                    raise NotImplementedError(f"Activation {self.config['activation']} is not implemented")

                
            self.layer_is_linear.append(0)
            
            if self.config['dropout'] > 0:
                layers.append(nn.Dropout(self.config['dropout']))
                self.layer_is_linear.append(0)

        layers.append(nn.Linear(layer_dims[-1], self.config['output_size'], bias=self.config['bias']))
        self.layer_is_linear.append(1)

        self.logger.debug('layers: {}'.format(layers))
        self.logger.debug(f"{layers}")
        self.logger.debug('layers_is_linear: {}'.format(self.layer_is_linear))
        self.logger.debug(f"{self.layer_is_linear}")
  
        super().__init__(*layers)
        self.layers = layers
        if self.residual:
            self.logger.debug('Number of residual layers = {}'.format(residual_cnt))
        
    def forward(self, input):
        residual_x = None
        j = 0
        for i in range(len(self.layer_is_linear)):
            # 1. Add the residual if there is any
            if self.layer_is_linear[i] > 0:
                if residual_x is not None:
                    # assert not torch.equal(residual_x, input)
                    assert not residual_x is input
                    input = input + residual_x

            # 2. Update the residual
            if self.layer_is_linear[i] == 1:
                residual_x = input
            elif self.layer_is_linear[i] == 2:
                residual_x = self.layers[j](input)
                j += 1
            elif self.layer_is_linear[i] == 3:
                residual_x = None

            # 3. Feed the sum to the new linear layer
            input = self.layers[j](input)
            j += 1

        return input


