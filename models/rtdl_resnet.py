import torch
import math
import torch.nn as nn
from typing import Callable, Union, List, Type
import torch.nn.functional as F
import enum


class rtdl_ResNet(nn.Module):
    """
    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    # class Block(nn.Module):
    #     def __init__(
    #             self,
    #             *,
    #             d_main: int,
    #             d_hidden: int,
    #             config: dict
    #     ) -> None:
    #         super().__init__()
    #         layers = []
    #         if config["batch_norm"]:
    #             layers.append(nn.BatchNorm1d(d_main))
    #         layers.append(nn.Linear(d_main, d_hidden, config["bias"]))
    #         layers.append(_config_activation(config))
    #         if config["dropout_first"] > 0.:
    #             layers.append(nn.Dropout(config["dropout_first"]))
    #         layers.append(nn.Linear(d_hidden, d_main, config["bias"]))
    #         if config["dropout_second"] > 0.:
    #             layers.append(nn.Dropout(config["dropout_second"]))
    #
    #         self.layers = nn.Sequential(*layers)
    #
    #         self.skip_connection = config["residual"]
    #
    #     def forward(self, x: torch.Tensor) -> torch.Tensor:
    #         x_input = x
    #         x = self.layers(x)
    #         if self.skip_connection:
    #             x = x_input + x
    #         return x
    #
    # class Head(nn.Module):
    #     def __init__(
    #             self,
    #             *,
    #             d_in: int,
    #             d_out: int,
    #             config: dict
    #     ) -> None:
    #         super().__init__()
    #         layers = []
    #         if config["batch_norm"]:
    #             layers.append(nn.BatchNorm1d(d_in))
    #         layers.append(_config_activation(config))
    #         layers.append(nn.Linear(d_in, d_out, config["bias"]))
    #         self.layers = nn.Sequential(*layers)
    #
    #     def forward(self, x: torch.Tensor) -> torch.Tensor:
    #         x = self.layers(x)
    #         return x

    def __init__(
            self,
            config: dict,
    ) -> None:
        """
        Requires:
            - input_size
            - main_dim
            - hidden_dim
            - output_size
            - n_blocks

        Input layer: linear
        ResBlock: batchnorm -> linear -> activation -> dropout -> linear -> dropout
        Output layer: batchnorm -> activation -> linear

        Shape of ResNet block goes like: main_dim -> hidden_dim -> main_dim
        """
        super().__init__()
        default_config = {
            'activation': 'relu',
            'dropout_first': 0.0,
            'dropout_second': 0.0,
            'bias': True,
            'batch_norm': True,
            'residual': True,
        }
        self.config = default_config | config
        self.resnet = ResNet.make_baseline(d_in=self.config["input_size"],
                                           n_blocks=self.config["n_blocks"],
                                           d_main=self.config["main_dim"],
                                           d_hidden=self.config["hidden_dim"],
                                           dropout_first=self.config["dropout_first"],
                                           dropout_second=self.config["dropout_second"],
                                           d_out=self.config["output_size"])

        # self.first_layer = nn.Linear(self.config["input_size"], self.config["main_dim"])
        # if self.config["main_dim"] is None:
        #     self.config["main_dim"] = self.config["input_size"]
        # self.blocks = nn.Sequential(
        #     *[
        #         rtdl_ResNet.Block(
        #             d_main=self.config["main_dim"],
        #             d_hidden=self.config["hidden_dim"],
        #             config=self.config
        #         )
        #         for _ in range(self.config["n_blocks"])
        #     ]
        # )
        # self.head = rtdl_ResNet.Head(
        #     d_in=self.config["main_dim"],
        #     d_out=self.config["output_size"],
        #     config=self.config
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.first_layer(x)
        # x = self.blocks(x)
        # x = self.head(x)
        x = self.resnet(x)
        return x


def make_nn_module(module_type: Union[str, Callable[..., nn.Module]], *args) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == 'ReGLU':
            return ReGLU()
        elif module_type == 'GEGLU':
            return GEGLU()
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(
                    f'Failed to construct the module {module_type} with the arguments {args}'
                ) from err
            return cls(*args)
    else:
        return module_type(*args)


# def config_activation(config):
#     activation = None
#     match config['activation']:
#         case 'relu':
#             activation = nn.ReLU()
#         case 'tanh':
#             activation = nn.Tanh()
#         case 'leaky_relu':
#             activation = nn.LeakyReLU(config.get('leaky_relu_slope', 0.01))
#         case 'sigmoid':
#             activation = nn.Sigmoid()
#         case 'softmax':
#             activation = nn.Softmax(dim=config.get('softmax_dim', 1))
#         case 'elu':
#             activation = nn.ELU(alpha=config.get('elu_alpha', 1.0))
#         case 'selu':
#             activation = nn.SELU()
#         case 'gelu':
#             activation = nn.GELU()
#         case 'swish':
#             activation = nn.SiLU()
#         case _:
#             raise NotImplementedError('Activation {} is not implemented'.format(config['activation']))
#     return activation


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)


class ResNet(nn.Module):
    """The ResNet model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)

                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)

          Head: (in) -> Norm -> Activation -> Linear -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                d_in=x.shape[1],
                n_blocks=2,
                d_main=3,
                d_hidden=4,
                dropout_first=0.25,
                dropout_second=0.0,
                d_out=1
            )
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
                self,
                *,
                d_main: int,
                d_hidden: int,
                bias_first: bool,
                bias_second: bool,
                dropout_first: float,
                dropout_second: float,
                normalization: Union[str, Callable[..., nn.Module]],
                activation: Union[str, Callable[..., nn.Module]],
                skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        """The final module of `ResNet`."""

        def __init__(
                self,
                *,
                d_in: int,
                d_out: int,
                bias: bool,
                normalization: Union[str, Callable[..., nn.Module]],
                activation: Union[str, Callable[..., nn.Module]],
        ) -> None:
            super().__init__()
            self.normalization = make_nn_module(normalization, d_in)
            self.activation = make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
            self,
            *,
            d_in: int,
            n_blocks: int,
            d_main: int,
            d_hidden: int,
            dropout_first: float,
            dropout_second: float,
            normalization: Union[str, Callable[..., nn.Module]],
            activation: Union[str, Callable[..., nn.Module]],
            d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()

        self.first_layer = nn.Linear(d_in, d_main)
        if d_main is None:
            d_main = d_in
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = ResNet.Head(
            d_in=d_main,
            d_out=d_out,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    @classmethod
    def make_baseline(
            cls: Type['ResNet'],
            *,
            d_in: int,
            n_blocks: int,
            d_main: int,
            d_hidden: int,
            dropout_first: float,
            dropout_second: float,
            d_out: int,
    ) -> 'ResNet':
        """Create a "baseline" `ResNet`.

        This variation of ResNet was used in [gorishniy2021revisiting]. Features:

        * :code:`Activation` = :code:`ReLU`
        * :code:`Norm` = :code:`BatchNorm1d`

        Args:
            d_in: the input size
            n_blocks: the number of Blocks
            d_main: the input size (or, equivalently, the output size) of each Block
            d_hidden: the output size of the first linear layer in each Block
            dropout_first: the dropout rate of the first dropout layer in each Block.
            dropout_second: the dropout rate of the second dropout layer in each Block.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        return cls(
            d_in=d_in,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization='BatchNorm1d',
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
