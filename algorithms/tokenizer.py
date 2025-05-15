import enum
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from framework.dataset import TabularDataset
from framework.base import LinkedModule
from framework.utils import to_numpy


class CategoricalFeatureTokenizer(nn.Module):
    """Transforms categorical features to tokens (embeddings).

    See `FeatureTokenizer` for the illustration.

    The module efficiently implements a collection of `torch.nn.Embedding` (with
    optional biases).

    Examples:
        .. testcode::

            # the input must contain integers. For example, if the first feature can
            # take 3 distinct values, then its cardinality is 3 and the first column
            # must contain values from the range `[0, 1, 2]`.
            cardinalities = [3, 10]
            x = torch.tensor([
                [0, 5],
                [1, 7],
                [0, 2],
                [2, 4]
            ])
            n_objects, n_features = x.shape
            d_token = 3
            tokenizer = CategoricalFeatureTokenizer(cardinalities, d_token, True, 'uniform')
            tokens = tokenizer(x)
            assert tokens.shape == (n_objects, n_features, d_token)
    """

    category_offsets: torch.Tensor

    def __init__(
            self,
            cardinalities: List[int],
            d_token: int,
            bias: bool,
            initialization: str,
    ) -> None:
        """
        Args:
            cardinalities: the number of distinct values for each feature. For example,
                :code:`cardinalities=[3, 4]` describes two features: the first one can
                take values in the range :code:`[0, 1, 2]` and the second one can take
                values in the range :code:`[0, 1, 2, 3]`.
            d_token: the size of one token.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of feature value. The bias vectors are not shared
                between features.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        assert cardinalities, 'cardinalities must be non-empty'
        assert d_token > 0, 'd_token must be positive'
        initialization_ = _TokenInitialization.from_str(initialization)

        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(cardinalities), d_token)
        self.bias = nn.Parameter(torch.Tensor(len(cardinalities), d_token)) if bias else None

        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.category_offsets)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.embeddings.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class NumericalFeatureTokenizer(nn.Module):
    """Transforms continuous features to tokens (embeddings).

    See `FeatureTokenizer` for the illustration.

    For one feature, the transformation consists of two steps:

    * the feature is multiplied by a trainable vector
    * another trainable vector is added

    Note that each feature has its separate pair of trainable vectors, i.e. the vectors
    are not shared between features.

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            n_objects, n_features = x.shape
            d_token = 3
            tokenizer = NumericalFeatureTokenizer(n_features, d_token, True, 'uniform')
            tokens = tokenizer(x)
            assert tokens.shape == (n_objects, n_features, d_token)
    """

    def __init__(
            self,
            n_features: int,
            d_token: int,
            bias: bool,
            initialization: str,
    ) -> None:
        """
        Args:
            n_features: the number of continuous (scalar) features
            d_token: the size of one token
            bias: if `False`, then the transformation will include only multiplication.
                **Warning**: :code:`bias=False` leads to significantly worse results for
                Transformer-like (token-based) architectures.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`.
                In [gorishniy2021revisiting], the 'uniform' initialization was used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(torch.Tensor(n_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(n_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.weight)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.weight.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class FeatureTokenizer(nn.Module):
    """Combines `NumericalFeatureTokenizer` and `CategoricalFeatureTokenizer`.

    The "Feature Tokenizer" module from [gorishniy2021revisiting]. The module transforms
    continuous and categorical features to tokens (embeddings).

    In the illustration below, the red module in the upper brackets represents
    `NumericalFeatureTokenizer` and the green module in the lower brackets represents
    `CategoricalFeatureTokenizer`.

    .. image:: ../images/feature_tokenizer.png
        :scale: 33%
        :alt: Feature Tokenizer

    Examples:
        .. testcode::

            n_objects = 4
            n_num_features = 3
            n_cat_features = 2
            d_token = 7
            x_num = torch.randn(n_objects, n_num_features)
            x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])
            # [2, 3] reflects cardinalities fr
            tokenizer = FeatureTokenizer(n_num_features, [2, 3], d_token)
            tokens = tokenizer(x_num, x_cat)
            assert tokens.shape == (n_objects, n_num_features + n_cat_features, d_token)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    def __init__(
            self,
            n_num_features: int,
            cat_cardinalities: List[int],
            d_token: int,
    ) -> None:
        """
        Args:
            n_num_features: the number of continuous features. Pass :code:`0` if there
                are no numerical features.
            cat_cardinalities: the number of unique values for each feature. See
                `CategoricalFeatureTokenizer` for details. Pass an empty list if there
                are no categorical features.
            d_token: the size of one token.
        """
        super().__init__()
        assert n_num_features >= 0, 'n_num_features must be non-negative'
        assert (
                n_num_features or cat_cardinalities
        ), 'at least one of n_num_features or cat_cardinalities must be positive/non-empty'
        self.initialization = 'uniform'
        self.num_tokenizer = (
            NumericalFeatureTokenizer(
                n_features=n_num_features,
                d_token=d_token,
                bias=True,
                initialization=self.initialization,
            )
            if n_num_features
            else None
        )
        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(
                cat_cardinalities, d_token, True, self.initialization
            )
            if cat_cardinalities
            else None
        )

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return sum(
            x.n_tokens
            for x in [self.num_tokenizer, self.cat_tokenizer]
            if x is not None
        )

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return (
            self.cat_tokenizer.d_token  # type: ignore
            if self.num_tokenizer is None
            else self.num_tokenizer.d_token
        )

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """Perform the forward pass.

        Args:
            x_num: continuous features. Must be presented if :code:`n_num_features > 0`
                was passed to the constructor.
            x_cat: categorical features (see `CategoricalFeatureTokenizer.forward` for
                details). Must be presented if non-empty :code:`cat_cardinalities` was
                passed to the constructor.
        Returns:
            tokens
        Raises:
            AssertionError: if the described requirements for the inputs are not met.
        """
        assert (
                x_num is not None or x_cat is not None
        ), 'At least one of x_num and x_cat must be presented'
        assert all_or_none(
            [self.num_tokenizer, x_num]
        ), 'If self.num_tokenizer is (not) None, then x_num must (not) be None'
        assert all_or_none(
            [self.cat_tokenizer, x_cat]
        ), 'If self.cat_tokenizer is (not) None, then x_cat must (not) be None'
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)


class CLSToken(nn.Module):
    """[CLS]-token for BERT-like inference.

    To learn about the [CLS]-based inference, see [devlin2018bert].

    When used as a module, the [CLS]-token is appended **to the end** of each item in
    the batch.

    Examples:
        .. testcode::

            batch_size = 2
            n_tokens = 3
            d_token = 4
            cls_token = CLSToken(d_token, 'uniform')
            x = torch.randn(batch_size, n_tokens, d_token)
            x = cls_token(x)
            assert x.shape == (batch_size, n_tokens + 1, d_token)
            assert (x[:, -1, :] == cls_token.expand(len(x))).all()

    References:
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    """

    def __init__(self, d_token: int, initialization: str) -> None:
        """
        Args:
            d_token: the size of token
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(torch.Tensor(d_token))
        initialization_.apply(self.weight, d_token)

    def expand(self, *leading_dimensions: int) -> torch.Tensor:
        """Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.

        A possible use case is building a batch of [CLS]-tokens. See `CLSToken` for
        examples of usage.

        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the
            underlying :code:`weight` parameter, so gradients will be propagated as
            expected.

        Args:
            leading_dimensions: the additional new dimensions

        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Append self **to the end** of each item in the batch (see `CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)


def all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)


class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: str) -> '_TokenInitialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')

    def apply(self, x: torch.Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class TokenizerModel(LinkedModule):
    def init_module(self, dataset: TabularDataset) -> TabularDataset:
        self.cat_idx = dataset.cat_idx
        self.num_idx = dataset.num_idx
        self.tokenizer = None
        self.trainable = self.config['trainable']
        self.token_dim = self.config['token_dim']
        if len(self.cat_idx) > 0:
            if self.trainable:
                self.tokenizer = CategoricalFeatureTokenizer(
                    dataset.cat_dims, self.token_dim, False, "uniform"
                ).to(self.device)
                dataset.data_dim = len(self.num_idx) + self.tokenizer.n_tokens * self.tokenizer.d_token
        self.loadable_items = ['tokenizer']
        return dataset

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        if len(self.cat_idx) > 0:
            if self.trainable:
                # Scaling cat features with the tokenizer
                x_num = X[:, self.num_idx]
                x_cat = X[:, self.cat_idx].to(torch.int)
                x_ordered = torch.cat([x_num, self.tokenizer(x_cat).flatten(1, -1)], dim=1)
                return x_ordered, y
            else:
                # Standard scaler on the cat features
                X = X.to(torch.float)
                with torch.no_grad():
                    mu = X.mean(0, keepdim=True)
                    mu[:, self.num_idx] = 0
                    stdv = X.std(0, unbiased=False, keepdim=True)
                    stdv[:, self.num_idx] = 1
                X = X - mu 
                X = X / stdv

                # X[:, self.cat_idx] -= X[:, self.cat_idx].mean(0, keepdim=True)
                # X[:, self.cat_idx] /= X[:, self.cat_idx].std(0, unbiased=False, keepdim=True)
                
                # x_cat = X[:, self.cat_idx].to(torch.float)
                # x_cat -= x_cat.mean(0, keepdim=True)
                # x_cat /= x_cat.std(0, unbiased=False, keepdim=True)
                # X[:, self.cat_idx] = x_cat
                return X, y
        else:
            return X, y
