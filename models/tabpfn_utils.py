import os
import math
import random
import typing
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.nn.modules.transformer as nn_transformer


########################################################################################################################
#################################################  utils.py  ###########################################################
########################################################################################################################

class SeqBN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.d_model = d_model

    def forward(self, x):
        assert self.d_model == x.shape[-1]
        flat_x = x.view(-1, self.d_model)
        flat_x = self.bn(flat_x)
        return flat_x.view(*x.shape)


# TODO: Is there a better way to do this?
#   1. Cmparing to unique elements: When all values are different we still get quadratic blowup
#   2. Argsort(Argsort()) returns ranking, but with duplicate values there is an ordering which is problematic
#   3. Argsort(Argsort(Unique))->Scatter seems a bit complicated, doesn't have quadratic blowup, but how fast?
def to_ranking_low_mem(data):
    x = torch.zeros_like(data)
    for col in range(data.shape[-1]):
        x_ = (data[:, :, col] >= data[:, :, col].unsqueeze(-2))
        x_ = x_.sum(0)
        x[:, :, col] = x_
    return x


def torch_masked_mean(x, mask, dim=0, return_share_of_ignored_values=False):
    """
    Returns the mean of a torch tensor and only considers the elements, where the mask is true.
    If return_share_of_ignored_values is true it returns a second tensor with the percentage of ignored values
    because of the mask.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    if return_share_of_ignored_values:
        return value / num, 1. - num / x.shape[dim]
    return value / num


def torch_masked_std(x, mask, dim=0):
    """
    Returns the std of a torch tensor and only considers the elements, where the mask is true.
    If get_mean is true it returns as a first Tensor the mean and as a second tensor the std.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(dim), x.shape[dim], dim=dim)
    quadratic_difference_from_mean = torch.square(torch.where(mask, mean_broadcast - x, torch.full_like(x, 0)))
    return torch.sqrt(torch.sum(quadratic_difference_from_mean, dim=dim) / (num - 1))


def torch_nanmean(x, dim=0, return_nanshare=False):
    return torch_masked_mean(x, ~torch.isnan(x), dim=dim, return_share_of_ignored_values=return_nanshare)


def torch_nanstd(x, dim=0):
    return torch_masked_std(x, ~torch.isnan(x), dim=dim)


def normalize_data(data, normalize_positions=-1):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], dim=0)
        std = torch_nanstd(data[:normalize_positions], dim=0) + .000001
    else:
        mean = torch_nanmean(data, dim=0)
        std = torch_nanstd(data, dim=0) + .000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    return data


def bool_mask_to_att_mask(mask):
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))


def remove_outliers(X, n_sigma=4, normalize_positions=-1):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"

    data = X if normalize_positions == -1 else X[:normalize_positions]

    data_mean, data_std = torch_nanmean(data, dim=0), torch_nanstd(data, dim=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    mask = (data <= upper) & (data >= lower) & ~torch.isnan(data)
    data_mean, data_std = torch_masked_mean(data, mask), torch_masked_std(data, mask)

    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)
    return X


# NOP function for python with statements (x = NOP(); with x:)
class NOP():
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def normalize_by_used_features_f(x, num_features_used, num_features, normalize_with_sqrt=False):
    if normalize_with_sqrt:
        return x / (num_features_used / num_features) ** (1 / 2)
    return x / (num_features_used / num_features)


def get_uniform_single_eval_pos_sampler(max_len, min_len=0):
    """
    Just sample any evaluation position with the same weight
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(range(min_len, max_len))[0]


########################################################################################################################
##########################################  script.model_builder.py  ###################################################
########################################################################################################################


def load_model_only_inference(path, filename, device):
    """
    Loads a saved model from the specified position. This function only restores inference capabilities and
    cannot be used for further training.
    """

    model_state, optimizer_state, config_sample = torch.load(os.path.join(path, filename), map_location='cpu')

    if (('nan_prob_no_reason' in config_sample and config_sample['nan_prob_no_reason'] > 0.0) or
            ('nan_prob_a_reason' in config_sample and config_sample['nan_prob_a_reason'] > 0.0) or
            ('nan_prob_unknown_reason' in config_sample and config_sample['nan_prob_unknown_reason'] > 0.0)):
        encoder = NanHandlingEncoder
    else:
        encoder = partial(Linear, replace_nan_by_zero=True)

    n_out = config_sample['max_num_classes']

    device = device if torch.cuda.is_available() else 'cpu:0'
    encoder = encoder(config_sample['num_features'], config_sample['emsize'])

    nhid = config_sample['emsize'] * config_sample['nhid_factor']
    y_encoder_generator = get_Canonical(config_sample['max_num_classes']) \
        if config_sample.get('canonical_y_encoder', False) else Linear

    assert config_sample['max_num_classes'] > 2
    loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.ones(int(config_sample['max_num_classes'])))

    model = TransformerModel(encoder, n_out, config_sample['emsize'], config_sample['nhead'], nhid,
                             config_sample['nlayers'], y_encoder=y_encoder_generator(1, config_sample['emsize']),
                             dropout=config_sample['dropout'],
                             efficient_eval_masking=config_sample['efficient_eval_masking'])

    # print(f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters")

    model.criterion = loss
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return (float('inf'), float('inf'), model), config_sample  # no loss measured


########################################################################################################################
################################################  encoders.py  #########################################################
########################################################################################################################

class NanHandlingEncoder(nn.Module):
    def __init__(self, num_features, emsize, keep_nans=True):
        super().__init__()
        self.num_features = 2 * num_features if keep_nans else num_features
        self.emsize = emsize
        self.keep_nans = keep_nans
        self.layer = nn.Linear(self.num_features, self.emsize)

    def forward(self, x):
        if self.keep_nans:
            x = torch.cat([torch.nan_to_num(x, nan=0.0), normalize_data(torch.isnan(x) * -1
                                                                        + torch.logical_and(torch.isinf(x),
                                                                                            torch.sign(x) == 1) * 1
                                                                        + torch.logical_and(torch.isinf(x),
                                                                                            torch.sign(x) == -1) * 2
                                                                        )], -1)
        else:
            x = torch.nan_to_num(x, nan=0.0)
        return self.layer(x)


class Linear(nn.Linear):
    def __init__(self, num_features, emsize, replace_nan_by_zero=False):
        super().__init__(num_features, emsize)
        self.num_features = num_features
        self.emsize = emsize
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x):
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return super().forward(x)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('replace_nan_by_zero', True)


class CanEmb(nn.Embedding):
    def __init__(self, num_features, num_embeddings: int, embedding_dim: int, *args, **kwargs):
        assert embedding_dim % num_features == 0
        embedding_dim = embedding_dim // num_features
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)

    def forward(self, x):
        lx = x.long()
        assert (lx == x).all(), "CanEmb only works with tensors of whole numbers"
        x = super().forward(lx)
        return x.view(*x.shape[:-2], -1)


def get_Canonical(num_classes):
    return lambda num_features, emsize: CanEmb(num_features, num_classes, emsize)


########################################################################################################################
###############################################  transformer.py  #######################################################
########################################################################################################################

class TransformerModel(nn.Module):
    def __init__(self, encoder, n_out, ninp, nhead, nhid, nlayers, dropout=0.0, style_encoder=None, y_encoder=None,
                 pos_encoder=None, decoder=None, input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, num_global_att_tokens=0, full_attention=False,
                 all_layers_same_init=False, efficient_eval_masking=True):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layer_creator = lambda: TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation=activation,
                                                                pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer_creator(), nlayers) \
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.ninp = ninp
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder
        self.decoder = decoder(ninp, nhid, n_out) if decoder is not None else nn.Sequential(nn.Linear(ninp, nhid),
                                                                                            nn.GELU(),
                                                                                            nn.Linear(nhid, n_out))
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.style_encoder = style_encoder
        self.init_method = init_method
        if num_global_att_tokens is not None:
            assert not full_attention
        self.global_att_embeddings = nn.Embedding(num_global_att_tokens, ninp) if num_global_att_tokens else None
        self.full_attention = full_attention
        self.efficient_eval_masking = efficient_eval_masking

        self.n_out = n_out
        self.nhid = nhid

        self.init_weights()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('efficient_eval_masking', False)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        train_size = sz - query_size
        mask = torch.zeros(sz, sz) == 0
        mask[:, train_size:].zero_()
        mask |= torch.eye(sz) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_query_matrix(num_global_att_tokens, seq_len, num_query_tokens):
        train_size = seq_len + num_global_att_tokens - num_query_tokens
        sz = seq_len + num_global_att_tokens
        mask = torch.zeros(num_query_tokens, sz) == 0
        mask[:, train_size:].zero_()
        mask[:, train_size:] |= torch.eye(num_query_tokens) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_trainset_matrix(num_global_att_tokens, seq_len, num_query_tokens):
        train_size = seq_len + num_global_att_tokens - num_query_tokens
        trainset_size = seq_len - num_query_tokens
        mask = torch.zeros(trainset_size, num_global_att_tokens) == 0
        # mask[:,num_global_att_tokens:].zero_()
        # mask[:,num_global_att_tokens:] |= torch.eye(trainset_size) == 1
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_global_att_globaltokens_matrix(num_global_att_tokens, seq_len, num_query_tokens):
        mask = torch.zeros(num_global_att_tokens, num_global_att_tokens + seq_len - num_query_tokens) == 0
        return bool_mask_to_att_mask(mask)

    def init_weights(self):
        initrange = 1.
        # if isinstance(self.encoder,EmbeddingEncoder):
        #    self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.init_method is not None:
            self.apply(self.init_method)
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)

    def forward(self, src, src_mask=None, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src

        style_src, x_src, y_src = src
        x_src = self.encoder(x_src)
        y_src = self.y_encoder(y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src)
        style_src = self.style_encoder(style_src).unsqueeze(0) if self.style_encoder else \
            torch.tensor([], device=x_src.device)
        global_src = torch.tensor([], device=x_src.device) if self.global_att_embeddings is None else \
            self.global_att_embeddings.weight.unsqueeze(1).repeat(1, x_src.shape[1], 1)

        if src_mask is not None: assert self.global_att_embeddings is None or isinstance(src_mask, tuple)
        if src_mask is None:
            if self.global_att_embeddings is None:
                full_len = len(x_src) + len(style_src)
                if self.full_attention:
                    src_mask = bool_mask_to_att_mask(torch.ones((full_len, full_len), dtype=torch.bool)).to(
                        x_src.device)
                elif self.efficient_eval_masking:
                    src_mask = single_eval_pos + len(style_src)
                else:
                    src_mask = self.generate_D_q_matrix(full_len, len(x_src) - single_eval_pos).to(x_src.device)
            else:
                src_mask_args = (self.global_att_embeddings.num_embeddings,
                                 len(x_src) + len(style_src),
                                 len(x_src) + len(style_src) - single_eval_pos)
                src_mask = (self.generate_global_att_globaltokens_matrix(*src_mask_args).to(x_src.device),
                            self.generate_global_att_trainset_matrix(*src_mask_args).to(x_src.device),
                            self.generate_global_att_query_matrix(*src_mask_args).to(x_src.device))

        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        src = torch.cat([global_src, style_src, train_x, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output[single_eval_pos + len(style_src) + (
            self.global_att_embeddings.num_embeddings if self.global_att_embeddings else 0):]

    @torch.no_grad()
    def init_from_small_model(self, small_model):
        assert isinstance(self.decoder, nn.Linear) and isinstance(self.encoder, (nn.Linear, nn.Sequential)) \
               and isinstance(self.y_encoder, (nn.Linear, nn.Sequential))

        def set_encoder_weights(my_encoder, small_model_encoder):
            my_encoder_linear, small_encoder_linear = (my_encoder, small_model_encoder) \
                if isinstance(my_encoder, nn.Linear) else (my_encoder[-1], small_model_encoder[-1])
            small_in_dim = small_encoder_linear.out_features
            my_encoder_linear.weight.zero_()
            my_encoder_linear.bias.zero_()
            my_encoder_linear.weight[:small_in_dim] = small_encoder_linear.weight
            my_encoder_linear.bias[:small_in_dim] = small_encoder_linear.bias

        set_encoder_weights(self.encoder, small_model.encoder)
        set_encoder_weights(self.y_encoder, small_model.y_encoder)

        small_in_dim = small_model.decoder.in_features

        self.decoder.weight[:, :small_in_dim] = small_model.decoder.weight
        self.decoder.bias = small_model.decoder.bias

        for my_layer, small_layer in zip(self.transformer_encoder.layers, small_model.transformer_encoder.layers):
            small_hid_dim = small_layer.linear1.out_features
            my_in_dim = my_layer.linear1.in_features

            # packed along q,k,v order in first dim
            my_in_proj_w = my_layer.self_attn.in_proj_weight
            small_in_proj_w = small_layer.self_attn.in_proj_weight

            my_in_proj_w.view(3, my_in_dim, my_in_dim)[:, :small_in_dim, :small_in_dim] = small_in_proj_w.view(3,
                                                                                                               small_in_dim,
                                                                                                               small_in_dim)
            my_layer.self_attn.in_proj_bias.view(3, my_in_dim)[:,
            :small_in_dim] = small_layer.self_attn.in_proj_bias.view(3, small_in_dim)

            my_layer.self_attn.out_proj.weight[:small_in_dim, :small_in_dim] = small_layer.self_attn.out_proj.weight
            my_layer.self_attn.out_proj.bias[:small_in_dim] = small_layer.self_attn.out_proj.bias

            my_layer.linear1.weight[:small_hid_dim, :small_in_dim] = small_layer.linear1.weight
            my_layer.linear1.bias[:small_hid_dim] = small_layer.linear1.bias

            my_layer.linear2.weight[:small_in_dim, :small_hid_dim] = small_layer.linear2.weight
            my_layer.linear2.bias[:small_in_dim] = small_layer.linear2.bias

            my_layer.norm1.weight[:small_in_dim] = math.sqrt(small_in_dim / my_in_dim) * small_layer.norm1.weight
            my_layer.norm2.weight[:small_in_dim] = math.sqrt(small_in_dim / my_in_dim) * small_layer.norm2.weight

            my_layer.norm1.bias[:small_in_dim] = small_layer.norm1.bias
            my_layer.norm2.bias[:small_in_dim] = small_layer.norm2.bias


class TransformerEncoderDiffInit(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: typing.Optional[torch.Tensor] = None,
                src_key_padding_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


########################################################################################################################
###################################################  layer.py  #########################################################
########################################################################################################################

class TransformerEncoderLayer(nn_transformer.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, pre_norm=False,
                 device=None, dtype=None, recompute_attn=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn_transformer.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                           **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn_transformer.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn_transformer.Dropout(dropout)
        self.linear2 = nn_transformer.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn_transformer.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn_transformer.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn_transformer.Dropout(dropout)
        self.dropout2 = nn_transformer.Dropout(dropout)
        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn

        self.activation = nn_transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src: nn_transformer.Tensor, src_mask: nn_transformer.Optional[nn_transformer.Tensor] = None,
                src_key_padding_mask: nn_transformer.Optional[nn_transformer.Tensor] = None) -> nn_transformer.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.pre_norm:
            src_ = self.norm1(src)
        else:
            src_ = src
        if isinstance(src_mask, tuple):
            # global attention setup
            assert not self.self_attn.batch_first
            assert src_key_padding_mask is None

            global_src_mask, trainset_src_mask, valset_src_mask = src_mask

            num_global_tokens = global_src_mask.shape[0]
            num_train_tokens = trainset_src_mask.shape[0]

            global_tokens_src = src_[:num_global_tokens]
            train_tokens_src = src_[num_global_tokens:num_global_tokens + num_train_tokens]
            global_and_train_tokens_src = src_[:num_global_tokens + num_train_tokens]
            eval_tokens_src = src_[num_global_tokens + num_train_tokens:]

            attn = partial(checkpoint, self.self_attn) if self.recompute_attn else self.self_attn

            global_tokens_src2 = \
                attn(global_tokens_src, global_and_train_tokens_src, global_and_train_tokens_src, None, True,
                     global_src_mask)[0]
            train_tokens_src2 = \
                attn(train_tokens_src, global_tokens_src, global_tokens_src, None, True, trainset_src_mask)[0]
            eval_tokens_src2 = attn(eval_tokens_src, src_, src_,
                                    None, True, valset_src_mask)[0]

            src2 = torch.cat([global_tokens_src2, train_tokens_src2, eval_tokens_src2], dim=0)

        elif isinstance(src_mask, int):
            assert src_key_padding_mask is None
            single_eval_position = src_mask
            src_left = \
                self.self_attn(src_[:single_eval_position], src_[:single_eval_position], src_[:single_eval_position])[0]
            src_right = \
                self.self_attn(src_[single_eval_position:], src_[:single_eval_position], src_[:single_eval_position])[0]
            src2 = torch.cat([src_left, src_right], dim=0)
        else:
            if self.recompute_attn:
                src2 = checkpoint(self.self_attn, src_, src_, src_, src_key_padding_mask, True, src_mask)[0]
            else:
                src2 = self.self_attn(src_, src_, src_, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        if not self.pre_norm:
            src = self.norm1(src)

        if self.pre_norm:
            src_ = self.norm2(src)
        else:
            src_ = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        src = src + self.dropout2(src2)

        if not self.pre_norm:
            src = self.norm2(src)
        return src
