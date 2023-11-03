import math
import torch.nn as nn
import torch

from transformers import LlamaModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask
)
from typing import List, Optional, Tuple, Union

DEVICES = {
    -1: "cpu", **{i:i for i in range(10)}
}

def get_fcnet(inpt_size,
              outp_size,
              n_layers=2,
              h_size=256,
              noise=0,
              drop_p=0,
              bnorm=False,
              lnorm=False,
              scaleshift=True,
              legacy=False,
              actv_fxn="ReLU"):
    """
    Defines a simple fully connected Sequential module

    Args:
        inpt_size: int
            the dimension of the inputs
        outp_size: int
            the dimension of the final output
        n_layers: int
            the number of layers for the fc net
        h_size: int
            the dimensionality of the hidden layers
        noise: float
            the std of added noise before the relue at each layer.
        drop_p: float
            the probability of dropping a node
        bnorm: bool
            if true, batchnorm is included before each relu layer
        lnorm: bool
            if true, layer norm is included before each relu layer
        scaleshift: bool
            if true, a ScaleShift layer is added after the activation
            function
    """
    outsize = h_size if n_layers > 1 else outp_size
    block = [  ]
    block.append( nn.Linear(inpt_size, outsize) )
    prev_size = outsize
    for i in range(1, n_layers):
        block.append( GaussianNoise(noise) )
        block.append( nn.Dropout(drop_p) )
        block.append( globals()[actv_fxn]() )
        if bnorm: block.append( nn.BatchNorm1d(outsize) )
        if lnorm: block.append( nn.LayerNorm(outsize) )
        if scaleshift: block.append( ScaleShift((outsize,)) )
        if i+1 == n_layers: outsize = outp_size
        block.append( nn.Linear(prev_size, outsize) )
    return nn.Sequential(*block)

class CoreModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def is_cuda(self):
        try:
            return next(self.parameters()).is_cuda
        except:
            return False

    def get_device(self):
        return DEVICES[next(self.parameters()).get_device()]

    def sample_with_temperature(self, logits, temperature):
        """
        Args:
            logits: torch float tensor (..., L)
            temperature: float or None
                a value to increase the sampling entropy. ignored if
                0 or None
        """
        if not temperature: return torch.argmax(logits, dim=-1)
        ps = torch.nn.functional.softmax( logits/temperature, dim=-1 )
        return torch.multinomial(ps, num_samples=1)[...,0]


class Flatten(nn.Module):
    """
    Reshapes the activations to be of shape (B,-1) where B
    is the batch size
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape(nn.Module):
    """
    Reshapes the activations to be of shape (B, *shape) where B
    is the batch size.
    """
    def __init__(self, shape):
        """
        shape: tuple of ints
            do not include batch size
        """
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(len(x), *self.shape)

    def extra_repr(self):
        return "shape={}".format(self.shape)

class Transpose(nn.Module):
    """
    Transposes the argued axes. Do include the batch dimension in
    your argument
    """
    def __init__(self, axes, *args):
        """
        axes: tuple of ints
            do include the batch dimension
        """
        super().__init__()
        if type(axes)==int: axes = [axes] 
        else: axes = [*axes]

        if len(args) > 0:
            axes = axes + [*args]
        self.axes = axes
    
    def forward(self, x, *args, **kwargs):
        """
        x: torch tensor
        """
        return x.permute(self.axes)

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1, trainable=False, adapt=False,
                                               momentum=.95):
        """
        std - float
            the standard deviation of the noise to add to the layer.
            if adapt is true, this is used as the proportional value to
            set the std to based of the std of the activations.
            gauss_std = activ_std*std
        trainable - bool
            If trainable is set to True, then the std is turned into
            a learned parameter. Cannot be set to true if adapt is True
        adapt - bool
            adapts the gaussian std to a proportion of the
            std of the received activations. Cannot be set to True if
            trainable is True
        momentum - float (0 <= momentum < 1)
            this is the exponentially moving average factor for
            updating the activ_std. 0 uses the std of the current
            activations.
        """
        super(GaussianNoise, self).__init__()
        self.trainable = trainable
        self.adapt = adapt
        assert not (self.trainable and self.adapt)
        self.std = std
        self.sigma = nn.Parameter(torch.ones(1)*std,
                            requires_grad=trainable)
        self.running_std = 1
        self.momentum = momentum if adapt else None

    def forward(self, x):
        if not self.training or self.std == 0:
            return x
        if self.adapt:
            xstd = x.std().item()
            self.running_std = self.momentum*self.running_std +\
                                          (1-self.momentum)*xstd
            self.sigma.data[0] = self.std*self.running_std
        noise = self.sigma*torch.randn_like(x)
        return x + noise

    def extra_repr(self):
        s = 'std={}, trainable={}, adapt={}, momentum={}'
        return s.format(self.std, self.trainable,
                        self.adapt, self.momentum)

class PositionalEncoding(nn.Module):
    """
    Taken from pytorch tutorial. A simple positonal encoding taken from
    vaswani et al.
    """
    def __init__(self, d_model, dropout= 0.1, max_len= 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        Returns:
            enc: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

class ScaleShift(nn.Module):
    """
    Scales and shifts the activations by a learnable amount.
    """
    def __init__(self, shape, scale=True, shift=True):
        """
        shape: tuple (depth, height, width) or (length,)
            shape of the incoming activations discluding the
            batch dimension
        scale: bool
            include multiplicative parameter
        shift: bool
            include summing parameter
        """
        super(ScaleShift, self).__init__()
        self.shape = shape
        self.scale = scale
        self.shift = shift
        self.scale_param = nn.Parameter(torch.ones(shape).float(),
                                              requires_grad=scale)
        self.shift_param= nn.Parameter(torch.zeros(shape).float(),
                                              requires_grad=shift)
    def forward(self, x):
        return x*self.scale_param + self.shift_param

    def extra_repr(self):
        s = 'shape={}, scale={}, shift={}'
        return s.format(self.shape, self.scale, self.shift)

class NullOp(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class ContainedLSTM(nn.Module):
    """
    Contained lstms handle all recurrent vectors for you. You simply
    pass an input sequence to the forward function with the number of
    outputs you would like. It returns the outputs as a tensor (B,N,H).
    It also resets the h and c vectors at the beginning of each forward
    pass.
    """
    def __init__(self, inpt_size, h_size, lnorm=True, *args, **kwargs):
        super().__init__()
        self.inpt_size = inpt_size
        self.h_size = h_size
        self.lstm = nn.LSTMCell(self.inpt_size, self.h_size)
        self.lnorm = lnorm
        if self.lnorm:
            self.lnorm_h = nn.LayerNorm(self.h_size)
        self.register_buffer('h', torch.zeros(1,self.h_size))
        self.register_buffer('c', torch.zeros(1,self.h_size))

    def forward(self, x, mask=None):
        """
        Args:
            x: torch tensor (B, S, I)
            mask: torch bool tensor (B,S)
                a boolean tensor where true denotes that the end of the
                sequence has been reached. These inputs are not
                included.
        Returns:
            fx: torch tensor (B, H)
        """
        h = self.h.repeat(len(x), 1)
        c = self.c.repeat(len(x), 1)
        output = torch.zeros_like(h)
        for i in range(x.shape[1]):
            if self.lnorm:
                h = self.lnorm_h(h)
            h, c = self.lstm(x[:,i], (h,c))
            if mask is not None:
                output[~mask[:,i]] = h[~mask[:,i]]
            else: output = h
        return output

class GenerativeLSTM(nn.Module):
    """
    This module handles all recurrent vectors for you. You simply
    pass the input in to the forward function with the number of
    outputs you would like. It returns the outputs as a tensor (B,N,H).
    It also resets the h and c vectors at the beginning of each forward
    pass.
    """
    def __init__(self, inpt_size, h_size, lnorm=True, *args, **kwargs):
        super().__init__()
        self.inpt_size = inpt_size
        self.h_size = h_size
        self.lstm = nn.LSTMCell(self.inpt_size, self.h_size)
        self.lnorm = lnorm
        if self.lnorm:
            self.lnorm_h = nn.LayerNorm(self.h_size)
            self.lnorm_c = nn.LayerNorm(self.h_size)
        self.register_buffer('h', torch.zeros(1,self.h_size))
        self.register_buffer('c', torch.zeros(1,self.h_size))

    def forward(self, x, n):
        """
        Args:
            x: torch tensor (B, I)
            n: int
                the number of recurrent loops
        Returns:
            fx: torch tensor (B, N, H)
        """
        h = self.h.repeat(len(x), 1)
        c = self.c.repeat(len(x), 1)
        outpts = []
        for _ in range(n):
            if self.lnorm:
                h,c = self.lnorm_h(h), self.lnorm_c(c)
            h, c = self.lstm(x, (h,c))
            outpts.append(h)
        return torch.stack(outpts, dim=1)

class CrossAttention(nn.Module):
    """
    Builds off the pytorch multihead attention module to combine multiple
    different modalities symettrcially into a single multi-head attention.
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_modes=2,
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=True,
                 kdim=None,
                 vdim=None,
                 batch_first=True,
                 device=None,
                 dtype=None,
                 *args, **kwargs) -> None:
        """
        Args:
            embed_dim: int
                Total dimension of the model.
            num_heads: int
                Number of parallel attention heads. Note that embed_dim
                will be split across num_heads (i.e. each head will have
                dimension embed_dim // num_heads).
            num_modes: int
                the number of modalities to be combined into the
                self-attention.
            dropout: float
                Dropout probability on attn_output_weights. Default:
                0.0 (no dropout).
            bias: bool
                If specified, adds bias to input / output projection
                layers. Default: True.
            add_bias_kv: bool
                If specified, adds bias to the key and value sequences
                at dim=0. Default: False.
            add_zero_attn: bool
                If specified, adds a new batch of zeros to the key and
                value sequences at dim=1. Default: False.
            kdim: int
                Total number of features for keys. Default: None
                (uses kdim=embed_dim).
            vdim: int
                Total number of features for values. Default: None
                (uses vdim=embed_dim).
            batch_first: bool
                If True, then the input and output tensors are provided
                as (batch, seq, feature). Default: False (seq, batch,
                feature).
            device: int or str
            dtype: str
        """
        super().__init__(*args, **kwargs)
        self.mh_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.num_modes = num_modes
        self.mode_encodings = nn.Embedding(self.num_modes, embed_dim)
        torch.nn.init.kaiming_uniform_(
            self.mode_encodings.weight,
            mode='fan_in',
            nonlinearity='leaky_relu'
        )

    def forward(self,
                queries,
                keys,
                values,
                key_padding_masks=None,
                need_weights=True,
                step_masks=None,
                is_causal=True,
                average_attn_weights=True,
                tforce=True,
                *args, **kwargs):
        """
        Args:
            queries: (List of Tensors)
                One entry for each modality.
                Query embeddings of shape (L,Eq)(L,Eq​) for unbatched
                input, (L,N,Eq)(L,N,Eq​) when batch_first=False or
                (N,L,Eq)(N,L,Eq​) when batch_first=True, where LL is the
                target sequence length, NN is the batch size, and EqEq​
                is the query embedding dimension embed_dim. Queries are
                compared against key-value pairs to produce the output.
                See “Attention Is All You Need” for more details.
            keys: (List of Tensors)
                One entry for each modality.
                Key embeddings of shape (S,Ek)(S,Ek​) for unbatched input,
                (S,N,Ek)(S,N,Ek​) when batch_first=False or
                (N,S,Ek)(N,S,Ek​) when batch_first=True, where SS is the
                source sequence length, NN is the batch size, and EkEk​
                is the key embedding dimension kdim. See “Attention Is
                All You Need” for more details.
            values: (List of Tensors)
                One entry for each modality.
                Value embeddings of shape (S,Ev)(S,Ev​) for unbatched
                input, (S,N,Ev)(S,N,Ev​) when batch_first=False or
                (N,S,Ev)(N,S,Ev​) when batch_first=True, where SS is the
                source sequence length, NN is the batch size, and EvEv​
                is the value embedding dimension vdim. See “Attention Is
                All You Need” for more details.
            key_padding_masks: (Optional[List of Tensors])
                One entry for each modality.
                If specified, a mask of shape (N,S)(N,S) indicating
                which elements within key to ignore for the purpose of
                attention (i.e. treat as “padding”). For unbatched query,
                shape should be (S)(S). Binary and float masks are
                supported. For a binary mask, a True value indicates that
                the corresponding key value will be ignored for the
                purpose of attention. For a float mask, it will be
                directly added to the corresponding key value.
            need_weights: (bool)
                If specified, returns attn_output_weights in addition
                to attn_outputs. Set need_weights=False to use the
                optimized scaled_dot_product_attention and achieve the
                best performance for MHA. Default: True.
            step_masks: (Optional[List of LongTensors])
                One entry for each modality. A list of 2D masks denoting
                step of the information relative to the other modalities.
                This allows you to use causal masking based on the step
                of an environment instead of the step of each embedding,
                preventing attention to positions that are at a future
                state of the environment. Must be of shape [(NN,S1), ...,
                (NN,Sk)], where NN is the batch size, S1 is the sequence
                length of the first modality and Sk is the sequence
                length of the kth modality. Only Long type masks
                are supported.
            average_attn_weights: (bool)
                If true, indicates that the returned attn_weights should
                be averaged across heads. Otherwise, attn_weights are
                provided separately per head. Note that this flag only
                has an effect when need_weights=True. Default: True
                (i.e. average weights across heads)
            is_causal: (bool)
                If true, applies a causal mask within each modality.
            tforce: bool
                If true, will use all queries. If false,
                will use only the last embedding of each modality as
                the queries (saving computation).
        """
        cross_mask = self.get_cross_mask(step_masks)
        if not tforce:
            # only take the latest queries and the corresponding masks
            running_sum = 0
            idxs = []
            for q in queries:
                running_sum += len(q)
                idxs.append(running_sum-1)
            idxs = torch.LongTensor(idxs)
            cross_mask = cross_mask[idxs]
            queries = [ q[:,-1:] for q in queries ]
        queries = [
          q + self.mode_encodings.weight[i] for i,q in enumerate(queries)
        ]
        keys = [
          k + self.mode_encodings.weight[i] for i,k in enumerate(keys)
        ]
        values = [
          v + self.mode_encodings.weight[i] for i,v in enumerate(values)
        ]
        pad_mask = None
        if key_padding_masks is not None:
            pad_mask = torch.cat(key_padding_masks, dim=1)
        query = torch.cat(queries, dim=1)
        key =   torch.cat(keys, dim=1)
        value = torch.cat(values, dim=1)
        attn_out, attn_weights = self.mh_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=pad_mask,
            attn_mask=cross_mask,
        )
        if need_weights:
            return attn_out, attn_weights
        return attn_out


    def get_cross_mask(self, step_masks):
        """

        """

class FlexibleLlamaModel(LlamaModel):
    """
    Overrides the forward function for more flexible attention inputs.
    Allows the model to use non-causal attention if desired. Does not
    allow flash attention when using non-cauasal attention.
    """
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions is None:
            output_attentions = self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None\
                else self.config.output_hidden_states
        )
        if use_cache is None:
            use_cache = self.config.use_cache

        if return_dict is None:
            return_dict = self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds "
                "at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            if input_ids is not None: device = input_ids.device
            else: device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        flash = getattr(self.config, "_flash_attn_2_enabled", False)
        if len(attention_mask.shape)==2 and flash:
            # 2d mask is passed through the layers
            if not (attention_mask is not None and 0 in attention_mask):
                attention_mask = None
        elif len(attention_mask.shape)==2:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length
            )
        elif len(attention_mask.shape)==3:
            #n_heads = self.config.num_attention_heads
            #attention_mask = attention_mask[:,None].repeat(1,n_heads,1,1)
            attention_mask = attention_mask[:,None]

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache: use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if past_key_values is None:
                past_key_value = None
            else:
                past_key_value = past_key_values[idx]

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],
                )

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            el = [
              hidden_states,next_cache,all_hidden_states,all_self_attns
            ]
            return tuple( v for v in el if v is not None )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

