import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from .utils import (
    padmask2attnmask,
    get_causal_cross_mask,
    get_causal_mask,
    get_full_cross_mask,
)

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
        Returns:
            samples: torch Long Tensor (...,)
        """
        if not temperature: return torch.argmax(logits, dim=-1)
        ps = torch.nn.functional.softmax( logits/temperature, dim=-1 )
        og_shape = ps.shape
        ps = ps.reshape(-1, ps.shape[-1])
        samp = torch.multinomial(ps, num_samples=1)[...,0]
        return samp.reshape(og_shape[:-1])


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
    different modalities symetrically into a single multi-head attention.
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
                step_masks=None,
                need_weights=True,
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
        cross_mask = get_full_cross_mask(step_masks) # (B,S1+S2,S1+S2)
        pad_mask = torch.cat(key_padding_masks, dim=-1)# (B,S1+S2)
        pad_mask = padmask2attnmask(pad_mask) # (B,S1+S2,S1+S2)
        attn_mask = ~(cross_mask|pad_mask)

        if not tforce:
            # only take the latest queries and the corresponding masks
            running_sum = 0
            idxs = []
            # TODO: Need to transpose to index into correct axis
            for q in queries:
                running_sum += q.shape[1]
                idxs.append(running_sum-1)
            idxs = torch.LongTensor(idxs,device=self.get_device())
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

class CrossAttentionPrep(nn.Module):
    """
    This module preps the incoming sequence to be used by a standard
    transformer by applying modality specific embeddings and building
    the cross-modal attention mask.
    """
    def __init__(self,
                 embed_dim,
                 num_modes=2,
                 dtype=None,
                 *args, **kwargs) -> None:
        """
        Args:
            embed_dim: int
                Total dimension of the model.
            num_modes: int
                the number of modalities to be combined into the
                self-attention.
            device: int or str
            dtype: str
        """
        super().__init__(*args, **kwargs)
        self.num_modes = num_modes
        self.mode_encodings = nn.Parameter(
            0.01*torch.randn(self.num_modes, embed_dim)
        )
        torch.nn.init.kaiming_uniform_(
            self.mode_encodings,
            mode='fan_in',
            nonlinearity='leaky_relu'
        )

    def forward(self,
                inpt_list,
                pad_masks=None,
                step_masks=None,
                *args, **kwargs):
        """
        Args:
            inpt_list: list of torch FloatTensors [(B,S1,E), (B,S2,E)]
                a list of the embedding/latent vectors.
            pad_masks: list of torch BoolTensors [(B,S1), (B,S2)]
                A list of the pad masks. A True value indicates that
                the corresponding key value will be ignored for the
                purpose of attention. True means padding.
            step_masks: list of torch LongTensors [(B,S1), (B,S2)]
                One entry for each modality. Currently only 2 modalities
                are supported. It should be a list of masks denoting
                step of the information relative to the other modalities.
                This allows you to use causal masking based on the step
                of an environment instead of the step of each embedding,
                preventing attention to positions that are at a future
                state of the environment. B is the batch size, S1 is
                the sequence length of the first modality and S2 is the
                sequence length of the 2nd modality. Only Long type masks
                are supported.
        """
        inpt_list = [
          inpt+self.mode_encodings[i] for i,inpt in enumerate(inpt_list)
        ]
        inpts = torch.cat(inpt_list, dim=1) # (B,S1+S2,E)
        # cross mask assumes true is padding
        cross_mask = get_full_cross_mask(step_masks) # (B,S1+S2,S1+S2)
        pad_mask = torch.cat(pad_masks, dim=-1)# (B,S1+S2)
        pad_mask = padmask2attnmask(pad_mask) # (B,S1+S2,S1+S2)
        attn_mask = cross_mask|pad_mask
        return inpts, attn_mask

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

def print_tensor(t, n_tab=0):
    if len(t.shape)==2:
        el = t.tolist()
        for e in el:
            print("\t"*n_tab, e)
    else:
        for tt in t:
            print_tensor(tt, n_tab=n_tab+1)
            print()

if __name__=="__main__":
    for i in range(3):
        mask1 = np.arange(3)
        mask2 = np.repeat(mask1, (i+1,))
        print("Mask1:", mask1)
        print("Mask2:", mask2)
        cross_mask = get_full_cross_mask(
            [torch.LongTensor(mask1[None]), torch.LongTensor(mask2[None])]
        )
        print("Output:")
        print_tensor(cross_mask.float())
        print()
        print()
        mask1,mask2 = mask2,mask1
        print("Mask1:", mask1)
        print("Mask2:", mask2)
        cross_mask = get_full_cross_mask(
            [torch.LongTensor(mask1[None]), torch.LongTensor(mask2[None])]
        )
        print("Output:")
        print_tensor(cross_mask.float())
        print()
        print()

## Adapted directly from pytorch implementation
#class TransformerEncoderLayer(torch.nn.Module):
#    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
#    This standard encoder layer is based on the paper "Attention Is All You Need".
#    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#    in a different way during application.
#
#    TransformerEncoderLayer can handle either traditional torch.tensor inputs,
#    or Nested Tensor inputs.  Derived classes are expected to similarly accept
#    both input formats.  (Not all combinations of inputs are currently
#    supported by TransformerEncoderLayer while Nested Tensor is in prototype
#    state.)
#
#    If you are implementing a custom layer, you may derive it either from
#    the Module or TransformerEncoderLayer class.  If your custom layer
#    supports both torch.Tensors and Nested Tensors inputs, make its
#    implementation a derived class of TransformerEncoderLayer. If your custom
#    Layer supports only torch.Tensor inputs, derive its implementation from
#    Module.
#
#    Args:
#        d_model: the number of expected features in the input (required).
#        nhead: the number of heads in the multiheadattention models (required).
#        dim_feedforward: the dimension of the feedforward network model (default=2048).
#        dropout: the dropout value (default=0.1).
#        activation: the activation function of the intermediate layer, can be a string
#            ("relu" or "gelu") or a unary callable. Default: relu
#        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
#        batch_first: If ``True``, then the input and output tensors are provided
#            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
#        norm_first: if ``True``, layer norm is done prior to attention and feedforward
#            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
#        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
#            bias. Default: ``True``.
#
#    Examples::
#        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#        >>> src = torch.rand(10, 32, 512)
#        >>> out = encoder_layer(src)
#
#    Alternatively, when ``batch_first`` is ``True``:
#        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
#        >>> src = torch.rand(32, 10, 512)
#        >>> out = encoder_layer(src)
#
#    Fast path:
#        forward() will use a special optimized implementation described in
#        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
#        conditions are met:
#
#        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
#          argument ``requires_grad``
#        - training is disabled (using ``.eval()``)
#        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
#        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
#        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
#        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
#          nor ``src_key_padding_mask`` is passed
#        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
#          unless the caller has manually modified one without modifying the other)
#
#        If the optimized implementation is in use, a
#        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
#        passed for ``src`` to represent padding more efficiently than using a padding
#        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
#        returned, and an additional speedup proportional to the fraction of the input that
#        is padding can be expected.
#
#        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
#         https://arxiv.org/abs/2205.14135
#
#    """
#    __constants__ = ['norm_first']
#
#    def __init__(self,
#                 d_model: int,
#                 nhead: int,
#                 dim_feedforward: int = 2048,
#                 dropout: float = 0.1,
#                 activation=F.gelu,
#                 layer_norm_eps: float = 1e-5,
#                 batch_first: bool = False,
#                 norm_first: bool = False,
#                 bias: bool = True,
#                 add_zero_attn=False,
#                 device=None, dtype=None) -> None:
#        factory_kwargs = {'device': device, 'dtype': dtype}
#        super().__init__()
#        self.add_zero_attn = add_zero_attn
#        self.self_attn = nn.MultiheadAttention(
#            d_model, nhead, dropout=dropout,
#            bias=bias, batch_first=batch_first,
#            add_zero_attn=self.add_zero_attn,
#            **factory_kwargs
#        )
#        # Implementation of Feedforward model
#        self.linear1 = nn.Linear(
#            d_model, dim_feedforward, bias=bias, **factory_kwargs
#        )
#        self.dropout = nn.Dropout(dropout)
#        self.linear2 = nn.Linear(
#            dim_feedforward, d_model, bias=bias, **factory_kwargs
#        )
#
#        self.norm_first = norm_first
#        self.norm1 =    nn.LayerNorm(
#            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
#        )
#        self.norm2 =    nn.LayerNorm(
#            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
#        )
#        self.dropout1 = nn.Dropout(dropout)
#        self.dropout2 = nn.Dropout(dropout)
#
#        # We can't test self.activation in forward() in TorchScript,
#        # so stash some information about it instead.
#        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
#            self.activation_relu_or_gelu = 1
#        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
#            self.activation_relu_or_gelu = 2
#        else:
#            self.activation_relu_or_gelu = 0
#        self.activation = activation
#
#    def __setstate__(self, state):
#        super().__setstate__(state)
#        if not hasattr(self, 'activation'):
#            self.activation = F.relu
#
#
#    def forward(self,
#                src,
#                src_mask= None,
#                src_key_padding_mask= None,
#                is_causal: bool = False,
#                use_cache=False,
#                past_key_values=None,
#                ):
#        r"""
#        Pass the input through the encoder layer.
#
#        Args:
#            src: torch Tensor (B,S)
#                the sequence to the encoder layer (required).
#            src_mask: BoolTensor (B,S) or (B,S,S)
#                the mask for the src sequence (optional).
#            src_key_padding_mask:
#                the mask for the src keys per batch (optional).
#            is_causal: If specified, applies a causal mask as
#                ``src mask``.  Default: ``False``.
#                Warning:
#                ``is_causal`` provides a hint that ``src_mask`` is the
#                causal mask. Providing incorrect hints can result in
#                incorrect execution, including forward and backward
#                compatibility.
#
#        Shape:
#            see the docs in Transformer class.
#        """
#        src_key_padding_mask = F._canonical_mask(
#            mask=src_key_padding_mask,
#            mask_name="src_key_padding_mask",
#            other_type=F._none_or_dtype(src_mask),
#            other_name="src_mask",
#            target_type=src.dtype
#        )
#
#        src_mask = F._canonical_mask(
#            mask=src_mask,
#            mask_name="src_mask",
#            other_type=None,
#            other_name="",
#            target_type=src.dtype,
#            check_other=False,
#        )
#
#        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
#        why_not_sparsity_fast_path = ''
#        if not src.dim() == 3:
#            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
#        elif self.training:
#            why_not_sparsity_fast_path = "training is enabled"
#        elif not self.self_attn.batch_first :
#            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
#        elif not self.self_attn._qkv_same_embed_dim :
#            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
#        elif not self.activation_relu_or_gelu:
#            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
#        elif not (self.norm1.eps == self.norm2.eps):
#            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
#        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
#            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
#        elif self.self_attn.num_heads % 2 == 1:
#            why_not_sparsity_fast_path = "num_head is odd"
#        elif torch.is_autocast_enabled():
#            why_not_sparsity_fast_path = "autocast is enabled"
#        if not why_not_sparsity_fast_path:
#            tensor_args = (
#                src,
#                self.self_attn.in_proj_weight,
#                self.self_attn.in_proj_bias,
#                self.self_attn.out_proj.weight,
#                self.self_attn.out_proj.bias,
#                self.norm1.weight,
#                self.norm1.bias,
#                self.norm2.weight,
#                self.norm2.bias,
#                self.linear1.weight,
#                self.linear1.bias,
#                self.linear2.weight,
#                self.linear2.bias,
#            )
#
#            # We have to use list comprehensions below because TorchScript
#            # does not support generator expressions.
#            _supported_device_type = [
#                "cpu", "cuda",
#                torch.utils.backend_registration._privateuse1_backend_name
#            ]
#            if torch.overrides.has_torch_function(tensor_args):
#                s = "some Tensor argument has_torch_function"
#                why_not_sparsity_fast_path = s
#            elif not all((x.device.type in _supported_device_type) for
#                                                        x in tensor_args):
#                why_not_sparsity_fast_path = (
#                    "some Tensor argument's device is neither one of "
#                    f"{_supported_device_type}"
#                )
#            elif torch.is_grad_enabled() and any(x.requires_grad for
#                                                    x in tensor_args):
#                why_not_sparsity_fast_path = (
#                  "grad is enabled and at least one of query or the "
#                  "input/output projection weights or biases requires_grad"
#                )
#
#            if not why_not_sparsity_fast_path:
#                merged_mask, mask_type = self.self_attn.merge_masks(
#                    src_mask, src_key_padding_mask, src
#                )
#                return torch._transformer_encoder_layer_fwd(
#                    src,
#                    self.self_attn.embed_dim,
#                    self.self_attn.num_heads,
#                    self.self_attn.in_proj_weight,
#                    self.self_attn.in_proj_bias,
#                    self.self_attn.out_proj.weight,
#                    self.self_attn.out_proj.bias,
#                    self.activation_relu_or_gelu == 2,
#                    self.norm_first,
#                    self.norm1.eps,
#                    self.norm1.weight,
#                    self.norm1.bias,
#                    self.norm2.weight,
#                    self.norm2.bias,
#                    self.linear1.weight,
#                    self.linear1.bias,
#                    self.linear2.weight,
#                    self.linear2.bias,
#                    merged_mask,
#                    mask_type,
#                )
#
#
#        x = src
#        if self.norm_first:
#            x = x + self._sa_block(
#                self.norm1(x),
#                src_mask,
#                src_key_padding_mask,
#                is_causal=is_causal
#            )
#            x = x + self._ff_block(self.norm2(x))
#        else:
#            x = self.norm1(x + self._sa_block(
#                x, src_mask, src_key_padding_mask, is_causal=is_causal
#            ))
#            x = self.norm2(x + self._ff_block(x))
#
#        return x
#
#    # self-attention block
#    def _sa_block(self,
#                  q,
#                  k,
#                  v,
#                  attn_mask,
#                  key_padding_mask,
#                  is_causal: bool = False):
#        x = self.self_attn(q,k,v,
#                           attn_mask=attn_mask,
#                           key_padding_mask=key_padding_mask,
#                           need_weights=False, is_causal=is_causal)[0]
#        return self.dropout1(x)
#
#    # feed forward block
#    def _ff_block(self, x):
#        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#        return self.dropout2(x)   
#   
#   
