from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import dl_utils.torch_modules as tmods
from .utils import (
    generate_square_subsequent_mask, arglast, top_k_acc,
    update_shape, padmask2attnmask,
)
import math

from transformers import (
    CONFIG_MAPPING,
    GPT2Config,
    AutoModelForCausalLM,
    OpenAIGPTConfig,
    GPTJConfig,
    LlamaConfig,
    TransfoXLConfig,
)

DEVICES = {
    -1: "cpu", **{i:i for i in range(10)}
}

class SequenceModule(tmods.CoreModule):
    def __init__(self,
                n_tokens: int=None,
                out_tokens: int=None,
                d_model:int=128,
                n_heads:int=4,
                h_mult:int=4,
                h_norm:bool=False,
                n_layers:int=3,
                norm_first:bool=True,
                drop_p:float=0.5,
                max_posencs:int=1000,
                posenc_drop_p:float=None,
                learn_posencs:bool=False,
                actv_fxn:str="gelu",
                scale_attn_weights:bool=True,
                scale_by_inv_layer:bool=True,
                reorder_and_upcast:bool=False,
                hf_model_type:str="llama",
                pretrained:bool=False,
                init_range:float=0.1,
                *args, **kwargs):
        """
        n_tokens: int
            the number of tokens for the embedding layer. if None,
            no embedding layer is used.
        out_tokens: int
            Determines the size of the output predictions
        d_model: int
            the number of dimensions for the latent vectors
        n_heads: int
            the number of attention heads
        h_mult: int
            a multiplier to determine the hidden dimensionality of the
            feed forward networks in the model.
        h_norm: bool
            if true, will use a layer norm on the lstm hidden state
        n_layers: int
            the number of transformer layers
        norm_first: bool
            if true, applies layer norm before the operations in the
            encoder layer (this seemed to be better in some paper I
            can't remember the name of)
        drop_p: float
            the dropout probability. 0 means no dropout.
        max_posencs: int
            the number of possible embeddings. If
        posenc_drop_p: float optional
            the dropout probability for positional encodings. 0 means
            no dropout. defaults to drop_p if none
        learn_posencs: bool
            determines whether or not gradients are backpropagated into
            the positional encodings.
        actv_fxn: str
            the transformer activation function
        hf_model_type: str
            the huggingface transformer base. only applies if using
            HFModel types. Specifies the hf model base type.
        scale_attn_weights: bool
            scale attention weights by dividing by sqrt(hidden_size)
        scale_by_inv_layer: bool
            scale attention weights by inverse layer index. see
            huggingface docs for details
        reorder_and_upcast: bool
            reorder and upcast attention. see huggingface docs for details
        pretrained: bool
            if true, will ignore model specs and use a pretrained
            huggingface model. only applies if using HF model types.
        init_range: float
            a lower and upper bound for uniform weight sampling for
            the embeddings and output dense layer.
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.out_tokens = out_tokens
        if self.out_tokens is None: self.out_tokens = self.n_tokens
        self.d_model = d_model
        self.n_heads = n_heads
        self.h_mult = h_mult
        self.h_norm = h_norm
        self.n_layers = n_layers
        self.drop_p = drop_p
        self.norm_first = norm_first
        self.max_posencs = max_posencs
        self.posenc_drop_p = posenc_drop_p
        if self.posenc_drop_p is None:
            self.posenc_drop_p = drop_p
        self.learn_posencs = learn_posencs
        self.actv_fxn = actv_fxn
        self.hf_model_type = hf_model_type
        self.scale_attn_weights = scale_attn_weights
        self.scale_by_inv_layer = scale_by_inv_layer
        self.reorder_and_upcast = reorder_and_upcast
        self.pretrained = pretrained
        self.init_range = init_range

    def init_weights(self, init_range=0.1) -> None:
        print("Weight initialization currently not implemented")
        pass

    def forward(self, src:torch.Tensor,
                      mask:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      is_causal:bool=None,
                      tforce:bool=True,
                      n_steps:int=10,
                      temperature=None,
                      *args, **kwargs):
        """
        Arguments:
            src: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
                an optional attention mask. if transformer uses auto-
                regressive prediction, simply mark `is_causal` to true
                and leave this field equal to None.
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            is_causal: bool
                If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product
                attention.
            tforce: bool
                determines whether or not to teacherforce
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
        Returns:
            if tforce:
              Tensor ``[B, S, N_Tokens]``
            else:
              Tensor ``[B, S+n_steps, N_Tokens]``
        """
        if tforce:
            ret_dict = self.tforce_fwd(
                src=src,
                mask=mask,
                pad_mask=pad_mask,
                is_causal=is_causal
            )
        else:
            ret_dict = self.freedom_fwd(
                src=src,
                mask=mask,
                pad_mask=pad_mask,
                is_causal=is_causal,
                n_steps=n_steps,
                temperature=temperature,
            )
        return ret_dict

class VisionModule(tmods.CoreModule):
    """
    This is the base class for vision modules.
    """
    def __init__(self,
                 inpt_shape,
                 outp_size,
                 bnorm=False,
                 lnorm=False,
                 drop_p=0,
                 actv_fxn="ReLU",
                 depths=[32, 48],
                 kernels=[3, 3],
                 strides=[4, 1],
                 paddings=[0, 0],
                 groups=1,
                 *args, **kwargs):
        """
        Args: 
            inpt_shape: tuple or listlike (..., C, H, W)
                the shape of the input
            outp_size: int
                the size of the final output vector
            bnorm: bool
                if true, the model uses batch normalization
            lnorm: bool
                if true, the model uses layer normalization on the h
                and c recurrent vectors after the recurrent cell
            drop_p: float
                the probability of zeroing a neuron within the dense
                layers of the network.
            actv_fxn: str
                the name of the activation function for each layer
            depths: tuple of ints
                the depth of each layer of the conv net
            kernels: tuple of ints
                the kernel size of each layer of the conv net
            strides: tuple of ints
                the stride of each layer of the conv net
            paddings: tuple of ints
                the padding of each layer of the conv net
            groups: int or tuple of ints
                the number of convolutional groups at each layer of the
                fully connected
                ouput networks
        """
        super().__init__()
        self.inpt_shape = inpt_shape
        self.outp_size = outp_size
        self.bnorm = bnorm
        self.lnorm = lnorm
        self.drop_p = drop_p
        self.actv_fxn = actv_fxn
        self.depths = [self.inpt_shape[-3], *depths]
        self.kernels = kernels
        if isinstance(kernels, int):
            self.kernels = [kernels for i in range(len(depths))]
        self.strides = strides
        if isinstance(strides, int):
            self.strides = [strides for i in range(len(depths))]
        self.paddings = paddings
        if isinstance(paddings, int):
            self.paddings = [paddings for i in range(len(depths))]
        self.groups = groups
        if isinstance(groups, int):
            self.groups = [groups for i in range(len(depths))]
        

class RawVision(VisionModule):
    """
    This vision module feeds the visual input directly, without
    preprocessing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = [ *self.inpt_shape[-3:] ]
        self.flat_size = int(np.prod(self.inpt_shape[-3:]))
        self.features = tmods.NullOp()

    def step(self, x, *args, **kwargs):
        return x

    def forward(self, x, *args, **kwargs):
        return x.reshape(len(x), -1)

class CNN(VisionModule):
    """
    A simple convolutional network
        conv2d
        bnorm/lnorm
        relu
        dropout
        repeat xN
        linear
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        modules = []
        shape = [*self.inpt_shape[-3:]]
        self.shapes = [shape]
        groups = 1
        for i in range(len(self.depths)-1):
            # CONV
            modules.append(
                nn.Conv2d(
                    self.depths[i],
                    self.depths[i+1],
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=self.paddings[i],
                    groups=max(int(groups*(i>0)), 1)
                )
            )
            # RELU
            modules.append(globals()[self.actv_fxn]())
            # Batch Norm
            if self.bnorm:
                modules.append(nn.BatchNorm2d(self.depths[i+1]))
            # Track Activation Shape Change
            shape = update_shape(
                shape, 
                depth=self.depths[i+1],
                kernel=self.kernels[i],
                stride=self.strides[i],
                padding=self.paddings[i]
            )
            self.shapes.append(shape)
        self.features = nn.Sequential(*modules)

        self.flat_size = int(np.prod(shape))
        self.projection = nn.Linear(self.flat_size, self.outp_size)

    def forward(self, x, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            pred: torch Float Tensor (B, K)
        """
        fx = self.features(x)
        fx = fx.reshape(len(fx), -1)
        return self.projection(fx)

class LSTM(SequenceModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'LSTM'

        if self.n_tokens:
            self.embeddings = torch.nn.Embedding(
                self.n_tokens, self.d_model
            )
        self.pre_norm = torch.nn.LayerNorm(self.d_model)
        self.lstms = torch.nn.ModuleList([])
        self.layer_norms = torch.nn.ModuleList([])
        self.h_norms = torch.nn.ModuleList([])
        for i in range(self.n_layers):
            self.layer_norms.append( torch.nn.LayerNorm(self.d_model) )
            self.h_norms.append( torch.nn.LayerNorm(self.d_model) )
            self.lstms.append(
                torch.nn.LSTMCell(self.d_model, self.d_model)
            )
        d_hid = self.d_model*4
        modules = []
        modules.append(torch.nn.Linear( self.d_model, d_hid ))
        modules.append(torch.nn.GELU())
        modules.append(torch.nn.LayerNorm(d_hid))
        modules.append(torch.nn.Dropout(self.drop_p))
        modules.append(torch.nn.Linear( d_hid, self.out_tokens ))
                       
        self.decoder = torch.nn.Sequential( *modules )

    def get_fresh_recurrent_vectors(self, batch_size=1):
        """
        Args:
            batch_size: int
        Returns:
            hs: list of tensors with shape (B, H)
                the length of the list is determined by the number of lstms
                in the model.
            cs: list of tensors with shape (B, H)
                the length of the list is determined by the number of lstms
                in the model.
        """
        n = self.n_layers
        hs = [torch.zeros(batch_size,self.d_model) for _ in range(n)]
        cs = [torch.zeros(batch_size,self.d_model) for _ in range(n)]
        d = self.get_device()
        return [h.to(d) for h in hs], [c.to(d) for c in cs]
    
    def step(self,
             src=None,
             pad_mask=None,
             hs=None,
             cs=None,
             temperature=None,
             prev_logits=None,
             inputs_embeds=None,
             *args, **kwargs):
        """
        Arguments:
            src: Tensor, shape ``[bsize]``
                if None, inputs_embeds must be not None
            pad_mask: BoolTensor, shape ``[bsize]``
                1s/Trues denote padding, 0s/Falses denote not padding
            hs: list of Tensors with shape (B, D)
                a list of h vectors for each lstm
            cs: list of Tensors with shape (B, D)
                a list of c vectors for each lstm
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            prev_logits: torch Tensor [bsize, n_tokens] or None
                optionally argue the previous logits as a vector to
                contain the most recent predictions
            inputs_embeds: None or Tensor, shape (B,D)
                optionally argue the embeddings directly instead of
                token ids.
        Returns:
            dict:
                logits: Tensor of shape (B, N)
                pred_ids: Tensor of shape (B,)
                hs: list of Tensors with shape (B, D)
                    a list of updated h vectors for each lstm
                cs: list of Tensors with shape (B, D)
                    a list of updated c vectors for each lstm
        """
        if src is None:
            B = inputs_embeds.shape[0]
            pred_ids = torch.zeros(B,device=self.get_device()).long()
        else:
            B = src.shape[0]
            pred_ids = src.detach().data.clone()

        if pad_mask is None:
            idx = torch.zeros(B, device=self.get_device()).bool()
        else:
            idx = ~pad_mask.bool()

        if prev_logits is None:
            # will be used to store the predicted logits
            logits = torch.zeros(B,self.out_tokens).to(self.get_device())
        else:
            logits = prev_logits.detach().data.clone()

        if inputs_embeds is None:
            inpt = self.embeddings(src)[idx]
        else: inpt = inputs_embeds[idx]
        
        new_hs = [ h.clone() for h in hs ]
        new_cs = [ c.clone() for c in cs ]
        inpt = self.pre_norm(inpt)
        if len(inpt)>0:
            # Loop through lstm layers of model
            z = zip(self.lstms, self.layer_norms, self.h_norms)
            for l,(lstm, norm, h_norm) in enumerate(z):
                h,c = (hs[l][idx], cs[l][idx])
                if self.h_norm: h = h_norm(h)
                h,c = lstm(inpt, (h,c))
                inpt = norm(h)
                new_hs[l][idx] = h
                new_cs[l][idx] = c
            logits[idx] = self.decoder(inpt)
            pred_ids[idx] = self.sample_with_temperature(
                logits[idx], temperature
            )
        return {
            "logits":   logits,
            "pred_ids": pred_ids,
            "hs": new_hs, 
            "cs": new_cs,
        }

    def forward(self, src:torch.Tensor,
                      pad_mask:torch.Tensor=None,
                      n_steps:int=0,
                      temperature=None,
                      inputs_embeds=None,
                      *args, **kwargs):
        """
        Arguments:
            src: Tensor, shape ``[bsize, seq_len]``
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            inputs_embeds: None or Tensor, shape (B,S,D)
                optionally argue the embeddings directly instead of
                token ids.
        Returns:
            ret_dict: dict
                logits: Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
                pred_ids: Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
        """
        if not inputs_embeds:
            embs = self.embeddings(src)
        else: embs = inputs_embeds

        B,S,D = embs.shape
        logits = []
        pred_ids = []
        hs,cs = self.get_fresh_recurrent_vectors(B)
        
        # Loop through sequence
        for step in range(S+n_steps):
            if step<embs.shape[1]:
                pmask = pad_mask[:,step]
                inpt = embs[:,step]
            else:
                inpt = self.embeddings(pred_ids[-1])
            ret_dict = self.step(
                inputs_embeds=inpt,
                pad_mask=pmask,
                hs=hs, cs=cs,
                temperature=temperature
            )
            hs,cs = ret_dict["hs"], ret_dict["cs"]
            logits.append(ret_dict["logits"])
            if step<S-1: pred_ids.append(src[:,step+1])
            else: pred_ids.append(ret_dict["pred_ids"])
        return {
            "logits": torch.stack(logits, dim=1),
            "pred_ids": torch.stack(pred_ids,dim=1),
            "hs": hs, "cs": cs
        }


class HFTransformer(SequenceModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'Transformer'

        if self.pretrained:
            # We will load the weights and then transfer them to our
            # custom model type.
            temp_model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_type
            )
            self.encoder = tmods.FlexibleLlamaModel(
                temp_model.config
            )
            self.encoder.load_state_dict(temp_model.state_dict())

            print("Properties:")
            for name in dir(self.encoder):
                print(name)
            print(self.encoder)
            d = self.encoder.get_input_embeddings().weight.shape[-1]
            self.d_model = d
        else:
            config = self.get_config()
            self.encoder = tmods.FlexibleLlamaModel( config )
            if hasattr(self.encoder, "transformer"):
                if hasattr(self.encoder.transformer, "wpe"):
                    wpe = self.encoder.transformer.wpe
                    for name, p in wpe.named_parameters():
                        p.requires_grad = self.learn_posencs

        if self.n_tokens:
            self.embeddings = self.encoder.get_input_embeddings()
        else:
            self.encoder.set_input_embeddings(None)
            self.embeddings = None

        if hasattr(self.encoder, "lm_head"):
            if self.n_tokens and self.n_tokens==self.out_tokens:
                self.decoder = self.encoder.lm_head
            else:
                self.decoder =  nn.Linear( self.d_model, self.out_tokens )
            self.encoder.lm_head = self.decoder
        else:
            self.decoder =  nn.Linear( self.d_model, self.out_tokens )
        self.init_weights()

    def get_config(self):
        """
        Finds the appropirate configuration when using Huggingface
        models.
        """
        n_tokens = self.n_tokens if self.n_tokens else self.out_tokens
        d_hid = self.h_mult*self.d_model
        config_kwargs = {
            "vocab_size": n_tokens,
            "hidden_size": self.d_model,
            "intermediate_size": d_hid,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_heads,
            "hidden_act": self.actv_fxn,
            "n_positions": self.max_posencs,
            "rotary_dim": self.d_model//self.n_heads,
            "rope_theta": self.d_model//self.n_heads,
            "n_ctx": self.max_posencs,
            "n_embd": self.d_model,
            "n_head": self.n_heads,
            "n_inner": d_hid,
            "activation_function": self.actv_fxn,
            "resid_pdrop": self.drop_p,
            "embd_pdrop":  0,
            "attn_pdrop":  self.drop_p,
            "scale_attn_weights": self.scale_attn_weights,
            "scale_attn_by_inverse_layer_idx": self.scale_by_inv_layer,
            "tie_word_embeddings": False,
            "torch_dtype": "float32",
            "reorder_and_upcast_attn": self.reorder_and_upcast,
            "add_cross_attention": False,
        }
        if self.hf_model_type=="gpt2":
            config = GPT2Config()
        elif self.hf_model_type == "gptj":
            config = GPTJConfig()
        elif self.hf_model_type == "llama":
            config = LlamaConfig()
        elif self.hf_model_type == "transxl":
            config = TransfoXLConfig()
        config.update(config_kwargs)
        return config

    def tforce_fwd(self,
                   src:torch.Tensor,
                   mask:torch.Tensor=None,
                   pad_mask:torch.Tensor=None,
                   inputs_embeds:torch.Tensor=None,
                   *args, **kwargs):
        """
        Arguments:
            src: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
                true means unattended locations
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            inputs_embeds: tensor (B,S,E)
                optionally argue embeddings instead of token ids
        Returns:
            output Tensor of shape ``[bsize, seq_len, n_tokens]``
        """
        # flipped so that true means attend to
        attn_mask = ~(pad_mask.bool())
        if mask is not None:
            attn_mask = padmask2attnmask(attn_mask)
            attn_mask = attn_mask&~mask
        output = self.encoder(
            src,
            attention_mask=attn_mask,
            inputs_embeds=inputs_embeds,
        )
        if not hasattr(output, "logits"):
            og_shape = output.last_hidden_state.shape
            inpts = output.last_hidden_state.reshape(-1,og_shape[-1])
            logits = self.decoder(inpts).reshape(*og_shape[:-1], -1)
        else: logits = output.logits
        return {
            "logits":logits,
            "pred_ids":torch.argmax(logits,dim=-1)
        }

    def freedom_fwd(self,
                    src:torch.Tensor,
                    mask:torch.Tensor=None,
                    pad_mask:torch.Tensor=None,
                    is_causal:bool=None,
                    n_steps:int=10,
                    incl_all_inpts:bool=False,
                    pad_pos_skip:bool=False,
                    temperature=None,
                    inputs_embeds=None,
                    past_key_values=None,
                    *args, **kwargs):
        """
        Arguments:
            src: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
                true means unattended locations
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            is_causal: bool
                If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product
                attention.
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            incl_all_inpts: bool
                if true, will include all input tokens in the output
                prediction tensor. otherwise only includes "predicted
                spaces". "predicted spaces" includes the shifted initial
                inputs. This is useful to save a concatenation during
                the data bootstrapping phase.
            pad_pos_skip: bool
                if true, will skip over masked tokens when applying
                positional encodings based on the pad mask. True values
                in the mask will be skipped.
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            inputs_embeds: tensor (B,S,E)
                optionally argue embeddings instead of token ids
            past_key_values: tuple of tuple of tensors
                the output of a huggingface cache. used to speed up
                computations. See huggingface documentation for more
                details
        Returns:
            output Tensor of shape ``[bsize, seq_len+n_steps, n_tokens]``
        """
        n_loops = n_steps + 1
        if src is None:
            B,S = inputs_embeds.shape[:2]
        else:
            B,S = src.shape

        pad_mask = torch.nn.functional.pad(
            ~(pad_mask.bool()), (0, n_loops), value=True
        )
        if mask is not None:
            mask = torch.nn.functional.pad(
                ~(mask.bool()), (0, n_loops, 0, n_loops), value=True
            )
        pred_ids = torch.zeros(
            (B,S+n_loops), device=self.get_device()
        ).long()
        if src is not None:
            pred_ids[:,:S] = src
        logits = torch.zeros(
            (B,S+n_steps+incl_all_inpts,self.n_tokens),
            device=self.get_device()
        )
        logits[:,:S-1+incl_all_inpts].scatter_(
            dim=-1,
            index=src[:, 1-incl_all_inpts:S, None],
            src=torch.ones_like(logits[:, :S-1+incl_all_inpts])
        )

        # Need to ensure we use the appropriate input type between
        # the src ids and the input embeddings
        if src is None:
            inpt_emb = inputs_embeds
            inpt = None
        else:
            inpt = pred_ids[:,:S]
            inpt_emb = None

        # Need to ensure the padding mask is the full length of the
        # past_key_values if past_key_values is not none
        p_end = S
        if past_key_values is not None:
            p_end = past_key_values[0][0].shape[1]
        attn_mask = pad_mask[:,:p_end]
        if mask is not None:
            attn_mask = padmask2attnmask(attn_mask)
            attn_mask = mask[:,:p_end,:p_end]&attn_mask

        for step in range(n_loops):
            output = self.encoder(
                input_ids=inpt,
                attention_mask=attn_mask,
                use_cache=True,
                past_key_values=past_key_values,
                inputs_embeds=inpt_emb,
            )
            past_key_values = output.past_key_values
            ## TODO: change FlexibleLlama model to output logits
            if not hasattr(output, "logits"):
                inpts = output.last_hidden_state[:,-1]
                pred = self.decoder(inpts)
            else: pred = output.logits[:,-1]
            logits[:,S-1+step+incl_all_inpts] = pred
            argmaxs = self.sample_with_temperature(
                pred, temperature
            ).squeeze()
            pred_ids[:,S+step] = argmaxs
            if step < n_steps:
                inpt_emb = None
                inpt = pred_ids[:,S+step:S+step+1]
                attn_mask = pad_mask[:,:p_end+step+1]
                if mask is not None:
                    attn_mask = padmask2attnmask(attn_mask)
                    e = p_end+step+1
                    attn_mask = mask[:,:e,:e]&attn_mask
        return {
            "logits": logits,
            "preds":  pred_ids[:,int(not incl_all_inpts):],
            "past_key_values": past_key_values,
        }


class IdentityPositionalEncoding(nn.Module):
    def __init__(self,
                 drop_p:float=0.1,
                 *args, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = self.dropout( x )
        return x

class PositionalEncoding(nn.Module):
    def __init__(self,
                 posenc_drop_p:float=0,
                 drop_p:float=0.1,
                 max_len:int=1000):
        super().__init__()
        self.posenc_dropout = nn.Dropout(p=posenc_drop_p)
        self.dropout = nn.Dropout(p=drop_p)
        self.arange = np.arange(max_len).astype("int")

    def rand_forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        n = np.random.randint(x.size(1), self.pe.shape[0]+1)
        idxs = torch.sort(torch.randperm(n)[:x.size(1)]).values.long()
        x = self.dropout( x + self.posenc_dropout(self.pe[idxs]) )
        return x

    def skip_rand_forward(
            self,
            x: Tensor,
            mask: Tensor,
            *args,
            **kwargs
        ) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            mask: Tensor, shape ``[batch_size, seq_len]``
                pad mask. true values represent padding/blotching
        """
        if mask is None: return self.rand_forward(x)
        # pe: N, E
        n = np.random.randint(x.size(1), self.pe.shape[0]+1)
        idxs = torch.sort(torch.randperm(n)[:x.size(1)]).values.long()
        pe = self.posenc_dropout(self.pe[idxs])

        sums = (~mask).float().sum(-1)
        idxs = torch.cat([torch.arange(s) for s in sums], axis=0).long()
        fx = torch.zeros_like(x)
        fx[~mask] += pe[idxs]
        fx = x + fx

        return self.dropout( fx )

    def vanil_forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = self.dropout( x + self.posenc_dropout(self.pe[:x.size(1)]) )
        return x

    def skip_vanil_forward(
            self,
            x: Tensor,
            mask: Tensor,
            *args,
            **kwargs
        ) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            mask: Tensor, shape ``[batch_size, seq_len]``
                pad mask. true values represent padding/blotching
        """
        if mask is None: return self.vanil_forward(x)
        pe = self.posenc_dropout(self.pe[:x.size(1)])

        sums = torch.sum((~mask).float(), -1)
        idxs = torch.cat([torch.arange(s) for s in sums], axis=0).long()
        fx = torch.zeros_like(x)
        fx[~mask] += pe[idxs]
        fx = x + fx

        return self.dropout( fx )

class RandPositionalEncoding(PositionalEncoding):
    def __init__(self,
                 d_model:int,
                 posenc_drop_p:float=0,
                 drop_p:float=0.1,
                 max_len:int=1000,
                 learnable:bool=False,
                 pad_pos_skip:bool=False):
        super().__init__(posenc_drop_p, drop_p, max_len=max_len)
        self.pad_pos_skip = pad_pos_skip

        pe = 0.1*math.sqrt(max_len/d_model)*torch.randn(max_len,d_model)
        if learnable: self.pe = torch.nn.Parameter(pe)
        else: self.register_buffer('pe', pe)

        if pad_pos_skip:
            self.forward = self.skip_rand_forward
        else:
            self.forward = self.rand_forward

class SinPositionalEncoding(PositionalEncoding):
    def __init__(self,
                 d_model:int,
                 posenc_drop_p:float=0,
                 drop_p:float=0.1,
                 max_len:int=1000,
                 learnable:bool=False,
                 pad_pos_skip:bool=False):
        super().__init__(posenc_drop_p, drop_p, max_len=max_len)
        self.pad_pos_skip = pad_pos_skip

        position = torch.arange(max_len).unsqueeze(1)
        scale = (-math.log(10000.0) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * scale)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if learnable: self.pe = torch.nn.Parameter(pe)
        else: self.register_buffer('pe', pe)

        if pad_pos_skip:
            self.forward = self.skip_vanil_forward
        else:
            self.forward = self.vanil_forward


class RandSinPositionalEncoding(SinPositionalEncoding):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pad_pos_skip:
            self.forward = self.skip_rand_forward
        else:
            self.forward = self.rand_forward


class LossWrapper(torch.nn.Module):
    """
    This class wraps the model to keep the loss calculations distributed
    on all GPUs. Otherwise one gpu is overloaded with computational
    costs.
    """
    def __init__(self,
                 model,
                 config,
                 pad_id=0,
                 bos_id=1,
                 eos_id=2,
                 tokenizer=None,
                 loss_fxn=torch.nn.functional.cross_entropy,
                 *args, **kwargs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer:
            pad_id = getattr(self.tokenizer, "pad_id", pad_id)
            bos_id = getattr(self.tokenizer, "bos_id", bos_id)
            eos_id = getattr(self.tokenizer, "eos_id", eos_id)
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.config = config
        self.label_smoothing = self.config.get("label_smoothing", 0)
        self.loss_scale = 1./self.config.get("n_grad_loops",1)
        self.loss_fxn = loss_fxn

    def forward(self,
                data,
                tforce=True,
                no_grad=False,
                temperature=None,
                top_k=5,
                reduce_metrics=True,
                *args, **kwargs):
        """
        Args:
            data: dict
                "input_ids": LongTensor (B,S1)
                    the token indices of the input sequence. The CMP
                    token should be appended to the end of each sentence.
                "input_pad_mask": BoolTensor (B,S1)
                    attention mask for padding purposes. trues mean
                    padding.
                "output_ids": LongTensor (B,S2)
                    the token indices of the target sequence. An EOS
                    token should be appended to the end of each sentence
                "output_pad_mask": BoolTensor (B,S1)
                    attention mask for padding purposes. trues mean
                    padding.
            ret_preds: bool
                if true, will return the predictions
            tforce: bool
                determines whether model should use teacher forcing for
                predictions or not.
            incl_intl_prob: bool
                if true, will include the initial problem in the loss.
                if false, will exclude initial problem from the loss.
            temperature: float
                a temperature parameter for softmax sampling. Set to
                low number for high confidence sampling, high value
                for low confidence sampling
            no_grad: bool
                if true, this function will not call .backward() on
                the loss. If false, this function will still only call
                .backward if in training mode.
            top_k: int optional
                if argued, returns a calculation of the top_k accuracy
            reduce_metrics: bool
                if true, loss and acc will be averaged over all samples.
                if false, loss and acc will be returned as tensors for
                each token prediction
        Returns:
            ret_dict: dict (keys: str, vals: torch tensor)
                "loss": torch tensor (1,) or (B,)
                "acc": torch tensor (1,) or (B,)
                    the raw accuracy for the non-rmb task
                "preds": torch tensor (B,S,P)
                    the prediction logits. only returned if ret_preds is
                    true
        """
        ret_dict = dict()
        pad_id = self.pad_id
        bos_id = self.bos_id
        eos_id = self.eos_id
        if "input_pad_mask" not in data:
            inpt_pad_mask = (data["input_ids"]==pad_id)
            inpt_pad_mask = inpt_pad_mask|(data["input_ids"]==eos_id)
        else: inpt_pad_mask = data["input_pad_mask"].clone()
        if "output_pad_mask" not in data:
            out_pad_mask = data["output_ids"]==pad_id
            out_pad_mask = out_pad_mask==bos_id
        else: out_pad_mask = data["output_pad_mask"].clone()

        # TODO: figure out more modular way to do this. We do this
        # because for this task, we need some seed input for the
        # model to begin predicting from. Also for LSTMs,
        # the training seems to be more robust without teacher forcing.
        inpts = data["input_ids"]
        if not tforce or type(self.model)==LSTM:
            leading_pad = torch.max(torch.argmax(
                (~inpt_pad_mask).long(), dim=1
            ))
            inpts = inpts[:,:leading_pad+3]

        tot_len = data["output_ids"].shape[-1]-inpts.shape[-1]
        ret_dict = self.model(
            inpts,
            pad_mask=inpt_pad_mask,
            tforce=tforce,
            n_steps=tot_len,
            temperature=temperature,
        )

        ## Loss
        #################################
        out_ids = data["output_ids"]
        inpt_mask = ~inpt_pad_mask.reshape(-1)
        out_mask =  ~out_pad_mask.reshape(-1)
        logits = ret_dict["logits"]
        ps = logits.reshape(
            -1, logits.shape[-1]
        )[inpt_mask]
        labels = out_ids.reshape(-1)[out_mask]
        try:
            loss = self.loss_scale*self.loss_fxn(
                ps,labels,
                reduction="none",
                label_smoothing=self.label_smoothing
            )
        except:
            self.print_data(
              data,inpt_pad_mask=inpt_pad_mask,out_pad_mask=out_pad_mask
            )
            assert False
        if not reduce_metrics:
            temp = torch.zeros_like(out_ids).float()
            temp[out_mask.reshape(out_ids.shape)] = loss
            loss = temp
        else:
            loss = loss.mean()
        ret_dict["loss"] = loss

        ## Acc
        #################################
        if "pred_ids" in ret_dict:
            pred_ids = ret_dict["pred_ids"]
            pred_ids = pred_ids.reshape(-1)[inpt_mask]
        else:
            pred_ids = torch.argmax(ps, dim=-1)
            ret_dict["pred_ids"] = torch.argmax(logits, dim=-1)
        acc = (pred_ids==labels).float()
        if reduce_metrics: acc = acc.mean()
        else: 
            temp = torch.zeros_like(out_ids).float()
            temp[out_mask.reshape(out_ids.shape)] = acc.long()
            acc = temp
        ret_dict["acc"] = acc

        ret_dict["top_k"] = top_k_acc(
            logits, out_ids, top_k, as_tensor=True
        )
        return ret_dict
    
    def print_data(self, data, inpt_pad_mask, out_pad_mask):
        if not self.tokenizer: self.tokenizer = EmptyTokenizer()
        for i in range(len(data["input_ids"])):
            print()
            print("Full inpt:",
              self.tokenizer.decode(data["input_ids"][i]))
            print("Full Outpt:",
              self.tokenizer.decode(data["output_ids"][i]))
            print("dropped inpt:",
              self.tokenizer.decode(
                data["input_ids"][i][inpt_pad_mask[i]]))
            print("dropped out:",
              self.tokenizer.decode(
                data["output_ids"][i][out_pad_mask[i]]))
            print("post inpt:",
              self.tokenizer.decode(
                data["input_ids"][i][~inpt_pad_mask[i]]))
            print("post out:",
              self.tokenizer.decode(
                data["output_ids"][i][~out_pad_mask[i]]))

        idx = inpt_pad_mask.float().sum(-1)!=out_pad_mask.float().sum(-1)
        print()
        print()
        print()
        print()
        for i in range(idx.long().sum(-1)):
            print("Full inpt:",
              self.tokenizer.decode(data["input_ids"][idx][i]))
            print("Full Outpt:",
              self.tokenizer.decode(data["output_ids"][idx][i]))
            print("dropped inpt:",
              self.tokenizer.decode(
                data["input_ids"][idx][i][inpt_pad_mask[idx][i]]))
            print("dropped out:",
              self.tokenizer.decode(
                data["output_ids"][idx][i][out_pad_mask[idx][i]]))
            print("post inpt:",
              self.tokenizer.decode(
                data["input_ids"][idx][i][~inpt_pad_mask[idx][i]]))
            print("post out:",
              self.tokenizer.decode(
                data["output_ids"][idx][i][~out_pad_mask[idx][i]]))

class EmptyTokenizer:
    def __init__(self): pass
    def decode(self, x):
        return x

def make_model(config):
    model = globals()[config.get("model_type","HFTransformer")](**config)
    return LossWrapper(model=model, config=config, **config)

