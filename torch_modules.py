import math
import torch.nn as nn
import torch

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

