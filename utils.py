import numpy as np
import torch
import os
try:
    import cv2
except:
    pass

def try_key(d, key, val):
    """
    d: dict
    key: str
    val: object
        the default value if the key does not exist in d
    """
    if key in d:
        return d[key]
    return val

def resize2Square(img, size):
    """
    resizes image to a square with the argued size. Preserves the aspect
    ratio.

    img: ndarray (H,W, optional C)
    size: int
    """
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: 
        return cv2.resize(img, (size, size), cv2.INTER_AREA)
    if h > w: 
        dif = h
    else:
        dif = w
    interpolation = cv2.INTER_AREA if dif > size else\
                    cv2.INTER_CUBIC
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
      mask = np.zeros((dif, dif), dtype=img.dtype)
      mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
      mask = np.zeros((dif, dif, c), dtype=img.dtype)
      mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)

def rand_sample(arr, n_samples=1):
    """
    Randomly samples a single element from the argued array.

    arr: sequence of some sort
    """
    if not isinstance(arr,list): arr = list(arr)
    if len(arr) == 0: print("len 0:", arr)
    samples = []
    perm = np.random.permutation(len(arr))
    for i in range(n_samples):
        samples.append(arr[perm[i]])
    if len(samples) == 1: return samples[0]
    return samples

def get_max_key(d):
    """
    Returns key corresponding to maxium value

    d: dict
        keys: object
        vals: int or float
    """
    max_v = -np.inf
    max_k = None
    for k,v in d.items():
        if v > max_v:
            max_v = v
            max_k = k
    return max_k

def update_shape(shape, kernel=3, padding=0, stride=1, op="conv"):
    """
    Calculates the new shape of the tensor following a convolution or
    deconvolution

    shape: list-like or int
        the height/width of the activations
    kernel: int or list-like
        size of the kernel
    padding: list-like or int
    stride: list-like or int
    op: str
        'conv' or 'deconv'
    """
    if type(shape) == type(int()):
        shape = np.asarray([shape])
    else:
        shape = np.asarray(shape)
    if type(kernel) == type(int()):
        kernel = np.asarray([kernel for i in range(len(shape))])
    else:
        kernel = np.asarray(kernel)
    if type(padding) == type(int()):
        padding = np.asarray([padding for i in range(len(shape))])
    else:
        padding = np.asarray(padding)
    if type(stride) == type(int()):
        stride = np.asarray([stride for i in range(len(shape))])
    else:
        stride = np.asarray(stride)

    if op == "conv":
        shape = (shape - kernel + 2*padding)/stride + 1
    elif op == "deconv" or op == "conv_transpose":
        shape = (shape - 1)*stride + kernel - 2*padding
    if len(shape) == 1:
        return int(shape[0])
    return [int(s) for s in shape]

def top_k_acc(preds, labels, k=5, as_tensor=False):
    """
    Returns the top_n accuracy for the argued predictions and labels

    Args:
        preds: torch float tensor (B, L)
            the logits or probabilities
        labels: torch long tensor (B,)
            the correct labels
        k: int
            the k to use for top k
        as_tensor: bool
            if true, returns result as a tensor
    Returns:
        top_n: float or tensor

    """
    ps = preds.reshape(-1,preds.shape[-1])
    args = torch.topk(ps,k,largest=True,sorted=False,dim=-1).indices
    acc = (args==labels.reshape(-1)[:,None]).float().sum(-1).mean()
    if as_tensor:
        return acc
    return acc.item()

def pad_list(arr, tot_len, fill_val=0, side="right"):
    """
    Pads the argued list to the goal length. Operates in place.

    Args:
        arr: list
        tot_len: int
            the length of the resulting array
        fill_val: object
            the value to use for the padding
        side: str
            pad on the left side or the right
    Returns:
        arr: list
            the padded list
    """
    n_pad = tot_len - len(arr)
    if n_pad<=0: return arr
    if side=="right":
        for _ in range(n_pad):
            arr.append(fill_val)
    else:
        padding = [fill_val for _ in range(n_pad)]
        arr = padding + arr
    return arr

def pad_to(arr, tot_len, fill_val=0, side="right", dim=-1):
    """
    Pads the argued list to the goal length along a single dimension.

    Args:
        arr: numpy or torch tensor
        tot_len: int
            the length of the resulting array
        fill_val: object
            the value to use for the padding
        side: str
            pad on the left side or the right
    Returns:
        arr: list
            the padded list
    """
    if dim<0: dim = len(arr.shape) + dim
    n_pad = tot_len - arr.shape[dim]
    if n_pad<=0: return arr
    tup = (0,n_pad) if side=="right" else (n_pad, 0)
    if type(arr)==type(np.zeros((1,))):
        pad_tups = [
            (0,0) if i!= dim else tup for i in range(len(arr.shape))
        ]
        arr = np.pad(arr, pad_tups, constant_values=fill_val)
    elif type(arr)==type(torch.zeros(1)):
        pad_tup = [0 for _ in range(2*len(arr.shape))]
        pad_tup[2*dim+int(side=="right")] = n_pad
        arr = np.pad(arr, tuple(pad_tup), value=fill_val)
    return arr

def generate_square_subsequent_mask(sz: int):
    """
    Generates an upper-triangular matrix of True, with Falses on
    diag and lower triangle.

    Returns:
        BoolTensor (sz,sz)
    """
    #return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return torch.triu(torch.ones(sz, sz), diagonal=1).bool()


def arglast(mask, dim=None, axis=-1):
    """
    This function finds the index of the last max value along a given
    dimension. torch.flip creates a copy of the tensor, so it's
    actually not as efficient as using numpy's np.flip which only
    returns a view.

    Args:
        mask: bool (B,N)
        dim: int
    Returns:
        the index of the last true value along the dimension
    """
    if dim is None: dim = axis
    if type(mask)==type(np.zeros(1)):
        argmaxs = np.argmax(np.flip(mask, axis=dim), axis=dim)
    else:
        argmaxs = torch.argmax(torch.flip(mask, dims=(dim,)), dim=dim)
    return mask.shape[dim] - argmaxs - 1

def padmask2attnmask(pad_mask):
    """
    Converts a padding mask into an attention mask to be argued to
    huggingface's attention_mask. Does so by doing an outer product
    of the row vectors with themselves. This allows you to combine masks
    with more flexibility.

    Args:
        pad_mask: Tensor (B,S)
    Returns:
        attn_mask: Tensor (B,S,S)
    """
    return torch.einsum("bs,bj->bsj", pad_mask, pad_mask)
