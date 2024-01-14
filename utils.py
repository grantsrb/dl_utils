import numpy as np
import torch
import os
import sys
import subprocess
from datetime import datetime
try:
    import cv2
except:
    pass

def device_fxn(device):
    if device<0: return "cpu"
    return device

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

def get_datetime_str():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

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

def pad_list_to(arr, tot_len, fill_val=0, side="right"):
    """
    Pads the argued array to the goal length. Operates in place.

    Args:
        arr: list
        tot_len: int
            the length to pad to
        fill_val: int
            the symbol to use for the padding
        side: str
            pad on the left side or the right
    Returns:
        arr: list
            the padded list
    """
    n_pad = tot_len - len(arr)
    if n_pad<=0: return arr
    if side=="right":
        for i in range(n_pad):
            arr.append(fill_val)
    else:
        padding = [fill_val for _ in range(n_pad)]
        arr = padding + arr
    return arr

def pad_to(arr, tot_len, fill_val=0, side="right", dim=-1):
    """
    Pads the argued list to the goal length along a single dimension.

    Args:
        arr: list or ndarray or torch tensor
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
    if type(arr)==list:
        return pad_list_to(
            arr,
            tot_len=tot_len,
            fill_val=fill_val,
            side=side,
        )
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
        # PyTorch decided to make things complicated by reversing the
        # order that the tuple refers to
        pad_tup[-2*(dim+1)+int(side=="right")] = n_pad
        arr = torch.nn.functional.pad(
            arr, tuple(pad_tup), value=fill_val
        )
    return arr

def num2base(n, b):
    """
    Converts a number to a new base returning a string.
    (Taken from Stack Overflow)

    Args:
        n: int
            the number that is currently in base 10 that you would
            like to convert to another base b
        b: int
            the new number base
    Returns:
        numerals: list of ints
            the numerals of the argued number in the new base
    """

    if n == 0:
        return [0]
    numerals = []
    while n:
        numerals.append(int(n % b))
        n //= b
    return numerals[::-1]

def get_one_hot(ids, L):
    """
    Args:
        ids: torch long tensor (..., N)
        L: int
            the length of the one-hot vector
    Returns:
        one_hots: torch long tensor (..., N, L)
    """
    shape = [*ids.shape, L]
    device = ids.get_device()
    if device<0: device = "cpu"
    one_hots = torch.zeros( shape, device=device )
    one_hots.scatter_(
        dim=-1,
        index=ids[...,None],
        src=torch.ones_like(one_hots)
    )
    return one_hots

def get_mask_past_id(src, id_, incl_id=False):
    """
    Returns a mask in which ones denote all spaces after the first
    occurance of the argued `id_`

    Args:
        src: long tensor  (B,S)
        id_: int
        incl_id: bool
            optionally include the first occurance of the id in the
            mask.
    Returns:
        mask: bool tensor (B,S)
            true values denote indexes past or including the first
            occurance of the `id_` along the last dimension
    """
    return get_mask_past_ids(src, ids=id_)

def get_mask_past_ids(src, ids, incl_id=False):
    """
    Returns a mask in which ones denote all spaces after the first
    occurance of any of the values within `ids`.

    Args:
        src: long tensor  (B,S)
        ids: sequence of ints or int or long tensor (M,)
        incl_id: bool
            optionally include the first occurance of the id in the
            mask.
    Returns:
        mask: bool tensor (B,S)
            true values denote indexes past (or including) the first
            occurance of a value within `ids` along the last dimension
    """
    if type(ids)==int:
        ids = torch.LongTensor([ids])
    elif type(ids)==list or type(ids)==set:
        ids = torch.LongTensor([*ids])
    device = device_fxn(src.get_device())
    ids = ids.to(device)
    B,S = src.shape
    is_id = torch.isin(src, ids).long()
    id_idxs = torch.argmax(is_id, dim=-1)
    # if ids does not appear, then default idx is past last idx
    id_idxs[torch.sum(is_id,dim=-1)==0] = src.shape[-1]
    arange = torch.arange(S)[None].repeat((B,1)).long()
    if incl_id:
        mask = arange.to(device)>=id_idxs[:,None]
    else:
        mask = arange.to(device)>id_idxs[:,None]
    return mask

def get_causal_mask_like(inpt: torch.Tensor):
    """
    Generates an upper-triangular matrix of True, with Falses on
    diag and lower triangle like the argued inpt. Thus, this
    generates a causal mask where True denotes the unattended tokens.

    Args:
        inpt: tensor (...,S,L)
    Returns:
        BoolTensor: (1,S,L)
    """
    if len(inpt.shape)==2:
        S = inpt.shape[-1]
        L = S
        mask = generate_square_subsequent_mask(S)
    else:
        S,L = inpt.shape[-2], inpt.shape[-1]
        mask = generate_square_subsequent_mask(max(S,L))
        mask = mask[:S,:L]
    device = inpt.get_device()
    if device<0: device = "cpu"
    return mask.to(device)[None]

def generate_square_subsequent_mask(
        sz: int,
        device=torch.device(torch._C._get_default_device()),
        dtype="bool"):
    """
    Generates an upper-triangular matrix of True, with Falses on
    diag and lower triangle. Thus, False represents attended indices.

    Args:
        sz: int
            the size of the square mask
        device: int or str or None
        dtype: str ("bool" or "float")
    Returns:
        BoolTensor (sz,sz)
            False values in lower left including the diagonal
    """
    if dtype=="float" or dtype==float:
        mask = torch.triu(
            torch.full(
                (sz, sz),
                float('-inf'),
                device=device
            ).float(),
            diagonal=1,
        )
    else:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask.to(device)


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
    of the row vectors with themselves. The result allows you to
    combine masks with more flexibility.

    IMPORTANT!!!!!!!! The mask must be made such that true values
        denote non-padding, i.e. do-attend-to tokens. Be very
            careful here! This function will not work if the mask is
            inverted.

    Args:
        pad_mask: Tensor (B,S)
            true values denote non-padding, false is padding. Be very
            careful here! This function will not work if the mask is
            inverted.
    Returns:
        attn_mask: Tensor (B,S,S)
    """
    B,S = pad_mask.shape
    reps = (1,S,1)
    return pad_mask[:,None].repeat(reps)
    #return torch.einsum("bs,bj->bsj", pad_mask, pad_mask)

def get_causal_mask(sz: int):
    """
    Generates an upper-triangular matrix of True, with Falses on
    diag and lower triangle.

    Returns:
        BoolTensor (sz,sz)
            True values are masked out (non-attended) values
    """

    return generate_square_subsequent_mask(sz)

def get_causal_cross_mask(step_masks):
    """
    This function uses the high level step indices to build a mask
    to prevent the different modalities from looking ahead in time
    while allowing different numbers of single modal sub steps for
    a given global multi modal time step. To make a cross mask for
    more than 2 modalities, use this function for every possible
    combination and stitch the masks together.

    Args:
        step_masks: list of long tensors [(B,S1), (B,S2)]
            a list of length 2 of tensors that denote the global,
            multi-modal time step of the individual mode.
    Returns:
        cross_mask: bool tensor (B,S1,S2)
            a cross mask to align modalities temporally. The length
            of the list is determined by the number of elements
            in `seq_lens`
    """
    device = step_masks[0].get_device()
    if device<0: device = "cpu"
    for smask in step_masks:
        smask[smask<0] = torch.max(smask)+1
    shape = [*step_masks[0].shape, step_masks[1].shape[-1]]
    cross_mask = torch.zeros(shape).to(device)
    cross_mask = cross_mask + step_masks[0][..., None]
    cross_mask = cross_mask - step_masks[1][:,None]
    cross_mask[cross_mask<=0] = -1
    cross_mask[cross_mask>0] = 0
    cross_mask[cross_mask<0] = 1
    return cross_mask.bool()

def get_full_cross_mask(step_masks):
    """
    Constructs a causal cross mask by stitching different types of
    masks together. The full mask consists of a standard causal mask
    for attending to positions intra-modality (within modality) and a
    causal cross mask for attending inter-modality (outside of modality).

    Mask: [mode1 causal mask,   cross causal mask1 ]
          [cross causal mask2, mode2 causal mask ]

    Args:
        step_masks: list of long tensors [(B,S1), (B,S2)]
            a list of length 2 of tensors that denote the global,
            multi-modal time step of the individual mode.
    Returns:
        cross_mask: bool tensor (B,S1+S2,S1+S2)
            a causal cross attention mask. true values mean padding,
            non-attended locations. Does not allow modality x to
            attend to current timestep of modality y and visa-versa.
    """
    # TODO: Allow longer sequence to attend to shorter sequence at
    #   current global timestep and allow shorter sequence to attend
    #   to first substep of longer sequence at current timestep
    device = step_masks[0].get_device()
    if device<0: device = "cpu"
    cross_mask1 = get_causal_cross_mask(step_masks)
    mode1_mask = get_causal_mask(step_masks[0].shape[-1]).to(device)
    mode2_mask = get_causal_mask(step_masks[1].shape[-1]).to(device)
    cross_mask2 = torch.flip(torch.rot90(
        cross_mask1,k=1,dims=(1,2)
    ),dims=(-1,))
    cross_mask = torch.cat([
        torch.cat([
            mode1_mask[None].repeat((len(cross_mask1),1,1)),
            cross_mask1
        ],dim=-1),
        torch.cat([ 
            cross_mask2, mode2_mask[None].repeat((len(cross_mask1),1,1))
        ],dim=-1)
    ],dim=1)
    return cross_mask

def package_versions(globals_dict=None, verbose=False):
    """
    Finds the versions of all packages used in this script

    Args:
        globals_dict: dict
            just argue `globals()`
    """
    if globals_dict is None: globals_dict = globals()
    packages = dict()
    modules = list(set(sys.modules) & set(globals_dict))
    if verbose:
        print("Packages:")
    for module_name in modules:
        module = sys.modules[module_name]
        try:
            v = getattr(module, '__version__', 'unknown')
            packages[module_name] = v
            if verbose:
                print("\t", module_name, v)
        except:
            packages[module_name] = "unknown"
    return packages

def get_git_revision_hash():
    """
    Finds the current git hash
    """
    return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()


if __name__=="__main__":
    step1 = torch.arange(5).long()
    step2 = []
    for i in range(5):
        step2.append(i)
        step2.append(i)
        if torch.rand(1)>0.5: step2.append(i)
    step2 = torch.LongTensor(step2)
    print("Step1:\n", step1)
    print("Step2:\n", step2)

    print(get_full_cross_mask([step1[None],step2[None]]).long())
