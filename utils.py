import numpy as np
import torch
import os
import sys
import subprocess
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
    of the row vectors with themselves. The result allows you to
    combine masks with more flexibility.

    Args:
        pad_mask: Tensor (B,S)
    Returns:
        attn_mask: Tensor (B,S,S)
    """
    return torch.einsum("bs,bj->bsj", pad_mask, pad_mask)

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
    for smask in step_masks:
        smask[smask<0] = torch.max(smask)+1
    shape = [*step_masks[0].shape, step_masks[1].shape[-1]]
    cross_mask = torch.zeros(shape)
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
            a causal cross attention mask. true values mean non-attended
            locations. Does not allow modality x to attend to current
            timestep of modality y and visa-versa.
    """
    # TODO: Allow longer sequence to attend to shorter sequence at
    #   current global timestep and allow shorter sequence to attend
    #   to first substep of longer sequence at current timestep
    cross_mask1 = get_causal_cross_mask(step_masks)
    mode1_mask = get_causal_mask(step_masks[0].shape[-1])
    mode2_mask = get_causal_mask(step_masks[1].shape[-1])
    print("Cross:", cross_mask1.shape)
    print("mode1:", mode1_mask.shape)
    print("mode2:", mode2_mask.shape)
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

def package_versions(globals_dict=None):
    """
    Finds the versions of all packages used in this script

    Args:
        globals_dict: dict
            just argue `globals()`
    """
    if globals_dict is None: globals_dict = globals()
    packages = dict()
    modules = list(set(sys.modules) & set(globals_dict))
    print("Packages:")
    for module_name in modules:
        module = sys.modules[module_name]
        try:
            v = getattr(module, '__version__', 'unknown')
            packages[module_name] = v
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
    print("Numpy")
    x = np.ones((3,))
    print("X:", x.shape)
    for xx in x:
        print(xx)

    px = pad_to(x, 5, dim=0)
    print("Padded dim 0:", px.shape)
    for xx in px:
        print(xx)
    print()
    px = pad_to(x, 5, dim=-1)
    print("Padded dim -1:", px.shape)
    for xx in px:
        print(xx)
    print()
    #px = pad_to(x, 5, side="left", dim=2)
    #print("Padded left dim 2:", px.shape)
    #for xx in px:
    #    print(xx)

    print()
    print("Torch:")
    x = torch.ones((3,))
    print("X:")
    for xx in x:
        print(xx)
    px = pad_to(x, 5, dim=0)
    print("Padded dim 0:", px.shape)
    for xx in px:
        print(xx)
    print()

    px = pad_to(x, 5, dim=-1)
    print("Padded dim -1:", px.shape)
    for xx in px:
        print(xx)
    print()

    px = pad_to(x, 5, side="left", dim=2)
    print("Padded left dim 2:", px.shape)
    for xx in px:
        print(xx)
