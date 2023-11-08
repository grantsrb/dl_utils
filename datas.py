import pickle
import dl_utils.tokenizer as tokenizer
import torch
import numpy as np
from tqdm import tqdm
import os


class CausalDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 pad_id=0,
                 bos_id=1,
                 eos_id=2,
                 *args, **kwargs):
        """
        Args:
            data: str or list or ndarray or torch LongTensor
                if string, the path to the data. Otherwise, 
                should be a number of sequences that are all the
                same length and already tokenized in id form.
            pad_id: int
                the padding id
        """
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        if type(data)==str:
            data_path = data
            if ".p" in data_path:
                with open(self.data_path, "rb") as f:
                    data = pickle.load(f)
        self.id_seqs = data
        if type(self.id_seqs)==type(np.zeros((1,))):
            self.id_seqs = torch.LongTensor(self.id_seqs)
        elif type(self.id_seqs)==list:
            self.id_seqs = torch.LongTensor(self.id_seqs)
        self.id_seqs = self.id_seqs.long()

    def __len__(self):
        return len(self.id_seqs)

    def __getitem__(self, idx):
        """
        Returns:
            ids: torch LongTensor (S,)
                the integer ids of the tokens
            step_mask: torch LongTensor (S,)
                integers representing which environment step the
                language belongs to.
        """
        samp = self.id_seqs[idx]
        input_ids = samp[:-1]
        in_pad_mask = (input_ids==self.pad_id)|(input_ids==self.eos_id)
        output_ids = samp[1:]
        out_pad_mask = (output_ids==self.pad_id)|(output_ids==self.bos_id)
        return {
            "input_ids": input_ids,
            "input_pad_mask": in_pad_mask,
            "output_ids": output_ids,
            "output_pad_mask": out_pad_mask,
        }

def get_datasets(config):
    """
    This function creates a toy dataset of sequences. The sequences
    consist of a beginning of sequence token, then a starting token k
    that can take K possible values, a string of N ordered digits
    ranging somewhere in the range of 1-100, and a final output of the
    starting token k.

    Args:
        config: dict
            a dict of configuration settings

    Returns:
        tokenizer: huggingface tokenizer or dl_utils tokenizer
        train_dataset: torch Dataset
        val_dataset: torch Dataset
    """
    tkzr = tokenizer.Tokenizer()
    pad_id = tkzr.pad_id
    bos_id = tkzr.bos_id
    eos_id = tkzr.eos_id
    special_ids = tkzr.special_ids
    for k,v in special_ids.items(): config[k] = v
    K = config.get("K", 5)
    N = config.get("N", 10)
    seq_max = config.get("seq_max", 100)
    n_padding = config.get("n_padding", 4)
    k_offset = len(special_ids)
    seq_offset = k_offset + K
    n_samples = config.get("n_samples", 1000)
    samples = []
    for i in range(n_samples):
        leading_pad = np.random.randint(n_padding)
        samp = [pad_id for _ in range(leading_pad)]
        samp.append(bos_id)
        k = np.random.randint(0,K) + k_offset
        samp.append(k)
        seq_start = np.random.randint(0,seq_max-N)
        s = seq_start+seq_offset
        samp += list(range(s, s+N))
        samp.append(k)
        samp.append(eos_id)
        samp += [pad_id for _ in range(n_padding-leading_pad)]
        samples.append(samp)
    samples = np.asarray(samples)
    config["n_tokens"] = seq_offset+seq_max
    config["seq_len"] = samples.shape[-1]
    n_train = int(n_samples*0.8)
    train_samps = samples[:n_train]
    val_samps = samples[n_train:]
    train_dataset = CausalDataset(train_samps, **special_ids)
    val_dataset =   CausalDataset(val_samps, **special_ids)
    return tkzr, train_dataset, val_dataset
