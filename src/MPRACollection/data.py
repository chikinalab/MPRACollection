"""PyTorch Dataset and DataLoader collate function for MPRA sequences.

`SeqDataset` keeps sequences as strings. `OneHotCollate`, used as the DataLoader's
`collate_fn`, one-hot encodes them one batch at a time.

The actual datasets used in model training are created through `MPRA_Dataset.to_Dataset()` and `MPRA_Dataset.to_DataLoader()`.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import seqs_to_onehot


class SeqDataset(Dataset):
    """Sequences (str), readouts and embeddings of one set of MPRA sequences.

    Args:
        seqs: n sequence strings
        Y: readouts, array of shape (n,) or (n, n_readout); stored as float32 (n, n_readout)
        Z: embeddings, array or tensor of shape (n, dim_Z); None gives shape (n, 0)

    Item i is (seqs[i], Y[i], Z[i]).
    """

    def __init__(self, seqs, Y, Z=None):
        self.seqs = list(seqs)
        self.Y = torch.as_tensor(np.asarray(Y, dtype=np.float32))
        if self.Y.ndim == 1:
            self.Y = self.Y[:, None]
        if Z is None:
            self.Z = torch.zeros((len(self.seqs), 0))
        else:
            self.Z = torch.as_tensor(Z, dtype=torch.float32)
        if not len(self.seqs) == len(self.Y) == len(self.Z):
            raise ValueError(
                f"seqs, Y and Z have different lengths: {len(self.seqs)}, {len(self.Y)}, {len(self.Z)}"
            )

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.Y[i], self.Z[i]


class OneHotCollate:
    """Collate function that one-hot encodes a batch of (seq, y, z) items.

    Returns (X, Y, Z): X is float32 of shape (batch, 4, len_max) if channels_first, else
    (batch, len_max, 4); Y is (batch, n_readout); Z is (batch, dim_Z).

    Args:
        len_max: length every sequence is padded (with N) or truncated to. None pads to
            the longest sequence in each batch.
        alphabet: base of each channel, in order.
        value_unknown: value of N in every channel (default 1 / len(alphabet)).
        channels_first: put the 4 channels before the sequence positions.

    A class instead of a closure, so that DataLoader worker processes can pickle it.
    """

    def __init__(self, len_max=None, alphabet="ATGC", value_unknown=None, channels_first=True):
        self.len_max = len_max
        self.alphabet = alphabet
        self.value_unknown = value_unknown
        self.channels_first = channels_first

    def __call__(self, batch):
        seqs, Y, Z = zip(*batch)
        len_max = self.len_max or max(len(seq) for seq in seqs)
        X = seqs_to_onehot(
            seqs, len_max=len_max, alphabet=self.alphabet, value_unknown=self.value_unknown
        )
        X = torch.from_numpy(X.astype(np.float32, copy=False))
        if self.channels_first:
            X = X.transpose(1, 2).contiguous()
        return X, torch.stack(Y), torch.stack(Z)


def random_split_indices(n: int, fractions=(0.7, 0.2, 0.1), seed: int = 42):
    """Randomly split range(n) into train, validation and test index arrays."""
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
