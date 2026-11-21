"""PyTorch Dataset and DataLoader collate function for MPRA sequences.

`SeqDataset` keeps sequences as strings. `OneHotCollate`, used as the DataLoader's
`collate_fn`, one-hot encodes them one batch at a time.

The actual datasets used in model training are created through `MPRA_Dataset.to_Dataset()` and `MPRA_Dataset.to_DataLoader()`.

Train/validation/test splits: `find_fold_file` and `fold_split_indices` split rows by
precomputed sequence folds saved as fold files (GraphPart folds or random folds).
"""

import glob
import os

import numpy as np
import pandas as pd
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


# ==== precomputed sequence folds ====
# A fold file is a CSV with columns 'sequence' and 'fold' (an integer from 0 to 4, or empty for a
# sequence that is in no fold). Two scripts write them, in the same layout:
#   scripts/graphpart_fold_split.py  similarity-aware folds; an empty fold marks a sequence that
#                                    GraphPart removed so that no pair of similar sequences ends up
#                                    in two different folds
#   scripts/random_fold_split.py     random folds; no empty folds


def find_fold_file(folders, name_paper: str, name_dataset: str):
    """Path of the fold file of one table, or None if there is none.

    `folders` is a root folder of fold files or a list of them, searched in the given order; the
    first fold file found is returned (e.g. [GraphPart folds, random folds]: random folds are used
    only for tables without GraphPart folds). In each root, looks for, in this order:
      <root>/<name_paper>/<name_dataset>/folds.csv.gz
          the table was split on its own
      <root>/<name_paper>/<any subfolder>/folds_<name_dataset>.csv.gz
          the table was split together with other tables of the same study
    """
    for root in [folders] if isinstance(folders, str) else folders:
        path = os.path.join(root, name_paper, name_dataset, "folds.csv.gz")
        if os.path.isfile(path):
            return path
        pattern = os.path.join(glob.escape(root), glob.escape(name_paper), "*", f"folds_{glob.escape(name_dataset)}.csv.gz")
        matches = sorted(glob.glob(pattern))
        if len(matches) > 1:
            raise ValueError(f"Several fold files for {name_paper}/{name_dataset}: {matches}")
        if matches:
            return matches[0]
    return None


def fold_split_indices(seqs, fold_file: str, folds_val=(3,), folds_test=(4,)):
    """Split rows into train, validation and test index arrays by precomputed sequence folds.

    Each row gets the fold of its sequence in `fold_file`.
    Rows in folds_val go to validation, rows in folds_test to test, and rows in every
    other fold to train. Rows whose sequence has an empty fold (removed by GraphPart) are in
    no split. Rows with the same sequence are always in the same split.

    Args:
        seqs: sequence string of each row
        fold_file: CSV with columns 'sequence' and 'fold' (see find_fold_file)
        folds_val, folds_test: fold numbers (an int or a list of ints) of the validation and test sets

    Returns train, validation and test arrays of row positions, each in increasing order.
    Raises ValueError if a sequence is missing from the fold file, which means the file was
    computed from a different version of the table.
    """
    folds_val = {int(f) for f in np.atleast_1d(folds_val)}
    folds_test = {int(f) for f in np.atleast_1d(folds_test)}
    if folds_val & folds_test:
        raise ValueError(f"Folds {sorted(folds_val & folds_test)} are in both the validation and the test set")

    table = pd.read_csv(fold_file, dtype={"sequence": str, "fold": "Int64"})
    if table["sequence"].duplicated().any():
        raise ValueError(f"{fold_file} lists some sequences more than once")
    fold_of = table.set_index("sequence")["fold"]
    seqs = pd.Series(list(seqs), dtype=str).str.upper()
    missing = ~seqs.isin(fold_of.index)
    if missing.any():
        raise ValueError(
            f"{missing.sum()} of {len(seqs)} sequences are not in {fold_file} (for example "
            f"{seqs[missing].iloc[0][:50]}...); the fold file was made from a different version of the table"
        )
    folds = seqs.map(fold_of).to_numpy(dtype=float, na_value=np.nan)  # NaN: removed by GraphPart
    in_val = np.isin(folds, list(folds_val))
    in_test = np.isin(folds, list(folds_test))
    in_train = ~np.isnan(folds) & ~in_val & ~in_test
    return np.flatnonzero(in_train), np.flatnonzero(in_val), np.flatnonzero(in_test)
