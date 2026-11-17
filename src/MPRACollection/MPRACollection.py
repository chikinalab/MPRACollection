"""Main module.
Defined MPRA_Dataset class that can be used load custom data.
"""

import yaml
import pandas as pd
import numpy as np
import os
import warnings

from .utils import (
    STUDY_YAML, data_to_XYobs, XYobs_to_data, format_metadata, mkdir, strip_prefix,
)

__SEED__ = 42

TABLE_EXTS = (".csv.gz", ".csv")


class MPRA_Dataset:

    def __init__(
        self,
        folder=None,
        name_paper="",
        name_dataset="",
        info=None,
        data=pd.DataFrame(),
        X=pd.DataFrame(),
        Y=pd.DataFrame(),
        obs_X=pd.DataFrame(),
        obs_Y=pd.DataFrame(),
        Z=dict(),
        names_Z=list(),
        _mpra_dataset=None,
        info_study=None,
    ):
        # info: contents of <name_dataset>.yaml; info_study: contents of study.yaml
        self.info = dict() if info is None else info
        self.info_study = dict() if info_study is None else info_study
        if data.shape == (0, 0) and X.shape == (0, 0):
            # empty dataset, to be filled in later by reload()
            self.data = pd.DataFrame()
            self.X, self.Y, self.obs_X, self.obs_Y = data_to_XYobs(self.data)
        elif data.shape == (0, 0):
            self.data = XYobs_to_data(X, Y, obs_X, obs_Y)
            self.X, self.Y, self.obs_X, self.obs_Y = X, Y, obs_X, obs_Y
        elif X.shape == (0, 0):
            self.data = data
            self.X, self.Y, self.obs_X, self.obs_Y = data_to_XYobs(data)
        else:
            raise ValueError(
                "ONLY ONE of 'data' or 'X' should be provided, but not BOTH."
            )
        
        if Z and names_Z:
            raise ValueError(
                "ONLY ONE of 'Z' or 'names_Z' should be provided, but not BOTH."
            )
        elif Z:
            for name in Z.keys():
                assert Z[name].shape[0] == self.n_seq
            self.Z = Z
        elif names_Z:
            self.Z = self.load_Z(names_Z)
        else:
            self.Z = dict()
        
        # TODO: inherit from _mpra_dataset (type: MPRA_Dataset) when provided
        self.folder = folder
        self.name_paper = name_paper
        self.name_dataset = name_dataset

    def __len__(self):
        return self.data.shape[0]

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_seq(self):
        return self.X.shape[0]

    @property
    def len_max(self):
        # int(): with any missing sequence, pandas returns the maximum as a float
        lengths = self.X['X'].str.len()
        return int(lengths.max()) if lengths.notna().any() else 0
    
    @property
    def dim_seq(self):
        return 4

    @property
    def n_readout(self):
        return self.Y.shape[1]
    
    @property
    def names_readout(self):
        return [col for col in self.Y.columns]
    
    @property
    def n_embed(self):
        return len(self.Z)
    
    @property
    def names_embed(self):
        return [name for name in self.Z.keys()]

    @property
    def n_seqXreadout(self):
        return self.n_seq * self.n_readout

    @property
    def n_readoutXseq(self):
        return self.n_readout * self.n_seq

    def __str__(self) -> str:
        lines = [f"MPRA_Dataset object with n_seq × n_readout = {self.n_seq} × {self.n_readout}"]
        for label, names in [
            ("obs seq", self.obs_X.columns),
            ("obs readout", self.obs_Y.columns),
            ("readout", self.Y.columns),
            ("embed", self.Z.keys()),
        ]:
            lines.append(f"    {label}: '" + "', '".join(names) + "'")
        summary = "".join(line + "\n" for line in lines)
        return summary + format_metadata(
            self.info, self.info_study, self.folder, self.name_paper, self.name_dataset
        )

    def __repr__(self):
        return self.__str__()

    # IO-related
    @classmethod
    def load(cls, name_paper: str, name_dataset: str, folder, usecols=None):
        """Loads dataset information and data from YAML and CSV (.csv.gz or .csv) files.

        usecols: passed to pandas.read_csv to read only some columns, e.g.
            `lambda col: not col.startswith("obs_")` to skip the obs columns of a large table.
        """
        mpra_dataset = cls(folder, name_paper, name_dataset)
        mpra_dataset.reload(usecols=usecols)
        return mpra_dataset

    def reload(self, usecols=None):
        """(Re)loads dataset in place from source files based on current configuration.
        Also fills in an empty dataset, e.g. MPRA_Dataset(folder, name_paper, name_dataset).
        usecols: passed to pandas.read_csv, see load()."""
        folder_paper = os.path.join(self.folder, self.name_paper)
        file_path = os.path.join(folder_paper, self.name_dataset)
        try:
            with open(f"{file_path}.yaml", "r") as f:
                self.info = yaml.safe_load(f) or dict()
            file_study = os.path.join(folder_paper, STUDY_YAML)
            if os.path.exists(file_study):
                with open(file_study, "r") as f:
                    self.info_study = yaml.safe_load(f) or dict()
            else:
                self.info_study = dict()

            file_table = next(
                (file_path + ext for ext in TABLE_EXTS if os.path.exists(file_path + ext)), None
            )
            if file_table is None:
                raise FileNotFoundError(
                    f"No table file found: {file_path}{' or '.join(TABLE_EXTS)}"
                )
            self.data = pd.read_csv(file_table, usecols=usecols, low_memory=False)
            self.X, self.Y, self.obs_X, self.obs_Y = data_to_XYobs(self.data)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Unable to load dataset: {e}")
        except Exception as e:
            raise Exception(f"An error occurred while loading the dataset: {e}")

    def save(self):
        """Saves the current dataset state to <name_dataset>.csv.gz and <name_dataset>.yaml,
        and study.yaml if there is study information."""

        folder_paper = os.path.join(self.folder, self.name_paper)
        mkdir(folder_paper)
        file = os.path.join(folder_paper, self.name_dataset)
        with open(file + ".yaml", "w") as f:
            yaml.safe_dump(self.info, f, sort_keys=False, allow_unicode=True)
        if self.info_study:
            with open(os.path.join(folder_paper, STUDY_YAML), "w") as f:
                yaml.safe_dump(self.info_study, f, sort_keys=False, allow_unicode=True)
        self.data.to_csv(file + ".csv.gz", index=False)
    
    def load_Z(self, names_Z: list):
        """Loads embeddings from files and stores them in the Z attribute."""
        import torch

        if isinstance(names_Z, str):
            names_Z = [names_Z]
        for name in names_Z:
            try:
                self.Z[name] = torch.load(os.path.join(self.folder, self.name_paper, self.name_dataset, name + '.pt'))
                assert self.Z[name].shape[0] == self.n_seq
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Unable to load embeddings: {e}")
            except Exception as e:
                raise Exception(f"An error occurred while loading the embeddings: {e}")
    
    def save_Z(self, names_Z: list):
        """Saves embeddings to files based on the names provided."""
        import torch

        mkdir(os.path.join(self.folder, self.name_paper, self.name_dataset))
        if isinstance(names_Z, str):
            names_Z = [names_Z]
        for name in names_Z:
            try:
                torch.save(self.Z[name], os.path.join(self.folder, self.name_paper, self.name_dataset, name + '.pt'))
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Unable to save embeddings: {e}")
            except Exception as e:
                raise Exception(f"An error occurred while saving the embeddings: {e}")

    # PyTorch-related
    def resolve_cols_Y(self, cols_Y=None):
        """Return readout column names (without the "Y: " prefix).

        cols_Y may be given with or without the prefix. If it is empty or None, the
        default is the table's 'Primary Target Columns', or all readouts if those are not
        given or not all present (e.g. after selecting readouts with dataset['name']).
        """
        if not cols_Y:
            cols_primary = [strip_prefix(col, "Y: ") for col in self.info.get("Primary Target Columns") or []]
            cols_Y = cols_primary if cols_primary and set(cols_primary) <= set(self.Y.columns) else list(self.Y.columns)
        if isinstance(cols_Y, str):
            cols_Y = [cols_Y]
        return [strip_prefix(col, "Y: ") for col in cols_Y]

    def _Y_numeric(self, cols_Y):
        """Readouts in cols_Y as a float64 array; values that are not numbers become NaN."""
        return self.Y[cols_Y].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    def mask_valid(self, cols_Y=None):
        """Boolean array: True for rows that have a sequence and a finite value in every
        readout of cols_Y (see resolve_cols_Y for the default)."""
        cols_Y = self.resolve_cols_Y(cols_Y)
        has_seq = self.X["X"].notna().to_numpy()
        return has_seq & np.isfinite(self._Y_numeric(cols_Y)).all(axis=1)

    def dropna(self, cols_Y=None):
        """Return a new MPRA_Dataset without the rows that mask_valid(cols_Y) rejects."""
        return self[self.mask_valid(cols_Y)]

    def to_Dataset(self, cols_Y=None, names_Z=None):
        """Return a SeqDataset of (sequence string, readouts, embeddings) items.

        Sequences stay strings (uppercased); they are one-hot encoded per batch by the
        collate function of to_DataLoader(). Rows without a sequence or with a missing or
        non-finite readout in cols_Y are dropped, with a warning.

        Args:
            cols_Y: readouts to use as targets (see resolve_cols_Y for the default)
            names_Z: names of embeddings in self.Z to concatenate as Z
        """
        import torch
        from .data import SeqDataset

        cols_Y = self.resolve_cols_Y(cols_Y)
        names_Z = [names_Z] if isinstance(names_Z, str) else list(names_Z or [])

        mask = self.mask_valid(cols_Y)
        if not mask.all():
            warnings.warn(
                f"to_Dataset: dropped {(~mask).sum()} of {len(mask)} rows without a sequence "
                f"or with a missing or non-finite value in {cols_Y}",
                stacklevel=2,
            )
        seqs = self.X["X"][mask].str.upper().tolist()
        Y = self._Y_numeric(cols_Y)[mask].astype(np.float32)
        Z = None
        if names_Z:
            mask_t = torch.from_numpy(mask)
            Z = torch.cat([torch.as_tensor(self.Z[name])[mask_t].float() for name in names_Z], dim=1)
        return SeqDataset(seqs, Y, Z)

    def to_DataLoader(
        self,
        batch_size: int,
        num_workers: int = 1,
        shuffle: bool = True,
        cols_Y=None,
        names_Z=None,
        len_max: int = None,
        collate_fn=None,
        **kwargs,
    ):
        """Return a DataLoader over to_Dataset(cols_Y, names_Z) whose batches are (X, Y, Z),
        with X one-hot encoded of shape (batch, 4, len_max).

        Args:
            len_max: length sequences are padded or truncated to (default: self.len_max).
                Pass the same value for the train, validation and test loaders.
            collate_fn: replaces the default OneHotCollate(len_max), e.g. to use a different
                one-hot encoding (see OneHotCollate in MPRACollection.data).
            **kwargs: passed to torch.utils.data.DataLoader (e.g. drop_last, pin_memory,
                generator, persistent_workers).
        """
        from torch.utils.data import DataLoader
        from .data import OneHotCollate

        if collate_fn is None:
            collate_fn = OneHotCollate(len_max or self.len_max)
        return DataLoader(
            self.to_Dataset(cols_Y, names_Z),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )

    # TODO: handle setitem
    # TODO: handle str or list of str for index (only apply index on Y, not X or Z)
    def __getitem__(self, index):
        import torch

        if isinstance(index, str) or (isinstance(index, list) and all(isinstance(_index, str) for _index in index)):
            if isinstance(index, str):
                index = [index]
            try:
                _Y = self.Y[index].copy()
            except KeyError:
                raise KeyError("Provided index is out of bounds or invalid.")
            _X = self.X.copy()
            _obs_X = self.obs_X.copy()
            _obs_Y = self.obs_Y.copy()
            _Z = self.Z.copy()
            return MPRA_Dataset(
                self.folder, self.name_paper, self.name_dataset, self.info,
                X=_X, Y=_Y, obs_X=_obs_X, obs_Y=_obs_Y, Z=_Z, info_study=self.info_study,
            )

        elif isinstance(index, (int, slice, pd.Series, pd.Index, np.ndarray, torch.Tensor)) \
            or (isinstance(index, list) and all(isinstance(_index, int) for _index in index)):
            try:
                if isinstance(index, int):
                    _data = self.data.iloc[index].copy()
                elif isinstance(index, slice):
                    _data = self.data.iloc[index].copy()
                elif isinstance(index, list):
                    _data = self.data.iloc[index].copy()
                
                elif isinstance(index, pd.Series):
                    _data = self.data.loc[index].copy()
                elif isinstance(index, pd.Index):
                    _data = self.data.loc[index].copy()
                    
                elif isinstance(index, torch.Tensor):
                    index = index.numpy()
                    if index.dtype == bool:
                        _data = self.data.loc[index].copy()
                    elif index.dtype == int:
                        _data = self.data.iloc[index].copy()
                elif isinstance(index, np.ndarray):
                    if index.dtype == bool:
                        _data = self.data.loc[index].copy()
                    elif index.dtype == int:
                        _data = self.data.iloc[index].copy()

                else:
                    raise TypeError(f"Unsupported index type: {type(index).__name__}")
                
            except KeyError:
                raise KeyError("Provided index is out of bounds or invalid.")
            _data = _data.reset_index(drop=True)

            _Z = dict()
            for name in self.Z.keys():
                if isinstance(index, pd.Series) or isinstance(index, pd.Index):
                    index = index.to_numpy()
                _Z[name] = self.Z[name][index]
                
            # Create a new MPRA_Dataset with the selected data
            return MPRA_Dataset(
                self.folder, self.name_paper, self.name_dataset, self.info, data=_data, Z=_Z,
                info_study=self.info_study,
            )
        else:
            raise TypeError(f"Unsupported index type: {type(index).__name__}")

    # @property
    # def embed(self):
    #     return self.Z
    
    # @embed.setter
    # def embed(self, value):
    #     self.Z = value
