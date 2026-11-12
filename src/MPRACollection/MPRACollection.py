"""Main module.
Defined MPRA_Dataset class that can be used load custom data.
"""

import yaml
import pandas as pd
import numpy as np
import os

from .utils import (
    STUDY_YAML, data_to_XYobs, XYobs_to_data, format_metadata, mkdir, seqs_to_onehot, strip_prefix,
)

__SEED__ = 42

TABLE_EXTS = (".csv.gz", ".csv")


class MPRA_Dataset:

    # @staticmethod
    # def list_papers(folder):
    #     # skip hidden entries such as the .cache/ folder made by huggingface
    #     return sorted(name for name in os.listdir(folder) if not name.startswith('.') and os.path.isdir(os.path.join(folder, name)))

    # @staticmethod
    # def list_datasets(name_paper, folder):
    #     return sorted(name[:-4] for name in os.listdir(os.path.join(folder, name_paper)) if name.endswith('.csv'))

    # @staticmethod
    # def print_all(folder):
    #     for name_paper in MPRA_Dataset.list_papers(folder):
    #         print(f'==== {name_paper} ====')
    #         for name_dataset in MPRA_Dataset.list_datasets(name_paper, folder):
    #             print(f'  ==== {name_dataset}')
    #     print(f'==== ==== ==== ====')

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
        return self.X['X'].str.len().max()
    
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
    def load(cls, name_paper: str, name_dataset: str, folder):
        """Loads dataset information and data from YAML and CSV (.csv.gz or .csv) files."""
        mpra_dataset = cls(folder, name_paper, name_dataset)
        mpra_dataset.reload()
        return mpra_dataset

    def reload(self):
        """(Re)loads dataset in place from source files based on current configuration.
        Also fills in an empty dataset, e.g. MPRA_Dataset(folder, name_paper, name_dataset)."""
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
            # pandas infers the compression from the file extension
            self.data = pd.read_csv(file_table, low_memory=False)
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
    def to_Dataset(self, cols_Y: list = [], names_Z: list = [], len_max: int = None):
        import torch
        from torch.utils.data import TensorDataset

        # default: the table's 'Primary Target Columns', or all readouts if they are not
        # given or not all present (e.g. after selecting readouts with dataset['name'])
        if not cols_Y:
            cols_primary = [strip_prefix(col, "Y: ") for col in self.info.get("Primary Target Columns") or []]
            cols_Y = cols_primary if cols_primary and set(cols_primary) <= set(self.Y.columns) else list(self.Y.columns)
        # accept names with or without the "Y: " prefix
        cols_Y = [strip_prefix(col, "Y: ") for col in cols_Y]

        # TODO: should not directly delete the rows with missing values without warning
        # TODO: keep sequence in str and convert to one-hot later in dataset getitem()
        mask = self.Y[cols_Y].notna().all(axis=1)
        len_max = len_max if len_max else self.len_max
        _X = torch.Tensor(
            seqs_to_onehot(self.X["X"][mask].values, len_max=len_max)
        ).transpose(1, 2)
        _Y = torch.Tensor(self.Y[cols_Y][mask].values)

        _Z = torch.zeros((_X.shape[0], 0))
        for name in names_Z:
            _Z = torch.cat((_Z, self.Z[name][mask]), dim=1)

        return TensorDataset(_X, _Y, _Z)

    def to_DataLoader(
        self,
        batch_size: int,
        num_workers: int = 1,
        shuffle: bool = True,
        *args,
        **kwargs,
    ):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.to_Dataset(*args, **kwargs),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
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

    @property
    def seq(self):
        return self.X

    @seq.setter
    def seq(self, value):
        self.X = value

    @property
    def obs_seq(self):
        return self.obs_X

    @obs_seq.setter
    def obs_seq(self, value):
        self.obs_X = value

    @property
    def readout(self):
        return self.Y

    @readout.setter
    def readout(self, value):
        self.Y = value

    @property
    def obs_readout(self):
        return self.obs_Y

    @obs_readout.setter
    def obs_readout(self, value):
        self.obs_Y = value

    # @property
    # def embed(self):
    #     return self.Z
    
    # @embed.setter
    # def embed(self, value):
    #     self.Z = value
