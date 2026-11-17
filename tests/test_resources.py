"""
Tests for downloading datasets from HuggingFace (`MPRACollection.resources`)
and for loading them with `MPRA_Dataset` (`MPRACollection.MPRACollection`).

These tests call the real HuggingFace API. Skip them with `pytest -m "not hf"`.
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from MPRACollection import resources
from MPRACollection import MPRA_Dataset
from MPRACollection.data import SeqDataset

pytestmark = pytest.mark.hf

# A small dataset (~700 sequences) to keep the tests fast
PAPER = "NatureGenetics_2023_Cohen"
DATASET = "state"


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    """Download the test dataset once into a temporary folder."""
    local_dir = tmp_path_factory.mktemp("data")
    resources.download_dataset(PAPER, DATASET, local_dir=local_dir)
    return local_dir


@pytest.fixture
def mpra_dataset(folder):
    return MPRA_Dataset.load(PAPER, DATASET, folder)


# ==== resources.py ====


def test_list_papers():
    """list_papers() returns the paper folders in the HuggingFace repo."""
    papers = resources.list_papers()

    assert isinstance(papers, list)
    assert PAPER in papers
    assert not any(p.startswith(".") for p in papers)


def test_list_datasets():
    """list_datasets() returns the dataset names of one paper."""
    assert resources.list_datasets(PAPER) == ["cycle", "mixed", "state"]
    # APA.csv.backup should not be listed
    assert resources.list_datasets("Cell_2019_Seelig") == ["APA"]


def test_download_dataset():
    """download_dataset() saves files in the HuggingFace cache by default."""
    path = resources.download_dataset(PAPER, DATASET)

    assert isinstance(path, str)
    assert path.endswith(f"{PAPER}/{DATASET}.csv")
    assert os.path.exists(path)
    assert os.path.exists(path[:-len(".csv")] + ".yaml")


def test_download_dataset_local_dir(tmp_path):
    """download_dataset() saves files in local_dir when it is given."""
    path = resources.download_dataset(PAPER, DATASET, local_dir=tmp_path)

    assert os.path.samefile(path, tmp_path / PAPER / f"{DATASET}.csv")
    assert os.path.exists(tmp_path / PAPER / f"{DATASET}.yaml")


def test_download_paper(tmp_path):
    """download_paper() downloads every dataset of one paper."""
    paths = resources.download_paper(PAPER, local_dir=tmp_path)

    assert len(paths) == 3
    assert all(os.path.exists(p) for p in paths)


def test_load_dataset():
    """load_dataset() loads from the folder that the files were downloaded to."""
    path = resources.download_dataset(PAPER, DATASET)
    mpra_dataset = resources.load_dataset(PAPER, DATASET)

    assert isinstance(mpra_dataset, MPRA_Dataset)
    assert os.path.samefile(mpra_dataset.folder, os.path.dirname(os.path.dirname(path)))
    assert mpra_dataset.name_paper == PAPER
    assert mpra_dataset.name_dataset == DATASET


def test_load_dataset_local_dir(tmp_path):
    """load_dataset() passes local_dir on to MPRA_Dataset."""
    mpra_dataset = resources.load_dataset(PAPER, DATASET, local_dir=tmp_path)

    assert os.path.samefile(mpra_dataset.folder, tmp_path)


# ==== MPRACollection.py ====


def test_list_papers_local(folder):
    """MPRA_Dataset.list_papers() skips the .cache/ folder made by huggingface."""
    assert os.path.isdir(folder / ".cache")
    assert MPRA_Dataset.list_papers(folder) == [PAPER]


def test_list_datasets_local(folder):
    assert MPRA_Dataset.list_datasets(PAPER, folder) == [DATASET]


def test_print_all(folder, capsys):
    MPRA_Dataset.print_all(folder)
    out = capsys.readouterr().out

    assert PAPER in out
    assert DATASET in out


def test_load(mpra_dataset):
    """MPRA_Dataset.load() reads the .csv and .yaml files."""
    assert len(mpra_dataset) == mpra_dataset.n_seq > 0
    assert mpra_dataset.names_readout == ["expr cd34+", "expr diff", "expr cd24+"]
    assert mpra_dataset.n_readout == 3
    assert list(mpra_dataset.X.columns) == ["X"]
    assert isinstance(mpra_dataset.info, dict)
    assert mpra_dataset.info["Species"] == "Human"


def test_load_missing_dataset(folder):
    with pytest.raises(FileNotFoundError):
        MPRA_Dataset.load(PAPER, "not_a_dataset", folder)


def test_init_empty():
    """MPRA_Dataset() without 'data' or 'X' is an empty dataset."""
    mpra_dataset = MPRA_Dataset()

    assert len(mpra_dataset) == mpra_dataset.n_seq == mpra_dataset.n_readout == 0
    assert mpra_dataset.info == {} and mpra_dataset.Z == {}
    # empty datasets should not share the same info dict
    assert mpra_dataset.info is not MPRA_Dataset().info


def test_init_empty_then_reload(folder, mpra_dataset):
    """An empty dataset is filled in by reload()."""
    empty = MPRA_Dataset(folder, PAPER, DATASET)
    empty.reload()

    pd.testing.assert_frame_equal(empty.data, mpra_dataset.data)
    assert empty.info == mpra_dataset.info


def test_init_with_data_and_X(mpra_dataset):
    with pytest.raises(ValueError):
        MPRA_Dataset(data=mpra_dataset.data, X=mpra_dataset.X)


@pytest.mark.parametrize(
    "index",
    [
        slice(0, 3),
        [0, 1, 2],
        np.arange(3),
        torch.arange(3),
    ],
)
def test_getitem_rows(mpra_dataset, index):
    """Indexing by rows keeps all columns and the folder."""
    subset = mpra_dataset[index]

    assert isinstance(subset, MPRA_Dataset)
    assert subset.shape == (3, mpra_dataset.shape[1])
    assert subset.folder == mpra_dataset.folder
    pd.testing.assert_frame_equal(subset.data, mpra_dataset.data.iloc[:3])


def test_getitem_mask(mpra_dataset):
    """Indexing by a boolean mask keeps the matching rows."""
    mask = mpra_dataset.Y["expr diff"] > 0
    subset = mpra_dataset[mask]

    assert len(subset) == mask.sum()
    assert (subset.Y["expr diff"] > 0).all()

    subset = mpra_dataset[mask.to_numpy()]
    assert len(subset) == mask.sum()


def test_getitem_readout(mpra_dataset):
    """Indexing by readout names keeps only those readouts."""
    subset = mpra_dataset["expr diff"]

    assert subset.names_readout == ["expr diff"]
    assert subset.n_seq == mpra_dataset.n_seq


def test_save_and_reload(mpra_dataset, tmp_path):
    """save() writes files that load() reads back unchanged."""
    copy = MPRA_Dataset(tmp_path, PAPER, "copy", mpra_dataset.info, mpra_dataset.data)
    copy.save()
    loaded = MPRA_Dataset.load(PAPER, "copy", tmp_path)

    pd.testing.assert_frame_equal(loaded.data, mpra_dataset.data)
    assert loaded.info == mpra_dataset.info

    loaded.data = loaded.data.iloc[:3]
    loaded.reload()
    assert len(loaded) == len(mpra_dataset)


def test_to_Dataset(mpra_dataset):
    """to_Dataset() keeps sequences as strings; rows with missing readouts are dropped."""
    cols_Y = mpra_dataset.resolve_cols_Y()
    n_valid = mpra_dataset.mask_valid(cols_Y).sum()
    dataset = mpra_dataset.to_Dataset()
    seq, y, z = dataset[0]

    assert isinstance(dataset, SeqDataset)
    assert len(dataset) == n_valid
    assert isinstance(seq, str)
    assert y.shape == (len(cols_Y),)
    assert z.shape == (0,)


def test_to_Dataset_cols_Y(mpra_dataset):
    dataset = mpra_dataset.to_Dataset(cols_Y=["expr diff"])

    assert dataset.Y.shape == (len(dataset), 1)


def test_to_DataLoader(mpra_dataset):
    """to_DataLoader() one-hot encodes each batch to shape (batch, 4, len_max)."""
    dataloader = mpra_dataset.to_DataLoader(batch_size=8, num_workers=0, shuffle=False)
    X, Y, Z = next(iter(dataloader))

    assert isinstance(dataloader, DataLoader)
    assert X.shape == (8, 4, mpra_dataset.len_max)
    assert Y.shape == (8, len(mpra_dataset.resolve_cols_Y()))
    assert Z.shape == (8, 0)
