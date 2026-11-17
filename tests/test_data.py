"""
Tests for the PyTorch dataset and collate function (`MPRACollection.data`), the
`MPRA_Dataset` methods that create them, and the run configuration helpers
(`MPRACollection.training`). These tests use a small in-memory table and do not go online.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from MPRACollection import MPRA_Dataset
from MPRACollection.data import OneHotCollate, SeqDataset, random_split_indices
from MPRACollection.model import build_model
from MPRACollection.training import compute_metrics, load_config


@pytest.fixture
def mpra_dataset():
    data = pd.DataFrame({
        "X": ["ACGT", "acgtA", None, "GGGN", "TTAC", "CA"],
        "Y: a": [0.5, 1.0, 2.0, np.nan, -1.0, 3.0],
        "Y: b": [1.0, 0.0, 1.0, 1.0, np.inf, 0.0],
        "obs_X: group": ["u", "v", "u", "v", "u", "v"],
    })
    return MPRA_Dataset(info={"Primary Target Columns": ["Y: a"]}, data=data)


def test_resolve_cols_Y(mpra_dataset):
    assert mpra_dataset.resolve_cols_Y() == ["a"]  # the primary target column
    assert mpra_dataset.resolve_cols_Y(["Y: b"]) == ["b"]
    assert mpra_dataset.resolve_cols_Y("a") == ["a"]


def test_mask_valid_and_dropna(mpra_dataset):
    """Rows without a sequence or with a missing/non-finite readout are invalid."""
    assert mpra_dataset.mask_valid().tolist() == [True, True, False, False, True, True]
    assert mpra_dataset.mask_valid(["a", "b"]).tolist() == [True, True, False, False, False, True]
    assert mpra_dataset.dropna(["a", "b"]).X["X"].tolist() == ["ACGT", "acgtA", "CA"]


def test_to_Dataset(mpra_dataset):
    with pytest.warns(UserWarning, match="dropped 2 of 6 rows"):
        dataset = mpra_dataset.to_Dataset()

    assert isinstance(dataset, SeqDataset)
    assert dataset.seqs == ["ACGT", "ACGTA", "TTAC", "CA"]  # uppercased
    assert dataset.Y.dtype == torch.float32
    assert dataset.Y[:, 0].tolist() == [0.5, 1.0, -1.0, 3.0]
    assert dataset.Z.shape == (4, 0)


def test_to_Dataset_names_Z(mpra_dataset):
    mpra_dataset.Z = {"e": torch.arange(12, dtype=torch.float32).reshape(6, 2)}
    dataset = mpra_dataset.to_Dataset(cols_Y=["a", "b"], names_Z=["e"])

    assert dataset.Y.shape == (3, 2)
    assert dataset.Z.tolist() == [[0, 1], [2, 3], [10, 11]]


def test_to_DataLoader(mpra_dataset):
    """Batches are one-hot encoded (ATGC order, N = 0.25) and padded to len_max."""
    loader = mpra_dataset.to_DataLoader(batch_size=2, num_workers=0, shuffle=False, cols_Y=["a"])
    X, Y, Z = next(iter(loader))

    assert X.shape == (2, 4, mpra_dataset.len_max)  # len_max = 5, from all rows
    assert Y.shape == (2, 1) and Z.shape == (2, 0)
    # "ACGT" + one N of padding; channels are A, T, G, C
    expected = torch.tensor([[1, 0, 0, 0, 0.25], [0, 0, 0, 1, 0.25], [0, 0, 1, 0, 0.25], [0, 1, 0, 0, 0.25]])
    assert torch.equal(X[0], expected)


def test_one_hot_collate_alphagenome_encoding():
    """ACGT order, N as all zeros, channels last, padded to the longest sequence in the batch."""
    collate = OneHotCollate(None, alphabet="ACGT", value_unknown=0.0, channels_first=False)
    X, Y, Z = collate([("ACGTN", torch.zeros(1), torch.zeros(0)), ("TG", torch.ones(1), torch.zeros(0))])

    assert X.shape == (2, 5, 4)
    assert torch.equal(X[0], torch.cat([torch.eye(4), torch.zeros(1, 4)]))
    assert torch.equal(X[1, 2:], torch.zeros(3, 4))
    assert Y.tolist() == [[0.0], [1.0]]


def test_random_split_indices():
    idx = random_split_indices(10, (0.7, 0.2, 0.1), seed=0)

    assert [len(i) for i in idx] == [7, 2, 1]
    assert sorted(np.concatenate(idx).tolist()) == list(range(10))


def test_build_model():
    model = build_model("ResCNN", seq_len=20, n_out=3, channels=8, fc_dim=8)

    assert model(torch.zeros(5, 4, 20)).shape == (5, 3)
    with pytest.raises(ValueError):
        build_model("NotAModel", seq_len=20, n_out=1)


def test_compute_metrics():
    y = np.array([[0, 1.0], [1, 2.0], [0, 3.0], [1, 5.0]])
    pred = np.array([[0.1, 1.0], [0.9, 2.0], [0.4, 3.0], [0.2, 5.0]])
    m = compute_metrics(pred, y, ["bin", "cont"], r2=True)

    assert m["per_readout"]["cont"]["pearson"] == pytest.approx(1.0)
    assert m["per_readout"]["cont"]["mse"] == 0
    assert "auroc" in m["per_readout"]["bin"] and "auroc" not in m["per_readout"]["cont"]
    assert m["pearson"] == pytest.approx(np.mean([m["per_readout"][c]["pearson"] for c in ["bin", "cont"]]))
    assert "auroc" not in m  # only averaged over readouts that all have it
    assert m["undefined_pearson"] == {}


def test_compute_metrics_undefined_pearson():
    t = np.array([1.0, 2.0, 3.0, 5.0])
    good = t.copy()  # Pearson r = 1
    const = np.full(4, 2.0)

    # constant target: readout left out of the mean Pearson r and the mean r2
    m = compute_metrics(np.c_[good, t], np.c_[t, const], ["a", "b"], r2=True)
    assert m["undefined_pearson"] == {"b": "constant target"}
    assert np.isnan(m["per_readout"]["b"]["pearson"])
    assert m["pearson"] == pytest.approx(1.0) and m["r2"] == pytest.approx(1.0)

    # constant prediction with a varying target: counts as 0
    m = compute_metrics(np.c_[good, const], np.c_[t, t], ["a", "b"])
    assert m["undefined_pearson"] == {"b": "constant prediction"}
    assert m["pearson"] == pytest.approx(0.5)

    # non-finite prediction: the mean is NaN
    m = compute_metrics(np.c_[good, [np.nan, 1, 2, 3]], np.c_[t, t], ["a", "b"])
    assert m["undefined_pearson"] == {"b": "non-finite prediction"}
    assert np.isnan(m["pearson"])

    # every readout has a constant target: the mean is NaN
    m = compute_metrics(np.c_[good], np.c_[const], ["a"], r2=True)
    assert np.isnan(m["pearson"]) and np.isnan(m["r2"])


def test_load_config(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("out_dir: /tmp/x\ndata: {name_paper: P, name_dataset: D}\nmodel: {kwargs: {channels: 8}}\n")
    cfg = load_config(str(path), ["train.lr_max=1e-3", "data.n_max=null"])

    assert cfg["name"] == "run"  # from the file name
    assert cfg["train"]["lr_max"] == 1e-3 and cfg["train"]["batch_size"] == 128  # override and default
    assert cfg["data"]["n_max"] is None
    assert cfg["model"] == {"name": "ResCNN", "kwargs": {"channels": 8}}

    path.write_text("out_dir: /tmp/x\ndata: {name_paper: P, name_dataset: D, typo: 1}\n")
    with pytest.raises(KeyError, match="data.typo"):
        load_config(str(path))
