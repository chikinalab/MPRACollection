"""Helpers shared by the training scripts (scripts/train.py, scripts/evaluate_splits.py):
run configuration files, data preparation, and evaluation.

A run configuration is a YAML file with four sections; missing keys take the values in
DEFAULT_CONFIG:

    name: cuperus                    # default: the file name without extension
    out_dir: /path/to/out_dir        # each run writes to <out_dir>_<YYYYmmdd-HHMMSS>
    data:
      folder: /path/to/data_folder   # null: download from HuggingFace
      name_paper: GenomeResearch_2017_Cuperus
      name_dataset: random
      cols_Y: null                   # null: the table's 'Primary Target Columns'
      n_max: 100000                  # random subsample after dropping invalid rows; null: all rows
      split_fractions: [0.7, 0.2, 0.1]
      seed: 42
    model:
      name: ResCNN                   # a key of MPRACollection.model.MODELS
      kwargs: {channels: 256}        # passed to the model's constructor
    train:
      batch_size: 128
      ...
"""

import copy
import os
import re
import time

import numpy as np
import torch
import yaml

from .MPRACollection import MPRA_Dataset
from .data import random_split_indices

SPLITS = ("train", "val", "test")

DEFAULT_CONFIG = {
    "name": None,
    "out_dir": None,
    "data": {
        "folder": None,
        "name_paper": None,
        "name_dataset": None,
        "cols_Y": None,
        "read_obs": False,  # read the obs_X/obs_Y columns (not needed for training)
        "n_max": None,
        "split_fractions": [0.7, 0.2, 0.1],
        "seed": 42,
    },
    "model": {
        "name": "ResCNN",
        "kwargs": {},
    },
    "train": {
        "max_epochs": 10,
        "max_steps": None,
        "batch_size": 128,
        "eval_batch_size": 1024,
        "eval_every": 10,  # steps between validation evaluations
        "patience": 30,  # validation evaluations without improvement before stopping
        "weight_decay": 1e-5,
        "lr_min": 3e-5,  # CyclicLR base_lr
        "lr_max": 3e-4,  # CyclicLR max_lr
        "step_size_up": 500,
        "seed": 42,
        "num_workers": 4,
    },
}


# ==== configuration ====


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader that also reads numbers like 3e-4 or 1e5 (no decimal point) as floats;
    PyYAML follows YAML 1.1, which reads them as strings."""


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def _load_yaml(text_or_file):
    return yaml.load(text_or_file, Loader=_ConfigLoader)


def _merge(default: dict, update: dict, path: str = "") -> dict:
    """Recursively merge `update` into a copy of `default`."""
    out = copy.deepcopy(default)
    for key, value in update.items():
        if default and key not in default:
            raise KeyError(f"Unknown config key '{path}{key}'")
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value, f"{path}{key}.")
        else:
            out[key] = value
    return out


def _set(cfg: dict, assignment: str):
    keys, sep, value = assignment.partition("=")
    if not sep:
        raise ValueError(f"Override '{assignment}' is not of the form key=value")
    *parents, last = keys.split(".")
    node = cfg
    for key in parents:
        node = node.setdefault(key, {})
    node[last] = _load_yaml(value)


def load_config(path: str, overrides=()) -> dict:
    with open(path) as f:
        cfg = _load_yaml(f) or {}
    for assignment in overrides:
        _set(cfg, assignment)
    cfg = _merge(DEFAULT_CONFIG, cfg)
    if cfg["name"] is None:
        cfg["name"] = os.path.splitext(os.path.basename(path))[0]
    for key in ("out_dir",):
        if cfg[key] is None:
            raise ValueError(f"'{key}' is missing in {path}")
    for key in ("name_paper", "name_dataset"):
        if cfg["data"][key] is None:
            raise ValueError(f"'data.{key}' is missing in {path}")
    return cfg


def save_config(cfg: dict, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ==== run directories ====
# Each training run writes to its own directory, named the configuration's out_dir plus the
# start time: out_dir "res/cnn_test/cuperus" -> "res/cnn_test/cuperus_20261115-104738".

RUN_TIME_FORMAT = "%Y%m%d-%H%M%S"


def make_run_dir(out_dir: str) -> str:
    """Create and return a new run directory '<out_dir>_<YYYYmmdd-HHMMSS>'."""
    run_dir = f"{out_dir.rstrip('/')}_{time.strftime(RUN_TIME_FORMAT)}"
    os.makedirs(run_dir, exist_ok=False)  # fails rather than overwrite a run started in the same second
    return run_dir


# ==== data ====


def _is_not_obs(col: str) -> bool:
    return not col.startswith("obs_")


def load_mpra_dataset(cfg_data: dict) -> MPRA_Dataset:
    """Load the table named in the 'data' section, from data.folder if given, otherwise
    from HuggingFace (downloaded into the HuggingFace cache)."""
    usecols = None if cfg_data["read_obs"] else _is_not_obs
    names = (cfg_data["name_paper"], cfg_data["name_dataset"])
    if cfg_data["folder"]:
        return MPRA_Dataset.load(*names, cfg_data["folder"], usecols=usecols)
    from .resources import load_dataset

    return load_dataset(*names, usecols=usecols)


def prepare_splits(cfg_data: dict):
    """Load the table, drop invalid rows, subsample, and split it randomly.

    Steps: load the table; drop rows without a sequence or with a missing or non-finite
    value in cols_Y; if more than n_max rows remain, draw n_max of them at random (the same
    draw as pandas.DataFrame.sample(n=n_max, random_state=seed)); split the rows randomly
    into train/val/test by split_fractions.

    Returns:
        mpra: the MPRA_Dataset after dropping and subsampling
        splits: dict of split name -> MPRA_Dataset
        cols_Y: readout columns used as targets
    """
    mpra = load_mpra_dataset(cfg_data)
    cols_Y = mpra.resolve_cols_Y(cfg_data["cols_Y"])
    mpra = mpra.dropna(cols_Y)
    seed, n_max = cfg_data["seed"], cfg_data["n_max"]
    if n_max is not None and len(mpra) > n_max:
        mpra = mpra[np.random.RandomState(seed).choice(len(mpra), n_max, replace=False)]
    idx_splits = random_split_indices(len(mpra), cfg_data["split_fractions"], seed)
    splits = {name: mpra[idx] for name, idx in zip(SPLITS, idx_splits)}
    return mpra, splits, cols_Y


def make_dataloaders(splits: dict, cfg: dict, cols_Y: list, len_max: int, device, shuffle_train: bool = True) -> dict:
    """DataLoaders for the splits, all with the same len_max. If shuffle_train, the train
    loader shuffles (seeded) and drops the last incomplete batch; the other loaders (and
    the train loader if not shuffle_train) keep the row order and use eval_batch_size."""
    cfg_train = cfg["train"]
    kwargs = dict(
        num_workers=cfg_train["num_workers"],
        cols_Y=cols_Y,
        len_max=len_max,
        pin_memory=torch.device(device).type == "cuda",
        persistent_workers=cfg_train["num_workers"] > 0,
    )
    loaders = {}
    for name, mpra in splits.items():
        if name == "train" and shuffle_train:
            loaders[name] = mpra.to_DataLoader(
                cfg_train["batch_size"], shuffle=True, drop_last=True,
                generator=torch.Generator().manual_seed(cfg_train["seed"]), **kwargs,
            )
        else:
            loaders[name] = mpra.to_DataLoader(cfg_train["eval_batch_size"], shuffle=False, **kwargs)
    return loaders


# ==== evaluation ====


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


@torch.no_grad()
def predict(model, loader, device):
    was_training = model.training
    model.eval()
    preds, ys = [], []
    for x, y, _ in loader:
        preds.append(model(x.to(device, non_blocking=True)).float().cpu())
        ys.append(y)
    model.train(was_training)
    return torch.cat(preds).numpy().astype(np.float64), torch.cat(ys).numpy().astype(np.float64)


def pearson_undefined_reason(p, t):
    """Why the Pearson r of predictions p and targets t is undefined, or None if it is defined."""
    if not np.isfinite(p).all():
        return "non-finite prediction"
    if not np.isfinite(t).all():
        return "non-finite target"
    if np.ptp(t) == 0:
        return "constant target"
    if np.ptp(p) == 0:
        return "constant prediction"
    return None


def compute_metrics(pred, y, cols_Y, r2=False):
    """Metrics per readout, and their mean over readouts.

    Each readout gets mse and pearson; with r2=True also target_var and
    r2 = 1 - mse / target_var; and auroc if its targets are all 0 or 1 (both present).

    The Pearson r of a readout is NaN when it is undefined, and the reason is recorded in
    'undefined_pearson'. The mean Pearson r handles each reason as follows:
      - constant target: the readout is left out (no model can be scored on it); r2 is
        left out too, since target_var is 0
      - constant prediction (target varies): counts as 0, since the predictions explain none
        of the variation; leaving the readout out would reward a model that collapsed on it
      - non-finite prediction or target: the mean is NaN
    The mean is also NaN if every readout is left out.

    Returns a dict with the means over readouts (keys as above; auroc only if every readout
    has it), 'undefined_pearson' (readout name -> reason, empty if all are defined), and
    'per_readout' (readout name -> its metrics).
    """
    per_readout, undefined = {}, {}
    for j, col in enumerate(cols_Y):
        p, t = pred[:, j], y[:, j]
        reason = pearson_undefined_reason(p, t)
        m = {"mse": float(np.mean((p - t) ** 2)), "pearson": float("nan") if reason else pearson(p, t)}
        if reason:
            undefined[col] = reason
        if r2:
            m["target_var"] = float(t.var())
            m["r2"] = 1 - m["mse"] / m["target_var"] if m["target_var"] > 0 else float("nan")
        if np.isin(t, (0, 1)).all() and len(np.unique(t)) == 2:
            from sklearn.metrics import roc_auc_score

            m["auroc"] = float(roc_auc_score(t, p))
        per_readout[col] = m

    scored = [col for col in cols_Y if undefined.get(col) != "constant target"]
    keys = set.intersection(*(set(m) for m in per_readout.values()))
    summary = {}
    for k in (k for k in per_readout[cols_Y[0]] if k in keys):
        if k == "pearson":
            if any(r.startswith("non-finite") for r in undefined.values()):
                values = [float("nan")]
            else:
                values = [0.0 if undefined.get(col) == "constant prediction" else per_readout[col]["pearson"]
                          for col in scored]
        elif k == "r2":
            values = [per_readout[col]["r2"] for col in scored]
        else:
            values = [m[k] for m in per_readout.values()]
        summary[k] = float(np.mean(values)) if values else float("nan")
    summary["undefined_pearson"] = undefined
    summary["per_readout"] = per_readout
    return summary
