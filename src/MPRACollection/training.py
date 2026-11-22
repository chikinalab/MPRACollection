"""Helpers shared by the training scripts (scripts/train.py, scripts/train_scaling_law.py,
scripts/evaluate_splits.py): run configuration files, data preparation, the training loop
(fit), and evaluation.

A run configuration is a YAML file with four sections; missing keys take the values in
DEFAULT_CONFIG:

    name: cuperus                    # default: the file name without extension
    out_dir: /path/to/out_dir        # each run writes to <out_dir>_<YYYYmmdd-HHMMSS>
    data:
      folder: /path/to/data_folder   # null: download from HuggingFace; a list of folders is
                                     # searched in order (e.g. [reviewed, mixtures])
      name_paper: GenomeResearch_2017_Cuperus
      name_dataset: random
      cols_Y: null                   # null: the table's 'Primary Target Columns'
      n_max: 100000                  # random subsample after dropping invalid rows; null: all rows
      split_folder:                  # root folder(s) of the sequence fold files, searched in order;
        - /path/to/graphpart_folds   # a table without a fold file in any of them is an error
        - /path/to/random_folds
      split_folds: {val: 3, test: 4} # folds of the validation and test sets; the other folds are train
      seed: 42                       # seed of the n_max subsample
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
from .data import find_fold_file, fold_split_indices

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
        "split_folder": None,  # root folder(s) of sequence fold files, searched in order (see data.find_fold_file)
        "split_folds": {"val": 3, "test": 4},  # an int or a list of ints each; the other folds are train
        "seed": 42,
    },
    "model": {
        "name": "ResCNN",
        "kwargs": {},
    },
    "train": {
        "max_epochs": 10,
        "max_steps": None,
        "min_steps": 0,  # keep training past max_epochs until this many steps (for small training sets)
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


def load_config(path: str, overrides=(), defaults: dict = None) -> dict:
    """Read a run configuration, apply the 'key=value' overrides, and fill in missing keys
    from `defaults` (DEFAULT_CONFIG if None; a script can pass DEFAULT_CONFIG plus extra sections)."""
    with open(path) as f:
        cfg = _load_yaml(f) or {}
    for assignment in overrides:
        _set(cfg, assignment)
    cfg = _merge(DEFAULT_CONFIG if defaults is None else defaults, cfg)
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


def find_table_folder(folders, name_paper: str, name_dataset: str):
    """The first of `folders` (one folder or a list) that has <name_paper>/<name_dataset>.yaml,
    or None if none has it."""
    for folder in [folders] if isinstance(folders, str) else folders:
        if os.path.isfile(os.path.join(folder, name_paper, f"{name_dataset}.yaml")):
            return folder
    return None


def load_mpra_dataset(cfg_data: dict) -> MPRA_Dataset:
    """Load the table named in the 'data' section, from data.folder if given, otherwise
    from HuggingFace (downloaded into the HuggingFace cache). data.folder may be a list of
    folders; the table is loaded from the first one that has it."""
    usecols = None if cfg_data["read_obs"] else _is_not_obs
    names = (cfg_data["name_paper"], cfg_data["name_dataset"])
    if cfg_data["folder"]:
        folder = find_table_folder(cfg_data["folder"], *names)
        if folder is None:
            raise FileNotFoundError(f"no table {names[0]}/{names[1]}.yaml in data.folder = {cfg_data['folder']}")
        return MPRA_Dataset.load(*names, folder, usecols=usecols)
    from .resources import load_dataset

    return load_dataset(*names, usecols=usecols)


def prepare_splits(cfg_data: dict):
    """Load the table, drop invalid rows, subsample, and split it into train/val/test.

    Steps: load the table; drop rows without a sequence or with a missing or non-finite
    value in cols_Y; if more than n_max rows remain, draw n_max of them at random (the same
    draw as pandas.DataFrame.sample(n=n_max, random_state=seed)); split the rows by the
    sequence folds in the first fold file for the table under the split_folder root(s) (see
    data.find_fold_file): folds split_folds['val'] -> val, split_folds['test'] -> test, the
    other folds -> train; rows whose sequence GraphPart removed are dropped. Raises
    FileNotFoundError if no root has a fold file for the table.

    Returns:
        mpra: the MPRA_Dataset after dropping and subsampling (also contains rows that a
            fold split leaves out)
        splits: dict of split name -> MPRA_Dataset
        cols_Y: readout columns used as targets
    """
    mpra = load_mpra_dataset(cfg_data)
    cols_Y = mpra.resolve_cols_Y(cfg_data["cols_Y"])
    mpra = mpra.dropna(cols_Y)
    seed, n_max = cfg_data["seed"], cfg_data["n_max"]
    if n_max is not None and len(mpra) > n_max:
        mpra = mpra[np.random.RandomState(seed).choice(len(mpra), n_max, replace=False)]

    split_folder = cfg_data["split_folder"]
    fold_file = find_fold_file(split_folder or [], cfg_data["name_paper"], cfg_data["name_dataset"])
    if fold_file is None:
        raise FileNotFoundError(f"no fold file for {cfg_data['name_paper']}/{cfg_data['name_dataset']} under "
                                f"data.split_folder = {split_folder}; make one with scripts/random_fold_split.py")
    folds = cfg_data["split_folds"]
    idx_splits = fold_split_indices(mpra.X["X"], fold_file, folds["val"], folds["test"])
    n_removed = len(mpra) - sum(len(idx) for idx in idx_splits)
    print(f"split: sequence folds from {fold_file}; val folds {folds['val']}, test folds {folds['test']}, "
          f"other folds train; {n_removed} of {len(mpra)} rows left out (sequence removed by GraphPart)")
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


# ==== training ====


def fit(model, loaders: dict, cfg_train: dict, cols_Y: list, device, best_path: str, log_every: int = 50) -> dict:
    """Train `model` on loaders['train'] with Adam, a triangular CyclicLR schedule and MSE loss,
    evaluating on loaders['val'] every eval_every steps and saving the state_dict with the best
    mean validation Pearson r to best_path.

    Training runs for max_epochs epochs, continued past max_epochs until at least min_steps
    steps have run, and stops earlier after `patience` evaluations without improvement or after
    max_steps steps. stop_reason is 'early_stopping', 'max_steps', 'max_epochs' or 'min_steps'
    (the run continued past max_epochs and stopped on reaching min_steps).

    Every `log_every` evaluations, a progress line is printed.

    Returns a dict with best_val_pearson, best_step, steps_run, last_epoch, stop_reason,
    train_seconds and history (one dict per evaluation). Raises ValueError if the validation
    targets are constant for every readout, and RuntimeError if no checkpoint was saved.
    """
    if len(loaders["train"]) == 0:
        raise ValueError("the training loader has no batches (fewer training rows than batch_size with drop_last)")
    # CyclicLR sets the learning rate at every step, starting from lr_min
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train["lr_min"], weight_decay=cfg_train["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=cfg_train["lr_min"], max_lr=cfg_train["lr_max"],
        step_size_up=cfg_train["step_size_up"], mode="triangular", cycle_momentum=False,
    )
    loss_fn = torch.nn.MSELoss()

    # readouts whose validation targets are constant have no defined Pearson r, so they are left
    # out of the mean validation Pearson r (see compute_metrics); if all are, nothing can be selected
    Y_val = loaders["val"].dataset.Y.numpy()
    cols_const = [col for j, col in enumerate(cols_Y) if np.ptp(Y_val[:, j]) == 0]
    if len(cols_const) == len(cols_Y):
        raise ValueError(f"validation targets are constant for every readout {cols_const}: the Pearson r is "
                         "undefined, so no checkpoint can be selected")
    if cols_const:
        print(f"warning: validation targets are constant for {cols_const}; these readouts are left out of "
              "the mean validation Pearson r used for checkpoint selection")

    max_epochs, min_steps, max_steps = cfg_train["max_epochs"], cfg_train["min_steps"], cfg_train["max_steps"]
    history = []
    best_val, best_step, n_bad, step = -np.inf, 0, 0, 0
    n_evals, n_evals_undefined, last_undefined = 0, 0, {}
    train_losses = []
    stop_reason, epoch, last_epoch = None, 0, -1
    t_train = time.time()
    model.train()
    while stop_reason is None:
        if epoch >= max_epochs and step >= min_steps:
            stop_reason = "max_epochs"
            break
        last_epoch = epoch
        for x, yb, _ in loaders["train"]:
            x, yb = x.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            loss = loss_fn(model(x), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            train_losses.append(loss.item())

            if step % cfg_train["eval_every"] == 0:
                val = compute_metrics(*predict(model, loaders["val"], device), cols_Y)
                history.append({
                    "step": step, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": float(np.mean(train_losses)), "val_mse": val["mse"], "val_pearson": val["pearson"],
                })
                train_losses = []
                n_evals += 1
                if np.isnan(val["pearson"]):  # e.g. non-finite predictions; never selected, counts toward patience
                    n_evals_undefined += 1
                    last_undefined = val["undefined_pearson"]
                    if n_evals_undefined == 1:
                        print(f"warning: mean validation Pearson r is undefined at step {step}: {last_undefined}",
                              flush=True)
                if val["pearson"] > best_val:
                    best_val, best_step, n_bad = val["pearson"], step, 0
                    torch.save(model.state_dict(), best_path)
                else:
                    n_bad += 1
                if n_evals % log_every == 0:
                    h = history[-1]
                    print(f"step {step} epoch {epoch} lr {h['lr']:.2e} train_loss {h['train_loss']:.4f} "
                          f"val_mse {h['val_mse']:.4f} val_r {h['val_pearson']:.4f} best_r {best_val:.4f}@{best_step}",
                          flush=True)
                if n_bad >= cfg_train["patience"]:
                    stop_reason = "early_stopping"
                    break
            if max_steps is not None and step >= max_steps:
                stop_reason = "max_steps"
                break
            if epoch >= max_epochs and step >= min_steps:
                stop_reason = "min_steps"
                break
        epoch += 1
    train_seconds = time.time() - t_train
    print(f"stopped: {stop_reason} at step {step} (epoch {last_epoch}); best val r {best_val:.4f} at step "
          f"{best_step}; {train_seconds:.1f} s")
    if n_evals_undefined:
        print(f"warning: {n_evals_undefined} of {n_evals} validation evaluations had an undefined mean Pearson r")
    if best_step == 0:
        if n_evals == 0:
            raise RuntimeError(f"no checkpoint was saved: training stopped after {step} steps, "
                               f"before the first evaluation at step {cfg_train['eval_every']}")
        raise RuntimeError(f"no checkpoint was saved: all {n_evals} validation evaluations gave an undefined "
                           f"mean Pearson r; readouts at the last evaluation: {last_undefined}")
    return {
        "best_val_pearson": float(best_val), "best_step": best_step, "steps_run": step, "last_epoch": last_epoch,
        "stop_reason": stop_reason, "train_seconds": train_seconds, "history": history,
    }


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
