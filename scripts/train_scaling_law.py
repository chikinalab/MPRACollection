"""Scaling law experiment: train the same model on nested random subsets of one MPRA table's
training set, several times per subset with different seeds, and evaluate every run on the
training subset and on the fixed validation and test sets.

Steps: read the configuration (scripts/train.py's format plus a 'scaling' section) -> load the
table named by --dataset (a short name in the data table CSV), drop invalid rows, subsample
and split it once into train/val/test, as scripts/train.py does -> put the training rows in
one random order (scaling.subset_seed); the subset of size n is the first n rows of that
order, so every subset contains all smaller ones -> for each subset size and replicate r,
train a new model with seed train.seed + r (model initialization, dropout and batch order),
with early stopping on the validation set, and evaluate the best checkpoint.

Subset sizes larger than the training set are skipped. For a subset smaller than
train.batch_size, the batch size is lowered to the subset size. train.min_steps keeps small
subsets training past train.max_epochs (see MPRACollection.training.fit).

Output folder: <out_dir>/<dataset>/ (fixed, no time stamp), so that an interrupted or split-up
experiment can be continued: runs whose metrics.json exists are skipped. The first process
writes config.yaml and train_order.csv.gz; later processes check that their configuration
and training order match these files and stop if they do not.
    config.yaml             the experiment configuration (all subset sizes and replicates)
    train_order.csv.gz      the training rows in subset order: columns rank, sequence
    results.csv             one row per finished run (rewritten at the end of every process)
    n<size>_rep<r>/
        config.yaml         the configuration of this run (batch_size and seed as used)
        best.pt             state_dict of the model with the best validation Pearson r
        history.csv         training loss and validation metrics at each evaluation
        metrics.json        run information and the mean metrics on train subset, val and test
        split_metrics.json  full metrics (with per-readout values) on train subset, val and test
        error.json          only if the run failed (the run is then retried by the next process)

Usage (from the repository root):
    python scripts/train_scaling_law.py --config configs/scaling_law/rescnn-medium.yaml --list
    python scripts/train_scaling_law.py --config configs/scaling_law/rescnn-medium.yaml --dataset SS_3
    python scripts/train_scaling_law.py --config configs/scaling_law/rescnn-medium.yaml --dataset SS_3 \\
        --sizes 1 4 16 --replicates 0 1 --set train.max_steps=50 out_dir=/tmp/scaling_test
"""

import argparse
import copy
import csv
import fnmatch
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

from MPRACollection.model import build_model
from MPRACollection.training import (
    DEFAULT_CONFIG, compute_metrics, fit, load_config, make_dataloaders, predict, prepare_splits, save_config,
)

SCALING_DEFAULTS = {
    "sizes": [1, 4, 16, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144],
    "n_replicates": 5,
    "subset_seed": 0,
}
# configuration keys that do not change the results, so they may differ between processes
KEYS_NOT_COMPARED = [("train", "num_workers"), ("train", "eval_batch_size")]
SPLIT_NAMES = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="run configuration file (YAML) with a 'scaling' section")
    p.add_argument("--table", default="configs/data_tables/table2.csv",
                   help="data table CSV with columns 'dataset short name', 'study', 'data table'")
    p.add_argument("--dataset", help="dataset short name from --table (see --list)")
    p.add_argument("--list", action="store_true", help="print the datasets of --table and exit")
    p.add_argument("--sizes", type=int, nargs="+", help="run only these subset sizes (from scaling.sizes)")
    p.add_argument("--replicates", type=int, nargs="+",
                   help="run only these replicates (0 to scaling.n_replicates - 1)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="override configuration values, e.g. train.max_steps=100 scaling.n_replicates=3")
    return p.parse_args()


# ==== datasets ====


def read_data_table(path: str, folder: str) -> dict:
    """Datasets of a data table CSV: short name -> (study, table).

    A 'data table' entry with a wildcard (e.g. '*h' for the Kreimer time points) stands for
    every table of the study whose name matches it; each matching table becomes its own dataset
    named '<short name>_<table>', e.g. Enhancer_2_0h. Matching needs the data folder.
    """
    rows = pd.read_csv(path, dtype=str)
    datasets = {}
    for short, study, table in zip(rows["dataset short name"], rows["study"], rows["data table"]):
        short, study, table = short.strip(), study.strip(), table.strip()
        if any(c in table for c in "*?["):
            if not folder:
                raise ValueError(f"{short}: matching '{table}' needs data.folder")
            names = sorted(os.path.basename(f)[:-len(".yaml")]
                           for f in glob.glob(os.path.join(glob.escape(folder), glob.escape(study), "*.yaml")))
            matches = [n for n in names if n != "study" and fnmatch.fnmatchcase(n, table)]
            if not matches:
                raise ValueError(f"{short}: no table of {study} matches '{table}' in {folder}")
            entries = {f"{short}_{name}": (study, name) for name in matches}
        else:
            entries = {short: (study, table)}
        for key, value in entries.items():
            if key in datasets:
                raise ValueError(f"dataset '{key}' appears twice in {path}")
            datasets[key] = value
    return datasets


# ==== experiment folder ====


def _write_atomic(path: str, write):
    """Write a file through a temporary file, so that another process never reads a partial file."""
    tmp = f"{path}.tmp{os.getpid()}"
    write(tmp)
    os.replace(tmp, path)


def _comparable(cfg: dict) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(cfg))  # same value types as a configuration read from a file
    for section, key in KEYS_NOT_COMPARED:
        cfg.get(section, {}).pop(key, None)
    return cfg


def _differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for key in sorted(set(a) | set(b), key=str)
                for d in _differences(a.get(key), b.get(key), f"{path}{key}.")]
    return [] if a == b else [f"{path.rstrip('.')}: {a!r} (saved) vs {b!r} (now)"]


def check_or_save_config(cfg: dict, path: str):
    """Save the experiment configuration, or check that it matches the saved one."""
    if os.path.exists(path):
        with open(path) as f:
            saved = yaml.safe_load(f)
        diffs = _differences(_comparable(saved), _comparable(cfg))
        if diffs:
            raise ValueError(f"the configuration differs from {path}, which earlier runs of this experiment "
                             "used; use another out_dir or remove the folder:\n  " + "\n  ".join(diffs))
    else:
        _write_atomic(path, lambda tmp: save_config(cfg, tmp))


def check_or_save_order(seqs_ordered: list, path: str):
    """Save the training rows in subset order, or check that they match the saved order."""
    if os.path.exists(path):
        saved = pd.read_csv(path, dtype={"sequence": str})["sequence"].tolist()
        if saved != list(seqs_ordered):
            raise ValueError(f"the training rows or their order differ from {path}; the table, the split "
                             "or the subset seed changed since the earlier runs of this experiment")
    else:
        table = pd.DataFrame({"rank": np.arange(len(seqs_ordered)), "sequence": seqs_ordered})
        _write_atomic(path, lambda tmp: table.to_csv(tmp, index=False, compression="gzip"))


def write_results(exp_dir: str):
    """Collect the metrics.json of every finished run into results.csv (scalar values only)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(glob.escape(exp_dir), "n*_rep*", "metrics.json"))):
        with open(path) as f:
            m = json.load(f)
        rows.append({k: v for k, v in m.items() if not isinstance(v, (dict, list))})
    if not rows:
        return None
    results = pd.DataFrame(rows).sort_values(["n_train", "replicate"])
    path = os.path.join(exp_dir, "results.csv")
    _write_atomic(path, lambda tmp: results.to_csv(tmp, index=False))
    return results


# ==== one run ====


def train_one(cfg, subset, loaders_eval, cols_Y, len_max, device, run_dir, size, replicate, info):
    """Train and evaluate one model on one training subset; writes the run's output files."""
    cfg_run = copy.deepcopy(cfg)
    cfg_train = cfg_run["train"]
    cfg_train["seed"] = cfg["train"]["seed"] + replicate
    cfg_train["batch_size"] = min(cfg["train"]["batch_size"], size)
    if size <= cfg_train["batch_size"]:
        cfg_train["num_workers"] = 0  # one batch per epoch: starting worker processes each epoch costs more
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg_run, os.path.join(run_dir, "config.yaml"))
    torch.manual_seed(cfg_train["seed"])
    np.random.seed(cfg_train["seed"])

    loader_train = make_dataloaders({"train": subset}, cfg_run, cols_Y, len_max, device)["train"]
    model = build_model(cfg["model"]["name"], seq_len=len_max, n_out=len(cols_Y), **cfg["model"]["kwargs"])
    model = model.to(device)
    best_path = os.path.join(run_dir, "best.pt")
    result = fit(model, {"train": loader_train, "val": loaders_eval["val"]}, cfg_train, cols_Y, device, best_path)
    history = result.pop("history")
    del loader_train

    # evaluation of the best checkpoint on the training subset (all rows, unshuffled), val and test
    model.load_state_dict(torch.load(best_path, map_location=device))
    loaders = {"train": make_dataloaders({"train": subset}, cfg_run, cols_Y, len_max, device,
                                         shuffle_train=False)["train"], **loaders_eval}
    split_metrics = {name: compute_metrics(*predict(model, loaders[name], device), cols_Y, r2=True)
                     for name in SPLIT_NAMES}
    metrics = {
        **info, "n_train": size, "replicate": replicate, "seed": cfg_train["seed"],
        "batch_size": cfg_train["batch_size"],
        **{f"{name}_{k}": v for name in SPLIT_NAMES for k, v in split_metrics[name].items()
           if not isinstance(v, dict)},
        **result,
    }
    with open(os.path.join(run_dir, "split_metrics.json"), "w") as f:
        json.dump(split_metrics, f, indent=2)
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()) if history else ["step"])
        writer.writeheader()
        writer.writerows(history)
    # written last: its presence marks the run as finished
    _write_atomic(os.path.join(run_dir, "metrics.json"), lambda tmp: _write_json(metrics, tmp))
    return metrics


def _write_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ==== main ====


def main():
    args = parse_args()
    defaults = {**DEFAULT_CONFIG, "scaling": SCALING_DEFAULTS}
    with open(args.config) as f:
        folder = (yaml.safe_load(f).get("data") or {}).get("folder")
    datasets = read_data_table(args.table, folder)
    if args.list:
        for name, (study, table) in datasets.items():
            print(f"{name}\t{study}/{table}")
        return
    if args.dataset is None:
        sys.exit("error: give --dataset (one of: " + ", ".join(datasets) + ")")
    if args.dataset not in datasets:
        sys.exit(f"error: dataset '{args.dataset}' is not in {args.table}; datasets: " + ", ".join(datasets))
    study, table = datasets[args.dataset]
    # json.dumps quotes the names, so that a name like 'no' or '0h' is read as a string
    overrides = [f"name={json.dumps(args.dataset)}", f"data.name_paper={json.dumps(study)}",
                 f"data.name_dataset={json.dumps(table)}", *args.overrides]
    cfg = load_config(args.config, overrides, defaults=defaults)
    cfg_scaling = cfg["scaling"]

    sizes = sorted(cfg_scaling["sizes"])
    if args.sizes:
        unknown = sorted(set(args.sizes) - set(sizes))
        if unknown:
            sys.exit(f"error: --sizes {unknown} are not in scaling.sizes {sizes}")
        sizes = sorted(set(args.sizes))
    replicates = list(range(cfg_scaling["n_replicates"]))
    if args.replicates:
        unknown = sorted(set(args.replicates) - set(replicates))
        if unknown:
            sys.exit(f"error: --replicates {unknown} are not in 0..{cfg_scaling['n_replicates'] - 1}")
        replicates = sorted(set(args.replicates))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(cfg["out_dir"], args.dataset)
    os.makedirs(exp_dir, exist_ok=True)
    check_or_save_config(cfg, os.path.join(exp_dir, "config.yaml"))
    print(f"scaling law experiment '{args.dataset}' ({study}/{table}) -> {exp_dir}")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # data: one split for all runs
    t0 = time.time()
    mpra, splits, cols_Y = prepare_splits(cfg["data"])
    lengths = mpra.X["X"].str.len()
    len_max = int(lengths.max())
    n_train_full = len(splits["train"])
    print(f"{study}/{table} targets {cols_Y}: {len(mpra)} rows, length {lengths.min()}-{len_max}; "
          + ", ".join(f"{k} {len(v)}" for k, v in splits.items()) + f"; loaded in {time.time() - t0:.1f} s")

    # nested subsets: the first n rows of one random order of the training rows
    order = np.random.default_rng(cfg_scaling["subset_seed"]).permutation(n_train_full)
    train_ordered = splits["train"][order]
    check_or_save_order(train_ordered.X["X"].tolist(), os.path.join(exp_dir, "train_order.csv.gz"))
    skipped = [n for n in sizes if n > n_train_full]
    sizes = [n for n in sizes if n <= n_train_full]
    if skipped:
        print(f"subset sizes {skipped} skipped: larger than the training set ({n_train_full} rows)")

    loaders_eval = make_dataloaders({"val": splits["val"], "test": splits["test"]}, cfg, cols_Y, len_max, device)
    info = {
        "dataset": args.dataset, "name_paper": study, "name_dataset": table, "cols_Y": ",".join(cols_Y),
        "n_train_full": n_train_full, "n_val": len(splits["val"]), "n_test": len(splits["test"]),
        "seq_len_max": len_max,
    }
    n_params = sum(p.numel() for p in build_model(cfg["model"]["name"], seq_len=len_max, n_out=len(cols_Y),
                                                  **cfg["model"]["kwargs"]).parameters())
    print(f"{cfg['model']['name']} parameters: {n_params:,}; sizes {sizes}; replicates {replicates}")

    failed = []
    for size in sizes:
        subset = train_ordered[np.arange(size)]
        for rep in replicates:
            run_dir = os.path.join(exp_dir, f"n{size}_rep{rep}")
            if os.path.exists(os.path.join(run_dir, "metrics.json")):
                print(f"\n== n_train {size}, replicate {rep}: already finished, skipped")
                continue
            print(f"\n== n_train {size}, replicate {rep} -> {run_dir}", flush=True)
            try:
                m = train_one(cfg, subset, loaders_eval, cols_Y, len_max, device, run_dir, size, rep, info)
            except RuntimeError as e:  # fit found no checkpoint to keep; record it and go on
                traceback.print_exc()
                failed.append((size, rep))
                with open(os.path.join(run_dir, "error.json"), "w") as f:
                    json.dump({"n_train": size, "replicate": rep, "error": str(e)}, f, indent=2)
                continue
            error_file = os.path.join(run_dir, "error.json")
            if os.path.exists(error_file):
                os.remove(error_file)
            print(f"n_train {size} rep {rep}: test pearson {m['test_pearson']:.4f}, r2 {m['test_r2']:.4f}; "
                  f"val pearson {m['val_pearson']:.4f}; train pearson {m['train_pearson']:.4f}; "
                  f"{m['steps_run']} steps ({m['stop_reason']}), {m['train_seconds']:.0f} s", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results = write_results(exp_dir)
    if results is not None:
        summary = results.groupby("n_train")[["test_pearson", "test_r2"]].agg(["mean", "std", "count"])
        print(f"\nfinished runs in {exp_dir}/results.csv (test metrics over replicates):")
        print(summary.round(4).to_string())
    if failed:
        sys.exit(f"{len(failed)} runs failed (see error.json in their folders): {failed}")


if __name__ == "__main__":
    main()
