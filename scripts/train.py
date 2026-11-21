
"""Train a model on one MPRA table, as described by a run configuration file, and evaluate
the best checkpoint on the train, val and test splits.

Steps: read the configuration -> load the table as an MPRA_Dataset, drop invalid rows,
subsample and split it -> create the DataLoaders (sequences are one-hot encoded per batch)
-> train with early stopping on the mean validation Pearson r -> evaluate the best
checkpoint on the train, val and test splits. See MPRACollection/training.py for the configuration format.

Each run writes to a new directory '<out_dir>_<YYYYmmdd-HHMMSS>' (out_dir from the
configuration, time = start of the run), so earlier runs are never overwritten. Output files:
    config.yaml         the configuration with defaults and overrides filled in
    best.pt             state_dict of the model with the best validation Pearson r
    history.csv         training loss and validation metrics at each evaluation
    metrics.json        test metrics of best.pt, with data sizes and training summary
    split_metrics.json  metrics of best.pt on the train, val and test splits

Usage (from the repository root):
    python scripts/train.py --config configs/rescnn/cuperus.yaml
    python scripts/train.py --config configs/rescnn/cuperus.yaml --set train.max_steps=100 out_dir=/tmp/test
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from MPRACollection.model import build_model
from MPRACollection.training import (
    compute_metrics, fit, load_config, make_dataloaders, make_run_dir, predict, prepare_splits, save_config,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="run configuration file (YAML)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="override configuration values, e.g. train.lr_max=1e-3 data.n_max=null")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg_train = cfg["train"]
    torch.manual_seed(cfg_train["seed"])
    np.random.seed(cfg_train["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = make_run_dir(cfg["out_dir"])
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    print(f"run '{cfg['name']}' -> {out_dir}")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # data
    t0 = time.time()
    mpra, splits, cols_Y = prepare_splits(cfg["data"])
    lengths = mpra.X["X"].str.len()
    len_max = int(lengths.max())
    loaders = make_dataloaders(splits, cfg, cols_Y, len_max, device)
    Y = mpra.Y[cols_Y].to_numpy(dtype=np.float64)
    print(f"{mpra.name_paper}/{mpra.name_dataset} targets {cols_Y}: {len(mpra)} rows, "
          f"length {lengths.min()}-{len_max}, target mean {Y.mean(axis=0).round(4).tolist()} "
          f"sd {Y.std(axis=0).round(4).tolist()}; "
          + ", ".join(f"{k} {len(v)}" for k, v in splits.items())
          + f"; loaded in {time.time() - t0:.1f} s")

    # model, then training with early stopping on the mean validation Pearson r
    model = build_model(cfg["model"]["name"], seq_len=len_max, n_out=len(cols_Y), **cfg["model"]["kwargs"])
    model = model.to(device)
    print(f"{cfg['model']['name']} parameters: {sum(p.numel() for p in model.parameters()):,}")
    best_path = os.path.join(out_dir, "best.pt")
    result = fit(model, loaders, cfg_train, cols_Y, device, best_path)
    history = result.pop("history")

    # evaluation of the best checkpoint on all three splits (the training loader shuffles and
    # drops the last incomplete batch, so the train split gets an unshuffled loader with all rows)
    model.load_state_dict(torch.load(best_path, map_location=device))
    loaders["train"] = make_dataloaders({"train": splits["train"]}, cfg, cols_Y, len_max, device,
                                        shuffle_train=False)["train"]
    split_metrics = {name: compute_metrics(*predict(model, loaders[name], device), cols_Y, r2=True)
                     for name in ("train", "val", "test")}
    test = split_metrics["test"]
    metrics = {
        "name": cfg["name"], "name_paper": mpra.name_paper, "name_dataset": mpra.name_dataset, "cols_Y": cols_Y,
        "n_train": len(splits["train"]), "n_val": len(splits["val"]), "n_test": len(splits["test"]),
        "seq_len_max": len_max,
        **{f"test_{k}": v for k, v in test.items()},
        **{k: result[k] for k in ("best_val_pearson", "best_step", "steps_run", "stop_reason", "train_seconds")},
    }
    print(json.dumps(metrics, indent=2))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "split_metrics.json"), "w") as f:
        json.dump(split_metrics, f, indent=2)
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()) if history else ["step"])
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    main()
