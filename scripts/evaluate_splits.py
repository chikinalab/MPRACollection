"""Evaluate the best.pt of one training run on the train, val and test splits (model in eval mode).

train.py already writes split_metrics.json at the end of each run; use this script to
re-evaluate an existing run, e.g. after a change to the metrics.

Reads the config.yaml that train.py saved in the run directory, so the subsample and split are
rebuilt exactly as in training, even if the original configuration file has changed since.
Writes split_metrics.json in the run directory.

Usage (from the repository root):
    python scripts/evaluate_splits.py --run_dir res/cnn_test/cuperus_20261115-104738
"""

import argparse
import json
import os

import torch

from MPRACollection.model import build_model
from MPRACollection.training import compute_metrics, load_config, make_dataloaders, predict, prepare_splits


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="run directory written by train.py (contains config.yaml and best.pt)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="override configuration values, e.g. data.folder=/path/to/data or train.num_workers=0")
    args = p.parse_args()
    run_dir = args.run_dir
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} not found; runs from before config.yaml was saved cannot be evaluated")
    cfg = load_config(config_path, args.overrides)
    print(f"run '{cfg['name']}' <- {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpra, splits, cols_Y = prepare_splits(cfg["data"])
    len_max = int(mpra.X["X"].str.len().max())
    loaders = make_dataloaders(splits, cfg, cols_Y, len_max, device, shuffle_train=False)

    model = build_model(cfg["model"]["name"], seq_len=len_max, n_out=len(cols_Y), **cfg["model"]["kwargs"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    model = model.to(device)

    results = {name: compute_metrics(*predict(model, loaders[name], device), cols_Y, r2=True)
               for name in ("train", "val", "test")}
    print(cfg["name"], json.dumps(results, indent=2))
    with open(os.path.join(run_dir, "split_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
