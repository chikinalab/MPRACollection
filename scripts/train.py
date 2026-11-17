
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
from torch import nn

from MPRACollection.model import build_model
from MPRACollection.training import (
    compute_metrics, load_config, make_dataloaders, make_run_dir, predict, prepare_splits, save_config,
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

    # model and optimization
    model = build_model(cfg["model"]["name"], seq_len=len_max, n_out=len(cols_Y), **cfg["model"]["kwargs"])
    model = model.to(device)
    print(f"{cfg['model']['name']} parameters: {sum(p.numel() for p in model.parameters()):,}")
    # CyclicLR sets the learning rate at every step, starting from lr_min
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train["lr_min"], weight_decay=cfg_train["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=cfg_train["lr_min"], max_lr=cfg_train["lr_max"],
        step_size_up=cfg_train["step_size_up"], mode="triangular", cycle_momentum=False,
    )
    loss_fn = nn.MSELoss()

    # readouts whose validation targets are constant have no defined Pearson r, so they are left
    # out of the mean validation Pearson r (see compute_metrics); if all are, nothing can be selected
    Y_val = splits["val"].Y[cols_Y].to_numpy(dtype=np.float64)
    cols_const = [col for j, col in enumerate(cols_Y) if np.ptp(Y_val[:, j]) == 0]
    if len(cols_const) == len(cols_Y):
        raise ValueError(f"validation targets are constant for every readout {cols_const}: the Pearson r is "
                         "undefined, so no checkpoint can be selected")
    if cols_const:
        print(f"warning: validation targets are constant for {cols_const}; these readouts are left out of "
              "the mean validation Pearson r used for checkpoint selection")

    # training with early stopping on the mean validation Pearson r
    best_path = os.path.join(out_dir, "best.pt")
    history = []
    best_val, best_step, n_bad, step = -np.inf, 0, 0, 0
    n_evals, n_evals_undefined, last_undefined = 0, 0, {}
    train_losses = []
    stop_reason = "max_epochs"
    t_train = time.time()
    model.train()
    for epoch in range(cfg_train["max_epochs"]):
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
                if step % (cfg_train["eval_every"] * 50) == 0:
                    h = history[-1]
                    print(f"step {step} epoch {epoch} lr {h['lr']:.2e} train_loss {h['train_loss']:.4f} "
                          f"val_mse {h['val_mse']:.4f} val_r {h['val_pearson']:.4f} best_r {best_val:.4f}@{best_step}",
                          flush=True)
                if n_bad >= cfg_train["patience"]:
                    stop_reason = "early_stopping"
                    break
            if cfg_train["max_steps"] is not None and step >= cfg_train["max_steps"]:
                stop_reason = "max_steps"
                break
        if stop_reason != "max_epochs":
            break
    train_seconds = time.time() - t_train
    print(f"stopped: {stop_reason} at step {step} (epoch {epoch}); best val r {best_val:.4f} at step {best_step}; "
          f"{train_seconds:.1f} s")
    if n_evals_undefined:
        print(f"warning: {n_evals_undefined} of {n_evals} validation evaluations had an undefined mean Pearson r")
    if best_step == 0:
        if n_evals == 0:
            raise RuntimeError(f"no checkpoint was saved: training stopped after {step} steps, "
                               f"before the first evaluation at step {cfg_train['eval_every']}")
        raise RuntimeError(f"no checkpoint was saved: all {n_evals} validation evaluations gave an undefined "
                           f"mean Pearson r; readouts at the last evaluation: {last_undefined}")

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
        "best_val_pearson": float(best_val), "best_step": best_step, "steps_run": step,
        "stop_reason": stop_reason, "train_seconds": train_seconds,
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
