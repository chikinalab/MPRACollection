"""Print one table of test metrics from the metrics.json files written by scripts/train.py,
one row per run directory in results_dir (column 'run' = directory name, which includes the
start time of the run).

Usage:
    python scripts/summarize.py [results_dir]
"""

import json
import os
import sys

results_dir = sys.argv[1] if len(sys.argv) > 1 else "res/cnn_test"
cols = ["run", "name_dataset", "n_train", "test_target_var", "test_mse", "test_pearson", "test_r2", "test_auroc",
        "best_val_pearson", "best_step", "steps_run", "stop_reason", "train_seconds"]
rows = []
for name in sorted(os.listdir(results_dir)):
    path = os.path.join(results_dir, name, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            rows.append({"run": name, **json.load(f)})


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.0f}"
    return "" if v is None else str(v)


table = [cols] + [[fmt(r.get(c)) for c in cols] for r in rows]
widths = [max(len(row[i]) for row in table) for i in range(len(cols))]
for row in table:
    print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
