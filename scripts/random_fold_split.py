"""Assign every sequence of MPRACollection data tables to one of 5 random folds and save the folds, in the
same file layout as scripts/graphpart_fold_split.py, so that random splits are fixed and reproducible.

Fold of a sequence = BLAKE2b hash of '<seed>:<sequence>' modulo 5, where the sequence is upper-cased and
replaced by the alphabetically smaller of itself and its reverse complement. The fold depends only on the
sequence, so:
  - a sequence gets the same fold in every table (e.g. the Kreimer time points) and in every rerun;
  - a sequence and its exact reverse complement are in the same fold;
  - fold sizes are random but close to even (within about 0.5% for 200,000 sequences).
With data.split_folds {val: 3, test: 4}, about 60% of the sequences are in train, 20% in val, 20% in test.

Tables: every table in configs/data_tables/*.csv, or those given with --tables. The 'study' column is a
study folder; 'data table' may be a wildcard such as '*h'. Entries that match no table file (e.g. the mixtures
'98%random + 2%native') are skipped with a warning. Tables with GraphPart folds get random folds too;
training uses the GraphPart folds for them, because data.split_folder lists the GraphPart folder first.

Output per table, in <out>/<study>/<table>/:
  folds.csv.gz   sequence (upper case, unique), fold (0 to 4)
  summary.json   table file, seed, number of rows and unique sequences, fold sizes, run time
Tables whose folds.csv.gz exists are skipped (unless --overwrite), so an interrupted run can be restarted.

Usage (from the repository root):
  python scripts/random_fold_split.py --list
  python scripts/random_fold_split.py
  python scripts/random_fold_split.py --tables PNAS_2023_Liao/HeLa --out /tmp/random_folds_test
"""
import argparse
import fnmatch
import glob
import hashlib
import json
import os
import sys
import time

import pandas as pd
import yaml

DATA = "/projects/bjgn/jhu30/MPRACollection_data/reviewed"
OUT = "data/training_data/random_folds"
N_FOLDS = 5


def revcomp(seq):
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def random_fold(seq, seed):
    """Fold of one upper-case sequence (see the module docstring)."""
    canonical = min(seq, revcomp(seq))
    digest = hashlib.blake2b(f"{seed}:{canonical}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % N_FOLDS


def tables_in_csvs(paths, data):
    """(study folder, table) pairs listed in the data table CSVs, each once, in order of first appearance."""
    tables = []
    for path in paths:
        for _, row in pd.read_csv(path, dtype=str).iterrows():
            study, pattern = row["study"].strip(), row["data table"].strip()
            study_dir = os.path.join(data, study)
            names = sorted(f[:-len(".csv.gz")] for f in os.listdir(study_dir)
                           if f.endswith(".csv.gz")) if os.path.isdir(study_dir) else []
            matches = [(study, name) for name in names if fnmatch.fnmatchcase(name, pattern)]
            if not matches:
                print(f"warning: {path}: {row['dataset short name']} ({row['study']}, {pattern}): "
                      "no table file; skipped", file=sys.stderr)
            tables += [t for t in matches if t not in tables]
    return tables


def split_table(study, table, args):
    out_dir = os.path.join(args.out, study, table)
    out_file = os.path.join(out_dir, "folds.csv.gz")
    if os.path.exists(out_file) and not args.overwrite:
        print(f"{study}/{table}: {out_file} exists; skipped")
        return
    start = time.time()
    table_file = os.path.join(args.data, study, f"{table}.csv.gz")
    column = "X"  # the primary sequence column, from the table's .yaml if there is one
    if os.path.exists(table_file.replace(".csv.gz", ".yaml")):
        with open(table_file.replace(".csv.gz", ".yaml")) as f:
            column = yaml.safe_load(f)["Primary Sequence Columns"][0]
    rows = pd.read_csv(table_file, usecols=[column], dtype={column: str})[column].dropna().str.upper()
    seqs = rows.drop_duplicates().tolist()
    folds = [random_fold(seq, args.seed) for seq in seqs]

    sizes = [folds.count(k) for k in range(N_FOLDS)]
    mean = len(seqs) / N_FOLDS
    summary = {
        "table": table_file,
        "mode": "random",
        "method": "fold = BLAKE2b('<seed>:<min(sequence, reverse complement)>') mod n_folds "
                  "(scripts/random_fold_split.py)",
        "seed": args.seed,
        "n_folds": N_FOLDS,
        "n_rows": len(rows),
        "n_unique_sequences": len(seqs),
        "fold_sizes": sizes,
        "max_deviation_from_mean": round(max(abs(n - mean) for n in sizes) / mean, 4),
        "minutes": round((time.time() - start) / 60, 2),
    }
    os.makedirs(out_dir, exist_ok=True)
    tmp = out_file + ".tmp.csv.gz"  # written under another name first, so a partial file is never used
    # gzip level 1: pandas' default (level 9) is ~45x slower on DNA text for ~20% smaller files
    pd.DataFrame({"sequence": seqs, "fold": folds}).to_csv(
        tmp, index=False, compression={"method": "gzip", "compresslevel": 1})
    os.replace(tmp, out_file)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"{study}/{table}: {len(rows)} rows, {len(seqs)} unique sequences, fold sizes {sizes} "
          f"(max deviation {summary['max_deviation_from_mean']}); {summary['minutes']} min", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-tables", nargs="+", default=sorted(glob.glob("configs/data_tables/*.csv")),
                   help="data table CSVs with columns 'study' and 'data table' (default configs/data_tables/*.csv)")
    p.add_argument("--tables", nargs="+", metavar="STUDY/TABLE", help="split these tables instead")
    p.add_argument("--data", default=DATA, help=f"folder of the study folders (default {DATA})")
    p.add_argument("--out", default=OUT, help=f"output root folder (default {OUT})")
    p.add_argument("--seed", type=int, default=0, help="hash seed (default 0)")
    p.add_argument("--overwrite", action="store_true", help="redo tables whose folds.csv.gz exists")
    p.add_argument("--list", action="store_true", help="print the tables that would be split and exit")
    args = p.parse_args()

    tables = [tuple(t.split("/")) for t in args.tables] if args.tables else tables_in_csvs(args.data_tables, args.data)
    if args.list:
        print("\n".join(f"{study}/{table}" for study, table in tables))
        return
    print(f"{len(tables)} tables -> {args.out} (seed {args.seed})", flush=True)
    for study, table in tables:
        split_table(study, table, args)


if __name__ == "__main__":
    main()
