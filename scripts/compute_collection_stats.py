"""Compute sequence-level summary statistics of the MPRA Collection.

This script reads every data table (`<study>/<table>.csv.gz`) once and writes small CSV files that the notebook
`figures/mpracollection_stat.ipynb` uses for plotting. Reading the raw tables (about 7 GB compressed, about 160 million
unique sequences) takes too long to repeat inside the notebook.

A "unique sequence" is a distinct string in the `X` column. A sequence and its reverse complement count as two
sequences, as in the `number of unique_inputs` field of the YAML files. To save memory, each sequence is replaced by a
64-bit hash (`pandas.util.hash_pandas_object`, deterministic across runs). With about 1.6e8 sequences the expected
number of hash collisions is below 0.001, so the counts are exact in practice.

Step 1 (one worker per table, run in parallel) stores, for every table, the distinct combinations of
(sequence hash, sequence length, sequence origin, control role) in a `.npz` cache file. Step 2 merges the cache files
per study and over the whole collection.

Output files in `--out-dir`:
    table_stats.csv          one row per table: rows, unique sequences, unique test sequences, length range
    study_stats.csv          one row per study: the same statistics with sequences deduplicated across its tables
    study_origin_counts.csv  unique sequences per study, sequence origin and control role
    global_origin_counts.csv unique sequences per sequence origin among test sequences, deduplicated across studies
    study_length_counts.csv  number of unique sequences of each length, per study

Usage:
    python scripts/compute_collection_stats.py --n-workers 4
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = "/projects/bjgn/jhu30/MPRACollection_data/reviewed"
CACHE_DIR = "/work/hdd/bjgn/jhu30/MPRACollection_stats_cache"
OUT_DIR = str(Path(__file__).resolve().parents[1] / "res" / "collection_stat" / "data")

COL_X = "X"
COL_ORIGIN = "obs_X: sequence origin"
COL_ROLE = "obs_X: control role"
ORIGINS = ["native", "variant", "perturbed", "synthetic", "model_designed", "random", "shuffled"]
ROLES = ["test", "positive_control", "negative_control", "control"]
CHUNK_ROWS = 2_000_000


def list_tables(data_dir):
    """Return (study, table, path) for every data table, largest file first."""
    tables = []
    for study in sorted(os.listdir(data_dir)):
        study_dir = Path(data_dir) / study
        if not study_dir.is_dir():
            continue
        for path in sorted(study_dir.glob("*.csv.gz")):
            tables.append((study, path.name[: -len(".csv.gz")], str(path)))
    tables.sort(key=lambda t: os.path.getsize(t[2]), reverse=True)
    return tables


def encode(values, categories):
    """Map labels to integer codes; labels outside `categories` (including missing values) become -1."""
    return pd.Categorical(values, categories=categories).codes.astype(np.int8)


def unique_rows(arrays):
    """Indices of the distinct (hash, origin, role) rows. Sorting with NumPy needs far less memory than
    pandas.drop_duplicates, which matters for tables with tens of millions of rows. The length is not part of the key
    because it is determined by the sequence, and therefore by the hash."""
    h, origin, role = arrays["h"], arrays["origin"], arrays["role"]
    order = np.lexsort((role, origin, h))
    hs, os_, rs = h[order], origin[order], role[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = (hs[1:] != hs[:-1]) | (os_[1:] != os_[:-1]) | (rs[1:] != rs[:-1])
    return order[keep]


def subset(arrays, idx):
    return {k: v[idx] for k, v in arrays.items()}


def concat(list_of_arrays):
    return {k: np.concatenate([a[k] for a in list_of_arrays]) for k in ("h", "length", "origin", "role")}


def process_table(args):
    """Step 1: reduce one table to its distinct (hash, length, origin, role) combinations and cache them."""
    study, table, path, cache_dir, overwrite = args
    cache_file = Path(cache_dir) / study / f"{table}.npz"
    if cache_file.exists() and not overwrite and cache_file.stat().st_mtime > os.path.getmtime(path):
        with np.load(cache_file) as z:
            return dict(study=study, table=table, n_rows=int(z["n_rows"]), n_missing_X=int(z["n_missing_X"]), cached=True)

    t0 = time.time()
    parts, n_rows, n_missing = [], 0, 0
    reader = pd.read_csv(path, usecols=[COL_X, COL_ORIGIN, COL_ROLE], dtype=str, chunksize=CHUNK_ROWS)
    for chunk in reader:
        n_rows += len(chunk)
        missing = chunk[COL_X].isna()
        n_missing += int(missing.sum())
        chunk = chunk[~missing]
        seqs = chunk[COL_X]
        part = {
            "h": pd.util.hash_pandas_object(seqs, index=False).to_numpy(np.uint64),
            "length": seqs.str.len().to_numpy(dtype=np.int32),
            "origin": encode(chunk[COL_ORIGIN].str.strip(), ORIGINS),
            "role": encode(chunk[COL_ROLE].str.strip(), ROLES),
        }
        del chunk, seqs
        parts.append(subset(part, unique_rows(part)))
    if parts:
        arrays = concat(parts)
        del parts
        arrays = subset(arrays, unique_rows(arrays))
    else:
        arrays = dict(h=np.empty(0, np.uint64), length=np.empty(0, np.int32), origin=np.empty(0, np.int8), role=np.empty(0, np.int8))

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_name(cache_file.name + ".tmp.npz")
    np.savez(tmp, **arrays, n_rows=n_rows, n_missing_X=n_missing)
    os.replace(tmp, cache_file)
    print(f"  {study}/{table}: {n_rows:,} rows, {time.time() - t0:.0f} s", flush=True)
    return dict(study=study, table=table, n_rows=n_rows, n_missing_X=n_missing, cached=False)


def load_cache(cache_dir, study, table):
    with np.load(Path(cache_dir) / study / f"{table}.npz") as z:
        return {k: z[k] for k in ("h", "length", "origin", "role")}


def length_summary(lengths):
    """Length statistics over one value per unique sequence."""
    if len(lengths) == 0:
        return dict(len_min=np.nan, len_q25=np.nan, len_median=np.nan, len_q75=np.nan, len_max=np.nan)
    q = np.percentile(lengths, [0, 25, 50, 75, 100])
    return dict(len_min=q[0], len_q25=q[1], len_median=q[2], len_q75=q[3], len_max=q[4])


def sequence_stats(arrays):
    """Unique-sequence counts and length statistics. Returns the statistics and the length of each unique sequence."""
    _, first = np.unique(arrays["h"], return_index=True)
    lengths = arrays["length"][first]
    n_test = len(np.unique(arrays["h"][arrays["role"] == ROLES.index("test")]))
    return dict(n_unique=len(first), n_unique_test=n_test, **length_summary(lengths)), lengths


def label(code, categories):
    return categories[code] if code >= 0 else "unlabeled"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n-workers", type=int, default=4, help="parallel workers in step 1 (each needs up to ~3 GB)")
    parser.add_argument("--overwrite", action="store_true", help="recompute cache files even if they are up to date")
    parser.add_argument("--studies", nargs="*", help="process only these study folders (for testing)")
    args = parser.parse_args()

    tables = list_tables(args.data_dir)
    if args.studies:
        tables = [t for t in tables if t[0] in args.studies]
    print(f"Step 1: {len(tables)} tables, {args.n_workers} workers", flush=True)
    t0 = time.time()
    jobs = [(s, t, p, args.cache_dir, args.overwrite) for s, t, p in tables]
    # maxtasksperchild=1 returns each worker's memory to the system after every table
    with Pool(args.n_workers, maxtasksperchild=1) as pool:
        row_info = pd.DataFrame(pool.map(process_table, jobs, chunksize=1))
    print(f"Step 1 done in {time.time() - t0:.0f} s", flush=True)

    print("Step 2: merging per study", flush=True)
    table_rows, study_rows, origin_rows, length_rows = [], [], [], []
    test_by_origin = {o: [] for o in range(-1, len(ORIGINS))}
    i_test = ROLES.index("test")
    for study, info in row_info.groupby("study", sort=True):
        parts = []
        for rec in info.itertuples():
            arrays = load_cache(args.cache_dir, study, rec.table)
            stats, _ = sequence_stats(arrays)
            table_rows.append(dict(study=study, table=rec.table, n_rows=rec.n_rows, n_missing_X=rec.n_missing_X, **stats))
            parts.append(arrays)
        arrays = concat(parts)
        del parts
        arrays = subset(arrays, unique_rows(arrays))  # now one row per distinct (sequence, origin, role)
        stats, lengths = sequence_stats(arrays)
        study_rows.append(dict(study=study, n_tables=len(info), n_rows=int(info["n_rows"].sum()), **stats))

        # unique sequences per (origin, role); a sequence with several labels is counted once for each label
        pairs, counts = np.unique(np.stack([arrays["origin"], arrays["role"]], axis=1), axis=0, return_counts=True)
        origin_rows += [
            dict(study=study, origin=label(o, ORIGINS), role=label(r, ROLES), n_unique=int(n))
            for (o, r), n in zip(pairs, counts)
        ]

        values, counts = np.unique(lengths, return_counts=True)
        length_rows.append(pd.DataFrame({"study": study, "length": values, "n_unique": counts}))

        is_test = arrays["role"] == i_test
        for o in np.unique(arrays["origin"][is_test]):
            test_by_origin[int(o)].append(arrays["h"][is_test & (arrays["origin"] == o)])  # already unique
        del arrays, lengths, is_test
        print(f"  {study}: {stats['n_unique']:,} unique sequences", flush=True)

    # deduplicate test sequences across studies, separately for each origin
    global_rows = []
    for o, arrays in test_by_origin.items():
        if not arrays:
            continue
        n_sum = sum(len(a) for a in arrays)
        n_unique = len(np.unique(np.concatenate(arrays)))
        global_rows.append(
            dict(origin=label(o, ORIGINS), n_unique_test=n_unique, n_unique_test_summed_over_studies=n_sum, n_studies=len(arrays))
        )
        test_by_origin[o] = None

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table_rows).to_csv(out / "table_stats.csv", index=False)
    pd.DataFrame(study_rows).to_csv(out / "study_stats.csv", index=False)
    pd.DataFrame(origin_rows).to_csv(out / "study_origin_counts.csv", index=False)
    pd.DataFrame(global_rows).sort_values("n_unique_test", ascending=False).to_csv(out / "global_origin_counts.csv", index=False)
    pd.concat(length_rows, ignore_index=True).to_csv(out / "study_length_counts.csv", index=False)
    print(f"Wrote summary files to {out} ({time.time() - t0:.0f} s total)", flush=True)


if __name__ == "__main__":
    main()
