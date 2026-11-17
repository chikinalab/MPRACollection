"""Sample MPRA sequences and compute their HyenaDNA embeddings.

The notebook `figures/hyenadna_emb.ipynb` plots the output of this script.

Stage `sample` (CPU): draws a uniform random sample of `--n-per-study` unique test sequences (rows with
`obs_X: control role = test`) from every study. Each sequence is given a 64-bit hash
(`pandas.util.hash_pandas_object` with a key derived from `--seed`), and the sequences with the smallest hashes are kept.
The hash works as a fixed random number per distinct sequence, so this is a uniform sample of the study's unique test
sequences without repeats. Each table is streamed in chunks and only the current smallest hashes are kept in memory, so
the largest tables (about 100 million rows) do not need to fit in memory. A sequence found in several tables keeps the
table and sequence origin of its first occurrence.

Stage `embed` (GPU): embeds the sampled sequences with HyenaDNA and averages the last hidden layer over the
nucleotides of each sequence (padding excluded), giving one 256-dimensional vector per sequence.
  --pooling mean      plain average (output embeddings.npy).
  --pooling baseline  before averaging, subtract the mean hidden state at each position, estimated from random
                      sequences with equal base frequencies (output embeddings_baseline.npy). With plain averaging the
                      embedding depends strongly on sequence length, because the mean hidden state changes with the
                      position in the sequence; the subtraction removes this dependence.

Output files in `--out-dir`:
    sampled_sequences.csv    one row per sampled sequence: study, table, sequence, length, origin
    embeddings.npy           float32 array (n_sequences, 256); row i belongs to row i of sampled_sequences.csv
    embeddings_baseline.npy  same, with the position baseline subtracted before averaging (--pooling baseline)
    position_baseline.npy    float32 array (max sequence length, 256): the subtracted baseline

Usage (from the repository root):
    sbatch sh/hyenadna_sample.sh   # python scripts/compute_hyenadna_embeddings.py --stage sample --n-workers 4
    sbatch sh/hyenadna_embed.sh [baseline]  # python scripts/compute_hyenadna_embeddings.py --stage embed [--pooling baseline]
"""

import argparse
import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = "/projects/bjgn/jhu30/MPRACollection_data/reviewed"
OUT_DIR = str(Path(__file__).resolve().parents[1] / "res" / "hyenadna_emb")
SAMPLE_FILE = "sampled_sequences.csv"
# embedding file for each pooling method (--pooling)
EMB_FILES = {"mean": "embeddings.npy", "baseline": "embeddings_baseline.npy"}
BASELINE_FILE = "position_baseline.npy"

COL_X = "X"
COL_ORIGIN = "obs_X: sequence origin"
COL_ROLE = "obs_X: control role"
CHUNK_ROWS = 2_000_000

CHECKPOINT = "LongSafari/hyenadna-large-1m-seqlen-hf"
REVISION = "0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446"


def list_tables(data_dir):
    """Return (study, table, path) for every data table, largest file first (as in compute_collection_stats.py)."""
    tables = []
    for study in sorted(os.listdir(data_dir)):
        study_dir = Path(data_dir) / study
        if not study_dir.is_dir():
            continue
        for path in sorted(study_dir.glob("*.csv.gz")):
            tables.append((study, path.name[: -len(".csv.gz")], str(path)))
    tables.sort(key=lambda t: os.path.getsize(t[2]), reverse=True)
    return tables


def smallest_hashes(df, k):
    """Rows of the k smallest distinct hashes; for a repeated hash the first row is kept."""
    return df.drop_duplicates("hash", keep="first").nsmallest(k, "hash", keep="first")


def sample_table(job, k, hash_key):
    """Candidates from one table: its k test sequences with the smallest hashes."""
    study, table, path = job
    t0 = time.time()
    best = None
    for chunk in pd.read_csv(path, usecols=[COL_X, COL_ORIGIN, COL_ROLE], dtype=str, chunksize=CHUNK_ROWS):
        chunk = chunk[(chunk[COL_ROLE].str.strip() == "test") & chunk[COL_X].notna()]
        if chunk.empty:
            continue
        df = pd.DataFrame({
            "sequence": chunk[COL_X].to_numpy(),
            "origin": chunk[COL_ORIGIN].str.strip().fillna("unlabeled").to_numpy(),
        })
        df["hash"] = pd.util.hash_pandas_object(df["sequence"], index=False, hash_key=hash_key).to_numpy(np.uint64)
        df = smallest_hashes(df, k)
        best = df if best is None else smallest_hashes(pd.concat([best, df], ignore_index=True), k)
    if best is None:
        best = pd.DataFrame(columns=["sequence", "origin", "hash"])
    best.insert(0, "table", table)
    best.insert(0, "study", study)
    print(f"  {study}/{table}: {len(best)} candidates, {time.time() - t0:.0f} s", flush=True)
    return best


def run_sample(args):
    tables = list_tables(args.data_dir)
    if args.studies:
        tables = [t for t in tables if t[0] in args.studies]
    # hash_pandas_object needs a 16-character key; different seeds give independent samples
    hash_key = f"{args.seed:016d}"[-16:]
    print(f"Sampling {args.n_per_study} test sequences per study from {len(tables)} tables, "
          f"{args.n_workers} workers", flush=True)
    t0 = time.time()
    worker = partial(sample_table, k=args.n_per_study, hash_key=hash_key)
    with Pool(args.n_workers, maxtasksperchild=1) as pool:
        parts = pool.map(worker, tables, chunksize=1)

    # sort by (study, table) so that "first occurrence" of a sequence repeated in several tables does not depend on
    # the order in which the workers finished
    # tables without test sequences give empty parts, which are dropped
    cand = pd.concat([p for p in parts if len(p)], ignore_index=True).sort_values(["study", "table"], kind="stable")
    cand = cand.drop_duplicates(["study", "hash"], keep="first")
    sample = cand.sort_values(["study", "hash"], kind="stable").groupby("study").head(args.n_per_study)
    sample["length"] = sample["sequence"].str.len()
    sample = sample[["study", "table", "sequence", "length", "origin"]].reset_index(drop=True)

    counts = sample["study"].value_counts()
    short = counts[counts < args.n_per_study]
    if len(short):
        print("Studies with fewer sequences than requested:\n" + short.to_string())
    out = Path(args.out_dir) / SAMPLE_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"Wrote {len(sample)} sequences from {sample['study'].nunique()} studies to {out} "
          f"({time.time() - t0:.0f} s)", flush=True)


def run_embed(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    out_dir = Path(args.out_dir)
    sample = pd.read_csv(out_dir / SAMPLE_FILE)
    seqs = sample["sequence"].str.upper().tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(seqs)} sequences, pooling '{args.pooling}', device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, revision=REVISION, trust_remote_code=True)
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(CHECKPOINT, revision=REVISION, trust_remote_code=True,
                                      torch_dtype=torch.float32).to(device).eval()

    def hidden_states(batch_seqs):
        """Last hidden layer (batch, padded length, d_model) and the mask of real nucleotides (batch, padded length)."""
        batch = tokenizer(batch_seqs, add_special_tokens=False, padding=True, truncation=False,
                          return_attention_mask=True, return_tensors="pt")
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device).bool()
        with torch.inference_mode():
            # this model does not accept attention_mask; it is causal, so right padding does not affect real tokens
            return model(input_ids=input_ids, return_dict=True).last_hidden_state, mask

    baseline = None
    if args.pooling == "baseline":
        baseline = position_baseline(hidden_states, max(len(s) for s in seqs), args)
        np.save(out_dir / BASELINE_FILE, baseline.cpu().numpy())
        print(f"Wrote position baseline of shape {tuple(baseline.shape)} to {out_dir / BASELINE_FILE}", flush=True)

    # sorting by length keeps sequences of similar length in one batch, which reduces padding
    order = np.argsort([len(s) for s in seqs], kind="stable")
    emb = np.zeros((len(seqs), model.config.d_model), dtype=np.float32)
    t0 = time.time()
    for start in range(0, len(seqs), args.batch_size):
        idx = order[start:start + args.batch_size]
        hidden, mask = hidden_states([seqs[i] for i in idx])
        with torch.inference_mode():
            if baseline is not None:
                hidden = hidden - baseline[: hidden.shape[1]]
            hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
            pooled = hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        emb[idx] = pooled.float().cpu().numpy()
    print(f"Embedded in {time.time() - t0:.0f} s", flush=True)

    assert np.isfinite(emb).all(), "non-finite values in embeddings"
    out = out_dir / EMB_FILES[args.pooling]
    np.save(out, emb)
    print(f"Wrote embeddings of shape {emb.shape} to {out}", flush=True)


def position_baseline(hidden_states, length, args):
    """Mean hidden state at each position over random sequences, shape (length, d_model).

    The mean hidden state of HyenaDNA changes with the position in the sequence over several hundred nucleotides, so
    plain mean pooling gives vectors that depend on sequence length. Subtracting this position-specific mean before
    pooling removes that dependence. Because the model is causal, the hidden state at position p depends only on
    positions 0..p, so random sequences of the maximum length give the baseline for every shorter length too.
    """
    import torch

    rng = np.random.default_rng(args.seed)
    total, t0 = None, time.time()
    for start in range(0, args.n_baseline, args.batch_size):
        n = min(args.batch_size, args.n_baseline - start)
        codes = rng.integers(0, 4, size=(n, length))
        batch = ["".join(np.array(list("ACGT"))[row]) for row in codes]
        hidden, _ = hidden_states(batch)
        s = hidden.sum(dim=0)
        total = s if total is None else total + s
    print(f"Position baseline from {args.n_baseline} random sequences of {length} nt "
          f"(equal base frequencies), {time.time() - t0:.0f} s", flush=True)
    return total / args.n_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["sample", "embed"], required=True)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n-per-study", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=4, help="parallel workers in the sample stage")
    parser.add_argument("--studies", nargs="*", help="sample only these study folders (for testing)")
    parser.add_argument("--batch-size", type=int, default=64, help="sequences per batch in the embed stage")
    parser.add_argument("--pooling", choices=["mean", "baseline"], default="mean",
                        help="embed stage: 'mean' averages the last hidden layer over nucleotides; 'baseline' first "
                             "subtracts the mean hidden state at each position over random sequences")
    parser.add_argument("--n-baseline", type=int, default=1024,
                        help="number of random sequences used to estimate the position baseline")
    args = parser.parse_args()
    if args.stage == "sample":
        run_sample(args)
    else:
        run_embed(args)


if __name__ == "__main__":
    main()
