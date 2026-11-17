"""Compute AlphaGenome encoder embeddings for the sampled MPRA sequences.

The sequences are the ones sampled by `scripts/compute_hyenadna_embeddings.py --stage sample`
(`res/hyenadna_emb/sampled_sequences.csv`), so the two embeddings can be compared row by row.

Model: the pretrained AlphaGenome convolutional encoder, loaded by
`MPRACollection.model.alphagenome.load_pretrained_encoder` (see that module for details).

Embedding: each sequence is one-hot encoded in ACGT order (N and other bases become all-zero rows) and passed through
the encoder at its own length, without padding. The encoder output has 1536 channels and one position per 128 bp
(ceil(length / 128) positions; the last position also covers the zero padding the pooling layers add at the end).
These positions are averaged, giving one 1536-dimensional vector per sequence. Sequences are batched only with
sequences of the same length, so the result for a sequence does not depend on the other sequences in its batch.

Output files in `--out-dir`:
    embeddings.npy  float32 array (n_sequences, 1536); row i belongs to row i of the sample file

Usage (from the repository root):
    sbatch sh/ag_encoder_embed.sh   # python scripts/compute_ag_encoder_emb.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alphagenome_pytorch.utils.sequence import sequence_to_onehot
from MPRACollection.model.alphagenome import REPO_ID, WEIGHTS_FILE, load_pretrained_encoder

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = str(ROOT / "res" / "hyenadna_emb" / "sampled_sequences.csv")
OUT_DIR = str(ROOT / "res" / "ag_encoder_emb")
EMB_FILE = "embeddings.npy"


def run_embed(args):
    sample = pd.read_csv(args.sample_file)
    seqs = sample["sequence"].str.upper().tolist()
    lengths = np.array([len(s) for s in seqs])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(seqs)} sequences ({len(np.unique(lengths))} distinct lengths, {lengths.min()}-{lengths.max()} nt), "
          f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    encoder = load_pretrained_encoder(args.weights, device=device)
    n_params = sum(t.numel() for t in encoder.parameters())
    print(f"Loaded {REPO_ID}/{args.weights} encoder ({n_params / 1e6:.1f}M parameters)", flush=True)

    emb = None
    t0 = time.time()
    for length in np.unique(lengths):
        same_length = np.flatnonzero(lengths == length)
        for start in range(0, len(same_length), args.batch_size):
            idx = same_length[start:start + args.batch_size]
            x = torch.from_numpy(np.stack([sequence_to_onehot(seqs[i], dtype=np.float32) for i in idx])).to(device)
            with torch.inference_mode():
                trunk, _ = encoder(x)  # (batch, 1536, ceil(length / 128))
                pooled = trunk.mean(dim=-1)
            if emb is None:
                emb = np.zeros((len(seqs), pooled.shape[1]), dtype=np.float32)
            emb[idx] = pooled.float().cpu().numpy()
    print(f"Embedded in {time.time() - t0:.0f} s", flush=True)

    assert np.isfinite(emb).all(), "non-finite values in embeddings"
    out = Path(args.out_dir) / EMB_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    print(f"Wrote embeddings of shape {emb.shape} to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-file", default=SAMPLE_FILE, help="CSV with a 'sequence' column")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--weights", default=WEIGHTS_FILE, help=f"weights file in the {REPO_ID} repository")
    parser.add_argument("--batch-size", type=int, default=128, help="sequences per batch")
    run_embed(parser.parse_args())


if __name__ == "__main__":
    main()
