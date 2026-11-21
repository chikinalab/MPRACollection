"""Assign 5 similarity-aware folds to one MPRACollection data table with GraphPart (plain mmseqs2 mode).

Several tables that measured (almost) the same library, for example the time points of one study, can be given
together: GraphPart then runs once on the union of their sequences, and each table gets the folds of its own
sequences in folds_<table name>.csv.gz.

Steps
1. Read the unique sequences of the table(s) (sequence column from each table's .yaml, default 'X').
2. Reverse-complement collapse: a sequence and its exact reverse complement form one group. Only the first
   one in the table is given to GraphPart; the other gets the same fold afterwards.
3. Run `graphpart mmseqs2` with thresholds 0.5, 0.6, 0.7 and select the smallest threshold whose folds are
   even (every fold size within 10% of the mean fold size).
   Fixed GraphPart settings: --denominator shortest (identical positions / length of the shorter sequence),
   5 partitions. MMseqs2 uses all CPUs available to the job.

Run modes
  default            run the thresholds in increasing order, stop at the first even one, then select.
  --threshold T      run only threshold T (for example one Slurm array task per threshold); writes t<T>/ only.
  --select           run no GraphPart; select from the t<T>/result.json files written by --threshold runs.

Outputs in --out:
  folds.csv.gz     sequence, fold (empty = removed by GraphPart); written only if a threshold gave even folds
  folds_<table name>.csv.gz
                   only when several tables are given: the rows of folds.csv.gz for that table's sequences
  summary.json     settings, per-threshold fold sizes, removed counts, run time, selected threshold
                   (and per-table fold sizes when several tables are given)
  t<threshold>/    per threshold: graphpart_input.fasta, GraphPart files (gp.csv, edges.csv, log.txt),
                   result.json, folds.csv.gz (folds at this threshold, even or not)

Usage:
  python scripts/graphpart_fold_split.py \
      /projects/bjgn/jhu30/MPRACollection_data/reviewed/Nature_2024_Gosai/MalinoisTraining_K562.csv.gz \
      --out /work/hdd/bjgn/jhu30/graphpart_folds/Nature_2024_Gosai/MalinoisTraining_K562
  python scripts/graphpart_fold_split.py $DATA/NatureCommunications_2022_Kreimer/{0h,3h,6h}.csv.gz \
      --out /work/hdd/bjgn/jhu30/graphpart_folds/NatureCommunications_2022_Kreimer/union
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd
import yaml

THRESHOLDS = [0.5, 0.6, 0.7]
N_FOLDS = 5
MAX_FOLD_DEV = 0.1  # each fold must be within 10% of the mean fold size
GRAPHPART_ARGS = ['--denominator', 'shortest', '--nucleotide', '--partitions', str(N_FOLDS)]

# graphpart and mmseqs live in the same conda env as this Python
ENV_BIN = os.path.dirname(sys.executable)
os.environ['PATH'] = ENV_BIN + os.pathsep + os.environ['PATH']


def revcomp(seq):
    return seq.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]


def table_name(table):
    return os.path.basename(table).replace('.csv.gz', '')


def load_sequences(table):
    """Unique upper-case sequences of the table's primary sequence column, in table order."""
    column = 'X'
    meta_file = table.replace('.csv.gz', '.yaml')
    if os.path.exists(meta_file):
        column = yaml.safe_load(open(meta_file))['Primary Sequence Columns'][0]
    seqs = pd.read_csv(table, usecols=[column])[column].dropna().str.upper()
    return seqs.drop_duplicates().tolist()


def load_union(tables):
    """Unique sequences of all tables, in order of first appearance."""
    return list(dict.fromkeys(seq for table in tables for seq in load_sequences(table)))


def collapse_reverse_complements(seqs):
    """Table with one row per sequence: id, sequence, rep_id (the id of the copy given to GraphPart)."""
    df = pd.DataFrame({'id': [f's{i:07d}' for i in range(len(seqs))], 'sequence': seqs})
    df['canonical'] = [min(s, revcomp(s)) for s in seqs]
    df['rep_id'] = df.groupby('canonical').id.transform('first')
    return df.drop(columns='canonical')


def run_graphpart(fasta, threshold, workdir):
    """Run GraphPart in its own folder (it writes temporary files to the working directory).
    Returns {sequence id: fold} or None if GraphPart failed."""
    cmd = ['graphpart', 'mmseqs2', '--fasta-file', os.path.abspath(fasta), '--threshold', str(threshold),
           '--out-file', 'gp.csv', '--save-checkpoint-path', 'edges.csv'] + GRAPHPART_ARGS
    with open(os.path.join(workdir, 'log.txt'), 'w') as log:
        result = subprocess.run(cmd, cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    out_file = os.path.join(workdir, 'gp.csv')
    if result.returncode != 0 or not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
        return None
    gp = pd.read_csv(out_file)
    return dict(zip(gp.AC, gp.cluster.astype(int)))


def run_threshold(seqs, threshold, out):
    """Run GraphPart at one threshold in out/t<threshold>/ and write result.json and folds.csv.gz there.
    Each threshold has its own folder and FASTA file, so several thresholds can run at the same time."""
    workdir = os.path.join(out, f't{threshold}')
    os.makedirs(workdir, exist_ok=True)
    reps = seqs[seqs.id == seqs.rep_id]
    fasta = os.path.join(workdir, 'graphpart_input.fasta')
    with open(fasta, 'w') as f:
        for seq_id, seq in zip(reps.id, reps.sequence):
            f.write(f'>{seq_id}\n{seq}\n')

    start = time.time()
    rep_fold = run_graphpart(fasta, threshold, workdir)
    result = {'minutes': round((time.time() - start) / 60, 1)}
    if rep_fold is None:
        result['error'] = f'GraphPart failed, see t{threshold}/log.txt'
    else:
        folds = seqs.rep_id.map(rep_fold).astype('Int64')
        result.update(fold_stats(folds))
        pd.DataFrame({'sequence': seqs.sequence, 'fold': folds}).to_csv(
            os.path.join(workdir, 'folds.csv.gz'), index=False)
    with open(os.path.join(workdir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=1)
    print(f'threshold {threshold}: {result}', flush=True)
    return result


def fold_stats(folds):
    sizes = folds.value_counts().reindex(range(N_FOLDS), fill_value=0)
    deviation = float((sizes - sizes.mean()).abs().max() / sizes.mean())
    return {'n_sequences': len(folds), 'fold_sizes': sizes.tolist(), 'n_removed': int(folds.isna().sum()),
            'max_deviation_from_mean': round(deviation, 4), 'even': deviation <= MAX_FOLD_DEV}


def write_table_folds(tables, final_folds, out):
    """For each table, write the folds of its own sequences to out/folds_<table name>.csv.gz."""
    union = pd.read_csv(final_folds, dtype={'fold': 'Int64'}).set_index('sequence').fold
    stats = {}
    for table in tables:
        seqs = pd.Series(load_sequences(table))
        folds = seqs.map(union).astype('Int64')
        pd.DataFrame({'sequence': seqs, 'fold': folds}).to_csv(
            os.path.join(out, f'folds_{table_name(table)}.csv.gz'), index=False)
        stats[table_name(table)] = fold_stats(folds)
        print(f'{table_name(table)}: {stats[table_name(table)]}')
    return stats


def select(seqs, tables, out):
    """Pick the smallest threshold with even folds from out/t<threshold>/result.json; write summary.json and,
    if a threshold was selected, copy its folds to out/folds.csv.gz (and per-table files for several tables)."""
    summary = {'tables': tables, 'mode': 'mmseqs2', 'graphpart_args': GRAPHPART_ARGS,
               'max_fold_dev': MAX_FOLD_DEV, 'n_unique_sequences': len(seqs),
               'n_given_to_graphpart': int((seqs.id == seqs.rep_id).sum()),
               'thresholds_tried': {}, 'selected_threshold': None}
    for threshold in THRESHOLDS:
        result_file = os.path.join(out, f't{threshold}', 'result.json')
        if not os.path.exists(result_file):
            continue
        result = json.load(open(result_file))
        summary['thresholds_tried'][str(threshold)] = result
        if result.get('even') and summary['selected_threshold'] is None:
            summary['selected_threshold'] = threshold

    final_folds = os.path.join(out, 'folds.csv.gz')
    old_files = [final_folds] + [os.path.join(out, f'folds_{table_name(t)}.csv.gz') for t in tables]
    for f in old_files:  # left over from an earlier run with different results
        if os.path.exists(f):
            os.remove(f)
    if summary['selected_threshold'] is None:
        print('No threshold gave even folds; folds.csv.gz not written.')
    else:
        shutil.copyfile(os.path.join(out, f"t{summary['selected_threshold']}", 'folds.csv.gz'), final_folds)
        print(f"Selected threshold {summary['selected_threshold']}")
        if len(tables) > 1:
            summary['per_table'] = write_table_folds(tables, final_folds, out)
    missing = [str(t) for t in THRESHOLDS if str(t) not in summary['thresholds_tried']]
    if missing:
        print(f'No result.json for thresholds: {", ".join(missing)}')
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tables', nargs='+',
                        help='path to a .csv.gz MPRA data table; with several tables, the union of their '
                             'sequences is split')
    parser.add_argument('--out', required=True, help='output folder')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--threshold', type=float, choices=THRESHOLDS,
                      help='run only this threshold; combine the thresholds afterwards with --select')
    mode.add_argument('--select', action='store_true',
                      help='run no GraphPart; select from existing t<threshold>/result.json files')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seqs = collapse_reverse_complements(load_union(args.tables))
    print(f'{len(seqs)} unique sequences, {(seqs.id == seqs.rep_id).sum()} after reverse-complement collapse')

    if args.threshold is not None:
        run_threshold(seqs, args.threshold, args.out)
        return
    if not args.select:
        for threshold in THRESHOLDS:
            if run_threshold(seqs, threshold, args.out).get('even'):
                break
    select(seqs, args.tables, args.out)


if __name__ == '__main__':
    main()
