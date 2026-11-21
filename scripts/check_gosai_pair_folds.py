"""External check of graphpart_fold_split.py output on a Gosai MalinoisTraining table.

Checks that the ref and alt sequence of each variant window, and all windows of one variant, are in one fold.
Gosai ids look like chr:pos:ref:alt:{R|A}:window[:extra]; dropping the R/A field pairs ref with alt.

Usage: python scripts/check_gosai_pair_folds.py <table.csv.gz> <folds.csv.gz>
"""
import sys

import pandas as pd

table, folds = sys.argv[1], sys.argv[2]
df = pd.read_csv(table, usecols=['X', 'obs_X: id', 'obs_X: allele'], low_memory=False)
df['X'] = df.X.str.upper()
df = df.merge(pd.read_csv(folds), left_on='X', right_on='sequence', how='left')
assert df.sequence.notna().all(), 'some table sequences are missing from folds.csv.gz'

df = df[df['obs_X: allele'].isin(['Ref', 'Alt'])]
parts = df['obs_X: id'].str.split(':')
df['window_key'] = parts.str[:4].str.join(':') + ':' + parts.str[5:].str.join(':')
df['variant_key'] = parts.str[:4].str.join(':')


def report(key, label):
    groups = df.groupby(key).fold
    has_both = df.groupby(key)['obs_X: allele'].nunique() == 2 if key == 'window_key' else groups.size() > 1
    n_folds = groups.nunique()
    removed = groups.apply(lambda f: f.isna().any())
    print(f'{label}: {has_both.sum()} groups with more than one sequence; '
          f'split across folds: {(n_folds[has_both] > 1).sum()}; '
          f'with a member removed by GraphPart: {removed[has_both].sum()}')
    return n_folds[has_both & (n_folds > 1)].index


print('fold sizes (table rows):', df.fold.value_counts(dropna=False).sort_index().to_dict())
split = report('window_key', 'ref/alt pairs (same window)')
report('variant_key', 'variants (all windows and alleles)')
if len(split):
    print('examples of split ref/alt pairs:')
    print(df[df.window_key.isin(split[:3])][['obs_X: id', 'fold']].to_string())
