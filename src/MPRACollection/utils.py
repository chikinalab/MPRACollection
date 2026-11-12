import os
import textwrap

import pandas as pd
import numpy as np

STUDY_YAML = "study.yaml"

# fields shown under "Study information" when printing an MPRA_Dataset
STUDY_FIELDS = (
    "Description",
    "MPRA Technique",
    "Readout Assay",
    "Readout Value Space",
    "Regulatory Element",
)
LEN_MAX_VALUE = 100  # printed metadata values longer than this are cut off with "..."


def _get_field(name, *dicts):
    """Return the first value found for `name` in `dicts`.
    A value stored as {'Name': ..., 'Source Details': ...} is reduced to its 'Name'."""
    for d in dicts:
        if isinstance(d, dict) and d.get(name) is not None:
            value = d[name]
            return value["Name"] if isinstance(value, dict) and "Name" in value else value
    return None


def _shorten(value, len_max=LEN_MAX_VALUE):
    value = " ".join(str(value).split())  # yaml block strings may contain newlines
    return value if len(value) <= len_max else value[: len_max - 3] + "..."


def format_metadata(info, info_study, folder=None, name_paper="", name_dataset=""):
    """Return the metadata part of MPRA_Dataset.__str__: the 'Study information',
    'Table information' and 'Full metadata' sections.

    info: contents of <name_dataset>.yaml; info_study: contents of study.yaml.
    """
    table_metadata = info.get("Table Metadata") or {}
    shared_metadata = info_study.get("Shared Metadata") or {}

    # Study information (study.yaml, falling back to the table yaml)
    lines_study = []
    for name in STUDY_FIELDS:
        value = _get_field(name, info_study, shared_metadata, table_metadata, info)
        if value is None:
            continue
        if name == "Description":  # wrap the full description instead of cutting it off
            lines_study.append(textwrap.fill(
                " ".join(str(value).split()), width=100,
                initial_indent=f"    {name}: ", subsequent_indent=" " * 8,
            ))
        else:
            lines_study.append(f"    {name}: {_shorten(value)}")

    # Table information (<name_dataset>.yaml)
    lines_table = []
    n_passing = info.get("number of unique_inputs_passing_available_filters")
    if n_passing is not None:
        lines_table.append(f"    Number of unique inputs passing available filters: {n_passing}")
    len_min, len_max = info.get("sequence-length-min_nt"), info.get("sequence-length-max_nt")
    if len_min is not None and len_max is not None:
        lines_table.append(f"    Sequence length: {len_min}-{len_max} nt")
    if table_metadata:
        lines_table.append("    Table Metadata:")
        lines_table += [f"        {key}: {_shorten(value)}" for key, value in table_metadata.items()]

    lines = []
    if lines_study:
        lines += ["Study information:"] + lines_study
    if lines_table:
        lines += ["Table information:"] + lines_table

    # Point to the full metadata
    if folder is not None and name_paper and name_dataset:
        folder_paper = os.path.join(folder, name_paper)
        lines += [
            "Full metadata:",
            f"    study: .info_study ({os.path.join(folder_paper, STUDY_YAML)})",
            f"    table: .info ({os.path.join(folder_paper, name_dataset + '.yaml')})",
        ]
    else:
        lines.append("Full metadata: .info_study (study.yaml) and .info (table yaml)")
    return "".join(line + "\n" for line in lines)


def strip_prefix(col, prefix):
    """Remove `prefix` (e.g. "Y: ") from the start of a column name, if present."""
    return col[len(prefix):] if col.startswith(prefix) else col


def data_to_XYobs(data):
    cols_X = ["X"] if "X" in data.columns else []
    # a readout column is either named "Y: <name>", or just "Y" when a table has one readout
    cols_Y = [col for col in data.columns if col == "Y" or col.startswith("Y: ")]
    cols_obs_X = [col for col in data.columns if col.startswith("obs_X: ")]
    cols_obs_Y = [col for col in data.columns if col.startswith("obs_Y: ")]
    X, Y, obs_X, obs_Y = data[cols_X], data[cols_Y], data[cols_obs_X], data[cols_obs_Y]

    # rename columns to be without prefix
    Y = Y.rename(columns=lambda col: strip_prefix(col, "Y: "))
    obs_X = obs_X.rename(columns=lambda col: strip_prefix(col, "obs_X: "))
    obs_Y = obs_Y.rename(columns=lambda col: strip_prefix(col, "obs_Y: "))
    return X, Y, obs_X, obs_Y


def XYobs_to_data(X, Y, obs_X, obs_Y):
    # add prefix to columns; a bare "Y" column stays "Y", as in the source files
    Y = Y.rename(columns=lambda col: col if col == "Y" else "Y: " + col)
    obs_X = obs_X.rename(columns=lambda col: "obs_X: " + col)
    obs_Y = obs_Y.rename(columns=lambda col: "obs_Y: " + col)
    data = pd.concat([X, Y, obs_X, obs_Y], axis=1)
    return data


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def seqs_to_onehot(
    seqs,
    len_max=64,
    len_div=1,
):
    len_max = (len_max + len_div - 1) // len_div * len_div

    # Function to adjust sequence length to len_max
    def to_len_max(seq):
        if len(seq) < len_max:
            # Pad sequence if it's shorter than len_max
            padding = len_max - len(seq)
            return seq + "N" * padding
        else:
            # Truncate sequence if it's longer than len_max
            return seq[:len_max]

    seqs = [to_len_max(seq) for seq in seqs]

    alphabet = "ATGC"
    alphabet_unknown = "NX"
    value_unknown = 1.0 / len(alphabet)

    def to_uint8(seq):
        return np.frombuffer(seq.encode("ascii"), dtype=np.uint8)

    hash_table = np.zeros((np.iinfo(np.uint8).max + 1, len(alphabet)))
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet))
    hash_table[to_uint8(alphabet_unknown)] = value_unknown

    def seq_to_onehot(seq):
        return hash_table[to_uint8(seq)]

    seqs = [seq_to_onehot(seq) for seq in seqs]

    return np.stack(seqs)
