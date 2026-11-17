"""
Functions to download MPRA datasets (data resources) from the MPRACollection repo on
HuggingFace. PyTorch datasets for training are in `MPRACollection.data`.

The repo is organized as `<name_paper>/<name_dataset>.csv.gz`,
`<name_paper>/<name_dataset>.yaml` and `<name_paper>/study.yaml`, the same layout
that `MPRA_Dataset.load` reads.
"""

from pathlib import Path
from typing import List, Optional
from huggingface_hub import hf_hub_download, list_repo_files
from .MPRACollection import MPRA_Dataset

DEFAULT_HF_REPO = "jnhu323/MPRA_Dataset_Collection"
TABLE_EXT = ".csv.gz"  # extension of the data tables in the repo
STUDY_YAML = "study.yaml"


def list_papers(repo_id: str = DEFAULT_HF_REPO) -> List[str]:
    """
    List all papers in the HuggingFace repo.

    Args:
        repo_id: HuggingFace dataset repository ID

    Returns:
        List of paper names (e.g., ["Cell_2015_Rosenberg", ...])
    """
    files = list_repo_files(repo_id, repo_type="dataset")
    return sorted({f.split("/")[0] for f in files if "/" in f and f.endswith(TABLE_EXT)})


def list_datasets(name_paper: str, repo_id: str = DEFAULT_HF_REPO) -> List[str]:
    """
    List all datasets of one paper in the HuggingFace repo.

    Args:
        name_paper: Name of the paper (e.g., "Cell_2015_Rosenberg")
        repo_id: HuggingFace dataset repository ID

    Returns:
        List of dataset names (e.g., ["A3SS", "A5SS"])
    """
    files = list_repo_files(repo_id, repo_type="dataset")
    prefix = name_paper + "/"
    return sorted(
        f[len(prefix):-len(TABLE_EXT)]
        for f in files
        if f.startswith(prefix) and f.endswith(TABLE_EXT)
    )


def download_dataset(
    name_paper: str,
    name_dataset: str,
    local_dir: Optional[str] = None,
    repo_id: str = DEFAULT_HF_REPO,
    **kwargs,
) -> str:
    """
    Download the .csv.gz and .yaml files of one dataset, and the study.yaml of its
    paper, from HuggingFace.
    Files that are already downloaded and up to date are not downloaded again.

    Args:
        name_paper: Name of the paper (e.g., "Cell_2015_Rosenberg")
        name_dataset: Name of the dataset (e.g., "A3SS")
        local_dir: Local folder to save the files into. If None, the files are
            saved in the HuggingFace cache (~/.cache/huggingface/hub by default).
        repo_id: HuggingFace dataset repository ID
        **kwargs: Additional arguments to pass to hf_hub_download

    Returns:
        Local path to the downloaded .csv.gz file
    """
    # the table is downloaded last, so that `path` is the local path of the table
    for filename in [STUDY_YAML, name_dataset + ".yaml", name_dataset + TABLE_EXT]:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{name_paper}/{filename}",
            repo_type="dataset",
            local_dir=local_dir,
            **kwargs,
        )
    return path


def download_paper(
    name_paper: str,
    local_dir: Optional[str] = None,
    repo_id: str = DEFAULT_HF_REPO,
    **kwargs,
) -> List[str]:
    """
    Download all datasets of one paper from HuggingFace.

    Args:
        name_paper: Name of the paper (e.g., "Cell_2015_Rosenberg")
        local_dir: Local folder to save the files into. If None, the files are
            saved in the HuggingFace cache (~/.cache/huggingface/hub by default).
        repo_id: HuggingFace dataset repository ID
        **kwargs: Additional arguments to pass to hf_hub_download

    Returns:
        List of local paths to the downloaded .csv.gz files
    """
    return [
        download_dataset(name_paper, name_dataset, local_dir, repo_id, **kwargs)
        for name_dataset in list_datasets(name_paper, repo_id)
    ]


def load_dataset(
    name_paper: str,
    name_dataset: str,
    local_dir: Optional[str] = None,
    repo_id: str = DEFAULT_HF_REPO,
    usecols=None,
) -> MPRA_Dataset:
    """
    Download one dataset from HuggingFace (if needed) and load it.

    Args:
        name_paper: Name of the paper (e.g., "Cell_2015_Rosenberg")
        name_dataset: Name of the dataset (e.g., "A3SS")
        local_dir: Local folder to save the files into. If None, the files are
            saved in the HuggingFace cache (~/.cache/huggingface/hub by default).
        repo_id: HuggingFace dataset repository ID
        usecols: columns to read, passed to pandas.read_csv (see MPRA_Dataset.load)

    Returns:
        An MPRA_Dataset object
    """
    path = download_dataset(name_paper, name_dataset, local_dir, repo_id)
    # path is <folder>/<name_paper>/<name_dataset>.csv.gz
    folder = Path(path).parents[1]
    return MPRA_Dataset.load(name_paper, name_dataset, folder, usecols=usecols)
