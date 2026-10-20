"""
Functions to download MPRA datasets from the MPRACollection repo on HuggingFace.

The repo is organized as `<name_paper>/<name_dataset>.csv` and
`<name_paper>/<name_dataset>.yaml`, the same layout that `MPRA_Dataset.load` reads.
"""

from pathlib import Path
from typing import List, Optional
from huggingface_hub import hf_hub_download, list_repo_files
from .MPRACollection import MPRA_Dataset

DEFAULT_HF_REPO = "simon1727/MPRACollection"


def list_papers(repo_id: str = DEFAULT_HF_REPO) -> List[str]:
    """
    List all papers in the HuggingFace repo.

    Args:
        repo_id: HuggingFace dataset repository ID

    Returns:
        List of paper names (e.g., ["Cell_2015_Seelig", ...])
    """
    files = list_repo_files(repo_id, repo_type="dataset")
    return sorted({f.split("/")[0] for f in files if "/" in f and f.endswith(".csv")})


def list_datasets(name_paper: str, repo_id: str = DEFAULT_HF_REPO) -> List[str]:
    """
    List all datasets of one paper in the HuggingFace repo.

    Args:
        name_paper: Name of the paper (e.g., "Cell_2015_Seelig")
        repo_id: HuggingFace dataset repository ID

    Returns:
        List of dataset names (e.g., ["A3", "A5"])
    """
    files = list_repo_files(repo_id, repo_type="dataset")
    prefix = name_paper + "/"
    return sorted(
        f[len(prefix):-len(".csv")]
        for f in files
        if f.startswith(prefix) and f.endswith(".csv")
    )


def download_dataset(
    name_paper: str,
    name_dataset: str,
    local_dir: Optional[str] = None,
    repo_id: str = DEFAULT_HF_REPO,
    **kwargs,
) -> str:
    """
    Download the .csv and .yaml files of one dataset from HuggingFace.
    Files that are already downloaded and up to date are not downloaded again.

    Args:
        name_paper: Name of the paper (e.g., "Cell_2015_Seelig")
        name_dataset: Name of the dataset (e.g., "A3")
        local_dir: Local folder to save the files into. If None, the files are
            saved in the HuggingFace cache (~/.cache/huggingface/hub by default).
        repo_id: HuggingFace dataset repository ID
        **kwargs: Additional arguments to pass to hf_hub_download

    Returns:
        Local path to the downloaded .csv file
    """
    for ext in ["yaml", "csv"]:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{name_paper}/{name_dataset}.{ext}",
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
        name_paper: Name of the paper (e.g., "Cell_2015_Seelig")
        local_dir: Local folder to save the files into. If None, the files are
            saved in the HuggingFace cache (~/.cache/huggingface/hub by default).
        repo_id: HuggingFace dataset repository ID
        **kwargs: Additional arguments to pass to hf_hub_download

    Returns:
        List of local paths to the downloaded .csv files
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
) -> MPRA_Dataset:
    """
    Download one dataset from HuggingFace (if needed) and load it.

    Args:
        name_paper: Name of the paper (e.g., "Cell_2015_Seelig")
        name_dataset: Name of the dataset (e.g., "A3")
        local_dir: Local folder to save the files into. If None, the files are
            saved in the HuggingFace cache (~/.cache/huggingface/hub by default).
        repo_id: HuggingFace dataset repository ID

    Returns:
        An MPRA_Dataset object
    """
    path = download_dataset(name_paper, name_dataset, local_dir, repo_id)
    # path is <folder>/<name_paper>/<name_dataset>.csv
    folder = Path(path).parents[1]
    return MPRA_Dataset.load(name_paper, name_dataset, folder)
