"""Console script for MPRACollection.

    mdc list_dataset
    mdc download_dataset [DATA_DIR] --paper PAPER [--dataset DATASET] [--overwrite]
    mdc download_dataset [DATA_DIR] --all [--overwrite]
    mdc load_dataset [DATA_DIR] --paper PAPER --dataset DATASET

If DATA_DIR is not given, files are saved in the HuggingFace cache.
"""
from typing import List, Optional

import typer
from huggingface_hub.errors import EntryNotFoundError
from rich.console import Console
from rich.table import Table

from MPRACollection import datasets

app = typer.Typer(
    help="Download and load MPRA datasets from the MPRACollection repo on HuggingFace.",
    no_args_is_help=True,
)
console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

DATA_DIR_HELP = (
    "Local folder to save the files into. If not given, the files are saved in "
    "the HuggingFace cache (~/.cache/huggingface/hub by default)."
)
DATA_DIR_ARGUMENT = typer.Argument(
    None, metavar="[DATA_DIR]", help=DATA_DIR_HELP, show_default=False
)
PAPER_HELP = "Name of the paper, e.g. eLife_2020_King. See `mdc list_dataset`."
DATASET_HELP = "Name of the dataset in the paper, e.g. MiniSpacer. Needs --paper."


def exit_not_found(name: str):
    err_console.print(
        f"[red]Error:[/red] {name} is not in the MPRACollection repo. "
        "Run `mdc list_dataset` to see all papers and datasets."
    )
    raise typer.Exit(code=1)


def print_downloaded(paths: List[str]):
    for path in paths:
        console.print(f"Downloaded {path}", soft_wrap=True)


@app.command("list_dataset")
def list_dataset():
    """List all papers and their datasets in the MPRACollection repo."""
    table = Table("Paper", "Datasets", title=f"HuggingFace: {datasets.DEFAULT_HF_REPO}")
    for name_paper in datasets.list_papers():
        table.add_row(name_paper, ", ".join(datasets.list_datasets(name_paper)))
    console.print(table)
    console.print(
        "Download with: mdc download_dataset [DATA_DIR] -p <paper> [-d <dataset>]", markup=False
    )


@app.command("download_dataset")
def download_dataset(
    data_dir: Optional[str] = DATA_DIR_ARGUMENT,
    name_paper: Optional[str] = typer.Option(
        None, "--paper", "-p", metavar="PAPER",
        help=PAPER_HELP + " Without --dataset, all datasets of the paper are downloaded.",
    ),
    name_dataset: Optional[str] = typer.Option(
        None, "--dataset", "-d", metavar="DATASET", help=DATASET_HELP
    ),
    download_all: bool = typer.Option(
        False, "--all", help="Download all datasets of all papers (~7.7 GB)."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Download again even if the files already exist."
    ),
):
    """Download one dataset, all datasets of one paper, or all datasets of all papers."""
    if name_dataset is not None and name_paper is None:
        raise typer.BadParameter("needs --paper.", param_hint="'--dataset'")
    if download_all and name_paper is not None:
        raise typer.BadParameter(
            "downloads all papers, so it can't be used with --paper.", param_hint="'--all'"
        )
    if not download_all and name_paper is None:
        raise typer.BadParameter(
            "missing, give a paper, or use --all to download all papers.", param_hint="'--paper'"
        )
    kwargs = {"force_download": True} if overwrite else {}

    if name_dataset is not None:
        try:
            paths = [datasets.download_dataset(name_paper, name_dataset, data_dir, **kwargs)]
        except EntryNotFoundError:
            exit_not_found(f"Dataset '{name_dataset}' of paper '{name_paper}'")
        print_downloaded(paths)
        return

    # all datasets of one paper, or of every paper with --all
    for name_paper in datasets.list_papers() if download_all else [name_paper]:
        paths = datasets.download_paper(name_paper, data_dir, **kwargs)
        if not paths:
            exit_not_found(f"Paper '{name_paper}'")
        print_downloaded(paths)


@app.command("load_dataset")
def load_dataset(
    data_dir: Optional[str] = DATA_DIR_ARGUMENT,
    name_paper: str = typer.Option(..., "--paper", "-p", metavar="PAPER", help=PAPER_HELP),
    name_dataset: str = typer.Option(..., "--dataset", "-d", metavar="DATASET", help=DATASET_HELP),
):
    """Load one dataset (downloading it first if needed) and print a summary."""
    try:
        mpra_dataset = datasets.load_dataset(name_paper, name_dataset, data_dir)
    except EntryNotFoundError:
        exit_not_found(f"Dataset '{name_dataset}' of paper '{name_paper}'")

    console.print(f"Loaded from {mpra_dataset.folder}", soft_wrap=True)
    console.print(str(mpra_dataset), markup=False, soft_wrap=True)


if __name__ == "__main__":
    app()
