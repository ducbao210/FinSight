from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

DEFAULT_REPO_URL = "https://github.com/patronus-ai/financebench.git"
DEFAULT_DATA_DIR = "data"
DEFAULT_PDF_DIR = "pdfs"


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command))

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def download_dataset(
    repo_url: str,
    output_dir: Path,
    data_dir: str,
    pdf_dir: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dataset_") as temp_dir:
        temp_path = Path(temp_dir)
        repo_path = temp_path / "repository"

        # Clone the repository without checking out all files.
        run_command(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-checkout",
                repo_url,
                str(repo_path),
            ]
        )

        # Enable sparse checkout to download only the required directories.
        run_command(
            [
                "git",
                "sparse-checkout",
                "init",
                "--cone",
            ],
            cwd=repo_path,
        )

        run_command(
            [
                "git",
                "sparse-checkout",
                "set",
                data_dir,
                pdf_dir,
            ],
            cwd=repo_path,
        )

        run_command(
            ["git", "checkout"],
            cwd=repo_path,
        )

        source_data = repo_path / data_dir
        source_pdfs = repo_path / pdf_dir

        target_pdfs = output_dir / pdf_dir

        # Copy JSONL files into the output directory.
        for jsonl_file in source_data.glob("*.jsonl"):
            destination = output_dir / jsonl_file.name
            shutil.copy2(jsonl_file, destination)
            print(f"Copied: {destination}")

        # Copy the PDF directory into the output directory.
        shutil.copytree(
            source_pdfs,
            target_pdfs,
            dirs_exist_ok=True,
        )

        print("\nDataset downloaded successfully.")
        print(f"Dataset location: {output_dir}")
        print(f"PDF location: {target_pdfs}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a dataset and its PDF files."
    )

    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO_URL,
        help="Git repository URL.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW,
        help="Directory where the dataset will be stored.",
    )

    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing the dataset metadata files.",
    )

    parser.add_argument(
        "--pdf-dir",
        default=DEFAULT_PDF_DIR,
        help="Directory containing the PDF files.",
    )

    args = parser.parse_args()

    download_dataset(
        repo_url=args.repo_url,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        pdf_dir=args.pdf_dir,
    )


if __name__ == "__main__":
    main()
