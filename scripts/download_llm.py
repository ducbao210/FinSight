"""Download a GGUF evaluation LLM from Hugging Face.

Example:
    python scripts/download_llm.py \
        --repo-id <org>/<repo> \
        --filename gpt-oss-20b-Q4_K_M.gguf

The script intentionally requires the repository ID so it cannot silently
 download an incorrect or unexpectedly large model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REPO_ID = "unsloth/gpt-oss-20b-GGUF"
DEFAULT_FILENAME = "gpt-oss-20b-Q4_K_M.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a GGUF LLM used by scripts/run_eval.py"
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repository (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help=f"Exact GGUF filename (default: {DEFAULT_FILENAME})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models",
        help="Directory where the model will be stored",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face branch, tag, or commit hash",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token; defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload the file even if it already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / Path(args.filename).name

    if destination.exists() and not args.force:
        print(f"LLM already exists; skipping download: {destination}")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install huggingface_hub with "
            "'pip install huggingface_hub'"
        ) from exc

    token = (
        args.token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or None
    )
    print(f"Downloading LLM: {args.repo_id}/{args.filename}", flush=True)
    if args.revision:
        print(f"Revision: {args.revision}", flush=True)

    cached_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        token=token,
        local_dir=str(output_dir),
    )

    cached = Path(cached_path)
    if cached.resolve() != destination.resolve() and cached.exists():
        cached.replace(destination)

    if not destination.exists():
        raise SystemExit(f"Download finished but file was not found: {destination}")

    size_gb = destination.stat().st_size / (1024**3)
    print(f"LLM downloaded: {destination}")
    print(f"Size: {size_gb:.2f} GiB")
    print("Use it with:")
    print(f"  --backend gguf --model {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
