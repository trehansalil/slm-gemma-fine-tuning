"""Download the SLM 125M QA model from HuggingFace.

Usage:
    python -m slm125m.download_model
    python -m slm125m.download_model --output models/slm125m/base
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


MODEL_ID = "thesreedath/slm-125m-qa"


def main():
    parser = argparse.ArgumentParser(description="Download SLM 125M QA model")
    parser.add_argument("--output", type=str, default="models/slm125m/base")
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    os.makedirs(args.output, exist_ok=True)
    print(f"Downloading {MODEL_ID} → {args.output}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=args.output,
        token=token,
    )
    print(f"Done. Model saved to {args.output}")


if __name__ == "__main__":
    main()
