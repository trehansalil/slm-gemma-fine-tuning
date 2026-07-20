"""Upload SFT training data to Modal volume for Gemma 2 2B fine-tuning.

Usage:
    modal run gemma2b/upload_data.py
"""
from __future__ import annotations

import os
import subprocess

import modal

PROJECT = "gemma2-2b-sft"
VOLUME_NAME = "gemma2-2b-sft"

app = modal.App(PROJECT)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

SFT_LOCAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "sft_train.jsonl"
)
FALLBACK_PATH = os.path.expanduser(
    "~/Documents/Python_n_R/Personal/slm-engineering/data/sft_train.jsonl"
)


@app.local_entrypoint()
def main():
    src = SFT_LOCAL_PATH if os.path.exists(SFT_LOCAL_PATH) else FALLBACK_PATH
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"SFT data not found at {SFT_LOCAL_PATH} or {FALLBACK_PATH}"
        )

    print(f"Uploading {src} to volume '{VOLUME_NAME}'...")
    subprocess.run(
        ["modal", "volume", "put", VOLUME_NAME, src, "sft_train.jsonl"],
        check=True,
    )
    print("Done.")
