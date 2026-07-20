"""Download base Gemma 2 2B model from HuggingFace with parallel progress."""

import argparse
import os
import time

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "google/gemma-2-2b"
DEFAULT_OUTPUT_DIR = "models/base"
DEFAULT_WORKERS = 8


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def enable_hf_transfer():
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        return True
    except ImportError:
        return False


def download(output_dir: str, workers: int = DEFAULT_WORKERS):
    os.makedirs(output_dir, exist_ok=True)

    has_hf_transfer = enable_hf_transfer()

    print(f"Downloading {BASE_MODEL} -> {output_dir}/")
    print(f"  Workers: {workers}")
    print(f"  hf_transfer: {'ON (Rust accelerator)' if has_hf_transfer else 'OFF (pip install hf_transfer for 3-5x speedup)'}")
    print()

    start = time.time()

    snapshot_download(
        BASE_MODEL,
        local_dir=output_dir,
        max_workers=workers,
    )

    elapsed = time.time() - start
    print(f"\nDownload complete in {format_time(elapsed)}")

    print("Loading model to verify parameter count...")
    model = AutoModelForCausalLM.from_pretrained(
        output_dir, torch_dtype=torch.float32
    )
    AutoTokenizer.from_pretrained(output_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Verified: {n_params / 1e6:.1f}M params in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download base Gemma 2 2B")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel download threads")
    args = parser.parse_args()
    download(args.output, args.workers)
