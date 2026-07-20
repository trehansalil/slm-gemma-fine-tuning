"""Merge QLoRA adapter into base Gemma 2 2B model for export/conversion."""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_ADAPTER_DIR = "models/sft_local/adapter"
DEFAULT_OUTPUT_DIR = "models/sft_local/merged"
LOCAL_BASE_DIR = "models/base"
HF_BASE_MODEL = "google/gemma-2-2b"


def resolve_base(base_arg: str | None) -> str:
    if base_arg:
        return base_arg
    if os.path.isdir(LOCAL_BASE_DIR) and os.listdir(LOCAL_BASE_DIR):
        return LOCAL_BASE_DIR
    return HF_BASE_MODEL


def merge(adapter_dir: str, output_dir: str, base: str | None = None):
    base_path = resolve_base(base)
    print(f"Loading base model: {base_path}...")
    model_base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float32,
    )

    print(f"Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(model_base, adapter_dir)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.save_pretrained(output_dir)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Done! Merged model: {n_params/1e6:.1f}M params saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge QLoRA adapter into base Gemma 2 2B")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER_DIR,
                        help="Path to adapter directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for merged model")
    parser.add_argument("--base", default=None,
                        help="Base model path or HF ID (default: models/base/ if exists, else google/gemma-2-2b)")
    args = parser.parse_args()

    merge(args.adapter, args.output, args.base)


if __name__ == "__main__":
    main()
