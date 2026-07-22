"""DPO training for the SLM 125M QA model.

Full-parameter fine-tuning (no LoRA needed for a 125M model).
Uses the same preference dataset as the Gemma 2B DPO pipeline.

Usage:
    python -m slm125m.train_dpo
    python -m slm125m.train_dpo --model models/slm125m/base --epochs 2
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def load_preference_data(path: str) -> Dataset:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            examples.append({
                "prompt": ex["prompt"],
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
            })
    return Dataset.from_list(examples)


def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="DPO training for SLM 125M")
    parser.add_argument("--model", type=str, default="models/slm125m/base")
    parser.add_argument("--data", type=str, default="data/preference_train.jsonl")
    parser.add_argument("--output", type=str, default="models/slm125m/dpo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    attn_impl = "eager" if device.type == "mps" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation=attn_impl,
    )

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} (full fine-tuning)")

    dataset = load_preference_data(args.data)
    print(f"Loaded {len(dataset)} preference pairs")

    training_args = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        precompute_ref_log_probs=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        report_to="none",
        optim="adamw_torch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting DPO training: {args.epochs} epoch(s), "
          f"batch={args.batch_size}x{args.grad_accum}, lr={args.lr}, beta={args.beta}")
    trainer.train()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nDPO model saved to {args.output}")


if __name__ == "__main__":
    main()
