"""RAFT training for the SLM 125M DPO model.

Full-parameter DPO fine-tuning on 18k RAFT examples, starting from the
DPO-aligned model. Same optimizations as train_dpo.py: precomputed ref
log probs, fp16 on MPS, cosine LR with warmup, early stopping.

Usage:
    python -m slm125m.train_raft
    python -m slm125m.train_raft --model models/slm125m/dpo --epochs 2
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import DPOConfig, DPOTrainer

from shared.train_utils import get_attn_impl, get_device


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


def main():
    parser = argparse.ArgumentParser(description="RAFT training for SLM 125M (on DPO model)")
    parser.add_argument("--model", type=str, default="models/slm125m/dpo")
    parser.add_argument("--data", type=str, default="data/raft_train.jsonl")
    parser.add_argument("--output", type=str, default="models/slm125m/raft_dpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--eval-split", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation=get_attn_impl(device),
    )

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} (full fine-tuning)")

    full_dataset = load_preference_data(args.data)
    split = full_dataset.train_test_split(test_size=args.eval_split, seed=42)
    dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Loaded {len(full_dataset)} RAFT preference pairs "
          f"(train={len(dataset)}, eval={len(eval_dataset)})")

    use_mps = device.type == "mps"
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
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        remove_unused_columns=False,
        bf16=(device.type == "cuda"),
        fp16=use_mps,
        gradient_checkpointing=False,
        report_to="none",
        optim="adamw_torch",
        dataloader_num_workers=0 if use_mps else 2,
        dataloader_prefetch_factor=None if use_mps else 2,
        dataloader_pin_memory=not use_mps,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"\nStarting RAFT training: {args.epochs} epoch(s), "
          f"batch={args.batch_size}x{args.grad_accum}, lr={args.lr}, beta={args.beta}")
    trainer.train()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nRAFT-DPO model saved to {args.output}")


if __name__ == "__main__":
    main()
