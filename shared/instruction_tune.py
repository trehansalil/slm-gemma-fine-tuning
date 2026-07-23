"""Generic instruction tuning, auto-selecting the training mode by model size.

Full-parameter fine-tuning for small models (<500M params, e.g. SLM 125M),
QLoRA for larger ones (same LoraConfig/BitsAndBytesConfig recipe as
gemma2b/finetune_local.py — bnb 4-bit on CUDA, fp32 + LoRA on MPS/CPU).

Conservative continued-fine-tuning defaults so Stage 01 QA knowledge is
preserved: 2 epochs, LR 1e-5 (full) / 2e-5 (QLoRA).

Usage:
    python -m shared.instruction_tune --model models/slm125m/base \
        --data data/instruction_train.jsonl --output models/slm125m/instruction
    python -m shared.instruction_tune --model models/sft_local/merged \
        --data data/instruction_train.jsonl --output models/instruction_local/adapter
"""
from __future__ import annotations

import argparse
import json

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from shared.train_utils import (MAX_SEQ_LEN, detect_chat_format, get_attn_impl,
                                get_device, get_dtype, maybe_compile,
                                tokenize_examples, train)

FULL_FINETUNE_MAX_PARAMS = 500_000_000

MODE_DEFAULTS = {
    # continued fine-tuning: conservative LRs, small batch + grad accum on MPS
    "full": {"lr": 1e-5, "batch_size": 8, "grad_accum": 2},
    "qlora": {"lr": 2e-5, "batch_size": 1, "grad_accum": 8},
}


def count_params(model_path: str) -> int:
    """Parameter count from config alone (meta device — no weights loaded)."""
    cfg = AutoConfig.from_pretrained(model_path)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    n = sum(p.numel() for p in model.parameters())
    del model
    return n


def load_examples(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Instruction-tune a model (auto full-parameter vs QLoRA)")
    parser.add_argument("--model", required=True, help="Base checkpoint path or HF ID")
    parser.add_argument("--data", required=True, help="Chat-messages JSONL training data")
    parser.add_argument("--output", required=True, help="Output dir (full model or adapter)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None,
                        help="Override mode default (full: 1e-5, qlora: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log train_loss to stdout + metrics.jsonl every N optimizer steps")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Run val_loss+perplexity + best-checkpoint every N steps (0=only per-epoch)")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    n_params = count_params(args.model)
    mode = "full" if n_params < FULL_FINETUNE_MAX_PARAMS else "qlora"
    defaults = MODE_DEFAULTS[mode]
    lr = args.lr if args.lr is not None else defaults["lr"]
    batch_size = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    grad_accum = args.grad_accum if args.grad_accum is not None else defaults["grad_accum"]
    print(f"Model: {args.model} ({n_params / 1e6:.0f}M params) -> {mode} mode | "
          f"lr={lr}, batch={batch_size}x{grad_accum}, epochs={args.epochs}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        chat_format = detect_chat_format(tokenizer)
    except ValueError as e:
        raise SystemExit(f"{args.model}: {e}")
    print(f"Chat format: {chat_format}")

    dtype = get_dtype(device)
    attn_impl = get_attn_impl(device)

    if mode == "full":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
        model = maybe_compile(model, device)
    else:
        use_bnb = device.type == "cuda"
        if use_bnb:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
            )

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model = model.to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    examples = load_examples(args.data)
    print(f"Loaded {len(examples)} training examples from {args.data}")
    features = tokenize_examples(examples, tokenizer, seq_len=args.seq_len)

    best = train(model, tokenizer, features, args.output, device,
                 epochs=args.epochs, lr=lr, batch_size=batch_size,
                 grad_accum=grad_accum, seed=args.seed, log_every=args.log_every,
                 eval_every=args.eval_every)
    print(f"\nDone ({mode} mode). Best val_loss={best:.4f}. "
          f"Saved to {args.output}")


if __name__ == "__main__":
    main()
