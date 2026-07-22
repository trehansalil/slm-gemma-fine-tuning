"""RLAIF / DPO preference-optimisation stage.

Takes an instruction-tuned checkpoint and runs DPO (Direct Preference
Optimisation) on AI-labelled preference data (RLAIF).  Auto-selects
full-parameter fine-tuning for small models (<500M params) and QLoRA for
larger ones — same strategy as shared/instruction_tune.py.

Pipeline position:  SFT -> Instruction Tuning -> DPO -> **RLAIF**
The instruction-tuned model is the *policy* starting point and also
serves as the implicit reference model (TRL's DPOTrainer copies the
policy weights internally for the ref model, or you can pass a frozen
ref explicitly via --ref-model).

Usage:
    # SLM 125M (full fine-tuning, ~125M params)
    python -m shared.rlaif_train \
        --model models/slm125m/instruction \
        --data  data/preference_train.jsonl \
        --output models/slm125m/rlaif

    # Gemma 2B (QLoRA)
    python -m shared.rlaif_train \
        --model models/instruction_local/adapter \
        --data  data/preference_train.jsonl \
        --output models/rlaif_local/adapter
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from trl import DPOConfig, DPOTrainer

from shared.train_utils import MAX_SEQ_LEN, get_device

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FULL_FINETUNE_MAX_PARAMS = 500_000_000

MODE_DEFAULTS = {
    "full":  {"lr": 5e-7, "batch_size": 4, "grad_accum": 4},
    "qlora": {"lr": 1e-6, "batch_size": 1, "grad_accum": 8},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params(model_path: str) -> int:
    """Parameter count from config alone (meta device -- no weights loaded)."""
    cfg = AutoConfig.from_pretrained(model_path)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    n = sum(p.numel() for p in model.parameters())
    del model
    return n


def load_preference_data(path: str) -> Dataset:
    """Load a JSONL file with prompt / chosen / rejected fields into a
    HuggingFace Dataset suitable for DPOTrainer."""
    examples: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            examples.append({
                "prompt": ex["prompt"],
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
            })
    return Dataset.from_list(examples)


def _open_metrics_log(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    return open(os.path.join(output_dir, "metrics.jsonl"), "a", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RLAIF / DPO preference optimisation (auto full-parameter vs QLoRA)")
    parser.add_argument("--model", required=True,
                        help="Instruction-tuned checkpoint path or HF ID (policy model)")
    parser.add_argument("--ref-model", default=None,
                        help="Explicit reference model path (default: copy of --model)")
    parser.add_argument("--data", required=True,
                        help="Preference JSONL with prompt/chosen/rejected fields")
    parser.add_argument("--output", required=True,
                        help="Output dir for trained model / adapter")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None,
                        help="Override mode default (full: 5e-7, qlora: 1e-6)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta / implicit reward temperature")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LEN,
                        help="Max sequence length for DPO pairs")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", type=float, default=0.05,
                        help="Fraction of data reserved for eval (0 to disable)")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Logging interval in steps")
    parser.add_argument("--save-every-epoch", action="store_true",
                        help="Save a checkpoint after every epoch")
    parser.add_argument("--precompute-ref", action="store_true", default=True,
                        help="Precompute reference log-probs (saves memory)")
    parser.add_argument("--no-precompute-ref", dest="precompute_ref",
                        action="store_false")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # ----- Decide mode (full vs qlora) ------------------------------------
    n_params = count_params(args.model)
    mode = "full" if n_params < FULL_FINETUNE_MAX_PARAMS else "qlora"
    defaults = MODE_DEFAULTS[mode]
    lr = args.lr if args.lr is not None else defaults["lr"]
    batch_size = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    grad_accum = args.grad_accum if args.grad_accum is not None else defaults["grad_accum"]
    print(f"Model: {args.model} ({n_params / 1e6:.0f}M params) -> {mode} mode | "
          f"lr={lr}, batch={batch_size}x{grad_accum}, epochs={args.epochs}, beta={args.beta}")

    # ----- Tokenizer -------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO left-pads for generation alignment

    attn_impl = "eager" if device.type == "mps" else None

    # ----- Load policy model -----------------------------------------------
    if mode == "full":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            attn_implementation=attn_impl,
        )
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
                torch_dtype=torch.float32,
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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ----- Load optional explicit reference model --------------------------
    ref_model = None
    if args.ref_model is not None:
        print(f"Loading explicit reference model from {args.ref_model}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model,
            torch_dtype=torch.float32,
            attn_implementation=attn_impl,
        )
        for p in ref_model.parameters():
            p.requires_grad = False

    # ----- Dataset ---------------------------------------------------------
    dataset = load_preference_data(args.data)
    print(f"Loaded {len(dataset)} preference pairs from {args.data}")

    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None

    # ----- DPO training config ---------------------------------------------
    save_strategy = "epoch" if args.save_every_epoch else "no"

    training_args = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=args.beta,
        max_length=args.max_length,
        precompute_ref_log_probs=args.precompute_ref,
        logging_steps=args.log_every,
        save_strategy=save_strategy,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        gradient_checkpointing=mode == "qlora",
        report_to="none",
        optim="adamw_torch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ----- Train -----------------------------------------------------------
    print(f"\nStarting RLAIF/DPO training: {args.epochs} epoch(s), "
          f"batch={batch_size}x{grad_accum}, lr={lr}, beta={args.beta}")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    # ----- Save final model ------------------------------------------------
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nModel saved to {args.output}")

    # ----- Write metrics.jsonl (match instruction_tune.py convention) ------
    metrics = result.metrics
    metrics["elapsed_minutes"] = round(elapsed / 60, 1)
    metrics["mode"] = mode
    metrics["beta"] = args.beta
    metrics["lr"] = lr
    metrics["epochs"] = args.epochs

    # Run final eval if we have an eval set
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        metrics.update(eval_metrics)

    mf = _open_metrics_log(args.output)
    mf.write(json.dumps(metrics) + "\n")
    mf.close()

    train_loss = metrics.get("train_loss", float("nan"))
    eval_loss = metrics.get("eval_loss", float("nan"))
    print(f"\nDone ({mode} mode, {elapsed / 60:.1f}min). "
          f"train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")
    print(f"Metrics written to {os.path.join(args.output, 'metrics.jsonl')}")


if __name__ == "__main__":
    main()
