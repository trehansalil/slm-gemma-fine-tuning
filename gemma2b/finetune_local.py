"""Local Gemma 2 2B QLoRA fine-tuning on Mac (MPS) or CPU.

Tokenizes locally, then runs QLoRA training with the same loss-masking
approach as the Modal pipeline. Downloads base model from HuggingFace on
first run. Resumes from the latest local checkpoint automatically.

Usage:
    python -m gemma2b.finetune_local                    # 3 epochs, auto device
    python -m gemma2b.finetune_local --n-epochs 10
    python -m gemma2b.finetune_local --lr 1e-4
    python -m gemma2b.finetune_local --device cpu
    python -m gemma2b.finetune_local --fresh             # ignore checkpoints

Requires: pip install torch transformers peft bitsandbytes accelerate numpy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from gemma2b import config

IGNORE_INDEX = config.IGNORE_INDEX


def _tokenize_chat(messages, tokenizer):
    """Tokenize a chat with explicit loss masking for Gemma 2 format."""
    start_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
    end_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    nl_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    input_ids = [tokenizer.bos_token_id]
    labels = [IGNORE_INDEX]

    system_content = None
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_content = content
            continue

        if role == "user":
            gemma_role = "user"
            if system_content:
                content = f"{system_content}\n\n{content}"
                system_content = None
        else:
            gemma_role = "model"

        role_ids = tokenizer.encode(gemma_role, add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False)

        turn_ids = [start_id] + role_ids + [nl_id] + content_ids + [end_id, nl_id]

        if msg["role"] == "assistant":
            header_len = 1 + len(role_ids) + 1
            turn_labels = [IGNORE_INDEX] * header_len + content_ids + [end_id, nl_id]
        else:
            turn_labels = [IGNORE_INDEX] * len(turn_ids)

        input_ids.extend(turn_ids)
        labels.extend(turn_labels)

    return input_ids, labels


def tokenize_locally(tokenizer, seq_len=1024):
    """Tokenize SFT data locally, save as numpy arrays."""
    sft_path = os.path.join(os.path.dirname(__file__), "..", "data", "sft_train.jsonl")
    if not os.path.exists(sft_path):
        sft_path = os.path.expanduser(
            "~/Documents/Python_n_R/Personal/slm-engineering/data/sft_train.jsonl"
        )
    if not os.path.exists(sft_path):
        raise FileNotFoundError(f"SFT data not found. Place sft_train.jsonl in data/")

    with open(sft_path, encoding="utf-8") as fh:
        examples = [json.loads(line) for line in fh]
    print(f"Tokenizing {len(examples)} examples (seq_len={seq_len})...")

    pad_id = tokenizer.pad_token_id
    all_input_ids, all_labels = [], []
    skipped = 0

    for ex in examples:
        input_ids, labels_list = _tokenize_chat(ex["messages"], tokenizer)

        if len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]
            labels_list = labels_list[:seq_len]

        trainable = sum(1 for l in labels_list if l != IGNORE_INDEX)
        if trainable < 5:
            skipped += 1
            continue

        pad_len = seq_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels_list = labels_list + [IGNORE_INDEX] * pad_len

        all_input_ids.append(input_ids)
        all_labels.append(labels_list)

    if not all_input_ids:
        raise ValueError(f"No valid examples after tokenization ({len(examples)} input)")

    os.makedirs(config.LOCAL_TOKENS_DIR, exist_ok=True)
    np.save(f"{config.LOCAL_TOKENS_DIR}/input_ids.npy", np.array(all_input_ids, dtype=np.int32))
    np.save(f"{config.LOCAL_TOKENS_DIR}/labels.npy", np.array(all_labels, dtype=np.int32))
    print(f"Tokenized {len(all_input_ids)} examples (skipped {skipped})")


class SFTDataset(Dataset):
    def __init__(self, input_ids, labels, pad_id=0):
        self.examples = []
        for ids_row, lab_row in zip(input_ids, labels):
            mask = ids_row != pad_id
            length = int(mask.sum())
            if length < 2:
                continue
            self.examples.append((
                ids_row[:length].astype(np.int64),
                lab_row[:length].astype(np.int64),
            ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, labs = self.examples[i]
        return torch.from_numpy(ids), torch.from_numpy(labs)


def collate_fn(batch, pad_id=0):
    """Dynamic padding — pad to longest in batch, not to max_seq_len."""
    ids_list, _ = zip(*batch)
    max_len = max(x.size(0) for x in ids_list)
    padded_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_labs = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (ids, labs) in enumerate(batch):
        padded_ids[i, :ids.size(0)] = ids
        padded_labs[i, :labs.size(0)] = labs
        attn_mask[i, :ids.size(0)] = 1
    return padded_ids, padded_labs, attn_mask


def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, val_dl, device):
    model.eval()
    total_loss = total_count = 0
    with torch.no_grad():
        for ids, labs, mask in val_dl:
            ids, labs, mask = ids.to(device), labs.to(device), mask.to(device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss
            total_loss += loss.item() * ids.size(0)
            total_count += ids.size(0)
    model.train()
    return total_loss / max(total_count, 1)


def find_latest_checkpoint():
    if not os.path.exists(config.LOCAL_CKPT_DIR):
        return None, 0
    ckpts = [d for d in os.listdir(config.LOCAL_CKPT_DIR) if d.startswith("epoch_")]
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda x: int(x.split("_")[1]))
    latest = ckpts[-1]
    epoch_num = int(latest.split("_")[1])
    return f"{config.LOCAL_CKPT_DIR}/{latest}", epoch_num


def save_checkpoint(model, optimizer, epoch, step, val_loss, ppl, tokenizer):
    ckpt_path = f"{config.LOCAL_CKPT_DIR}/epoch_{epoch:02d}"
    os.makedirs(ckpt_path, exist_ok=True)
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "ppl": ppl,
    }, f"{ckpt_path}/training_state.pt")
    print(f"  [ckpt] saved epoch {epoch} to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Local Gemma 2B QLoRA fine-tuning")
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every-epochs", type=int, default=1)
    parser.add_argument("--retokenize", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--base", type=str, default="models/base",
                        help="Path to pre-downloaded base model (default: models/base)")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    base_model_path = args.base
    print(f"Base model: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.retokenize and os.path.exists(config.LOCAL_TOKENS_DIR):
        shutil.rmtree(config.LOCAL_TOKENS_DIR)
        print("Cleared old tokenized data")

    ids_path = f"{config.LOCAL_TOKENS_DIR}/input_ids.npy"
    if not os.path.exists(ids_path):
        tokenize_locally(tokenizer, seq_len=config.MAX_SEQ_LEN)

    input_ids = np.load(f"{config.LOCAL_TOKENS_DIR}/input_ids.npy")
    labels = np.load(f"{config.LOCAL_TOKENS_DIR}/labels.npy")
    print(f"Loaded {len(input_ids)} tokenized examples")

    pad_id = tokenizer.pad_token_id
    full_ds = SFTDataset(input_ids, labels, pad_id=pad_id)
    val_size = max(100, int(len(full_ds) * 0.05))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {train_size}, Val: {val_size}")

    start_epoch = 0
    resume_path = args.resume

    if resume_path is None and not args.fresh:
        latest_ckpt, latest_epoch = find_latest_checkpoint()
        if latest_ckpt:
            resume_path = latest_ckpt
            start_epoch = latest_epoch

    use_bnb = device.type == "cuda"
    attn_impl = "eager" if device.type == "mps" else "sdpa"

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        if use_bnb:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float32,
                attn_implementation=attn_impl,
            )
        base_model = prepare_model_for_kbit_training(base_model) if use_bnb else base_model
        model = PeftModel.from_pretrained(base_model, resume_path, is_trainable=True)
        state_path = f"{resume_path}/training_state.pt"
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            start_epoch = state["epoch"]
            print(f"  Resuming after epoch {start_epoch} "
                  f"(val_loss={state.get('val_loss', '?')}, ppl={state.get('ppl', '?')})")
    else:
        print(f"Loading {base_model_path}...")
        if use_bnb:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
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

    model = model.to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.2f}%)")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=False, drop_last=True,
                          collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=0, pin_memory=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    if resume_path and os.path.exists(f"{resume_path}/training_state.pt"):
        opt_state = torch.load(f"{resume_path}/training_state.pt",
                               map_location="cpu", weights_only=False)
        if "optimizer" in opt_state:
            optimizer.load_state_dict(opt_state["optimizer"])
            print("  Restored optimizer state")

    total_epochs = args.n_epochs
    remaining_epochs = total_epochs - start_epoch
    if remaining_epochs <= 0:
        print(f"Already trained {start_epoch} epochs (target {total_epochs}). Nothing to do.")
        return

    steps_per_epoch = len(train_dl) // args.grad_accum
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = min(100, total_steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * step / max(1, warmup_steps)
        ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * ratio))

    print(f"\nTraining: epochs {start_epoch + 1} -> {total_epochs}, "
          f"{total_steps} steps, batch={args.batch_size}x{args.grad_accum}, lr={args.lr}")

    os.makedirs(os.path.dirname(config.LOCAL_METRICS_PATH), exist_ok=True)
    mf = open(config.LOCAL_METRICS_PATH, "a")
    ppl_log_path = f"{config.LOCAL_MODEL_DIR}/perplexity_log.jsonl"
    ppl_log = open(ppl_log_path, "a")

    init_vl = evaluate(model, val_dl, device)
    init_ppl = math.exp(min(init_vl, 20))
    print(f"Starting val_loss={init_vl:.4f} ppl={init_ppl:.2f}")
    ppl_log.write(json.dumps({
        "epoch": start_epoch, "val_loss": round(init_vl, 4),
        "ppl": round(init_ppl, 2), "phase": "start",
    }) + "\n")
    ppl_log.flush()

    step = 0
    micro = 0
    running_loss = 0.0
    best_val_loss = init_vl
    patience_counter = 0
    t0 = t_global = time.time()

    if resume_path and os.path.exists(f"{resume_path}/training_state.pt"):
        ckpt_state = torch.load(f"{resume_path}/training_state.pt",
                                map_location="cpu", weights_only=False)
        step = ckpt_state.get("step", 0)

    for epoch_idx in range(remaining_epochs):
        current_epoch = start_epoch + epoch_idx + 1
        epoch_train_loss = 0.0
        epoch_train_steps = 0

        for ids, labs, mask in train_dl:
            ids, labs, mask = ids.to(device), labs.to(device), mask.to(device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss / args.grad_accum
            loss.backward()
            running_loss += loss.item()
            epoch_train_loss += loss.item()
            micro += 1

            if micro % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            current_lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            epoch_train_steps += 1

            if step % args.log_every == 0:
                avg = running_loss / args.log_every
                vl_step = evaluate(model, val_dl, device)
                elapsed = time.time() - t_global
                sec_per_step = (time.time() - t0) / args.log_every
                print(f"step {step:>5}/{total_steps} | train_loss {avg:.4f} | "
                      f"val_loss {vl_step:.4f} | "
                      f"lr {current_lr:.2e} | {sec_per_step:.2f}s/step | "
                      f"{elapsed / 60:.1f}min")
                mf.write(json.dumps({"epoch": current_epoch, "step": step,
                                     "train_loss": round(avg, 4),
                                     "val_loss": round(vl_step, 4),
                                     "lr": current_lr}) + "\n")
                mf.flush()
                running_loss = 0.0
                t0 = time.time()

        avg_train_loss = epoch_train_loss / max(1, epoch_train_steps)
        vl = evaluate(model, val_dl, device)
        ppl = math.exp(min(vl, 20))
        elapsed = time.time() - t_global
        improving = "+" if vl < best_val_loss else "-"
        print(f"  [epoch {current_epoch}] train_loss={avg_train_loss:.4f} "
              f"val_loss={vl:.4f} ppl={ppl:.2f} "
              f"{improving} | {elapsed / 60:.1f}min")

        ppl_log.write(json.dumps({
            "epoch": current_epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(vl, 4),
            "ppl": round(ppl, 2), "improving": vl < best_val_loss,
        }) + "\n")
        ppl_log.flush()

        saved_ckpt = False
        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            os.makedirs(config.LOCAL_OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(config.LOCAL_OUTPUT_DIR)
            tokenizer.save_pretrained(config.LOCAL_OUTPUT_DIR)
            save_checkpoint(model, optimizer, current_epoch, step, vl, ppl, tokenizer)
            saved_ckpt = True
            print(f"  [best] saved to {config.LOCAL_OUTPUT_DIR} + checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  [early stop] no improvement for {args.patience} epochs")
                break

        if current_epoch % args.ckpt_every_epochs == 0 and not saved_ckpt:
            save_checkpoint(model, optimizer, current_epoch, step, vl, ppl, tokenizer)

    final_dir = f"{config.LOCAL_OUTPUT_DIR}_final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    final_vl = evaluate(model, val_dl, device)
    final_ppl = math.exp(min(final_vl, 20))
    ppl_log.write(json.dumps({
        "epoch": total_epochs, "val_loss": round(final_vl, 4),
        "ppl": round(final_ppl, 2), "phase": "final",
    }) + "\n")
    mf.close()
    ppl_log.close()
    elapsed = time.time() - t_global

    print(f"\nDone! {step} steps in {elapsed / 60:.1f}min")
    print(f"Final val_loss={final_vl:.4f} ppl={final_ppl:.2f}")
    print(f"Best val_loss={best_val_loss:.4f} ppl={math.exp(min(best_val_loss, 20)):.2f}")
    print(f"\nCheckpoints: {config.LOCAL_CKPT_DIR}/")
    print(f"Best model: {config.LOCAL_OUTPUT_DIR}/")
    print(f"Perplexity log: {ppl_log_path}")


if __name__ == "__main__":
    main()
