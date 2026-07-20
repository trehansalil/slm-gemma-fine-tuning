"""Gemma 2 2B QLoRA fine-tuning on Modal (8x A100).

Two-step pipeline:
  1. Tokenize — pre-tokenize sft_train.jsonl with Gemma's native tokenizer
     and explicit loss masking (only assistant tokens contribute to loss).
  2. Train — QLoRA 4-bit fine-tuning across 8 GPUs via accelerate.

Usage:
    modal run gemma2b/finetune_modal.py                # tokenize + train (1 epoch)
    modal run gemma2b/finetune_modal.py --n-epochs 3
    modal run gemma2b/finetune_modal.py --skip-tokenize
"""
from __future__ import annotations

import modal

PROJECT = "gemma2-2b-sft"
VOLUME_NAME = "gemma2-2b-sft"
DATA_ROOT = "/data"
SFT_DATA_PATH = f"{DATA_ROOT}/sft_train.jsonl"
TOKENIZED_DIR = f"{DATA_ROOT}/tokenized"
FINAL_MODEL_DIR = f"{DATA_ROOT}/final"
BASE_MODEL = "google/gemma-2-2b"
HF_SECRET_NAME = "huggingface-secret"
MAX_SEQ_LEN = 1024
IGNORE_INDEX = -100
NUM_GPUS = 8
GPU_TYPE = "A100"

app = modal.App(PROJECT)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3",
        "transformers>=4.42",
        "datasets",
        "peft>=0.11",
        "bitsandbytes>=0.43",
        "accelerate>=0.30",
        "numpy",
        "huggingface_hub",
    )
)


def _tokenize_chat(messages, tokenizer):
    """Tokenize a chat with explicit loss masking.

    Gemma 2 format: <start_of_turn>role\ncontent<end_of_turn>\n
    System content is folded into the first user turn (Gemma has no system role).
    Only assistant content + <end_of_turn> contribute to loss.
    """
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


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu=GPU_TYPE,
    timeout=1800,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def tokenize():
    """Pre-tokenize SFT data with loss masking, save as numpy arrays."""
    import json
    import os

    import numpy as np
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(SFT_DATA_PATH) as f:
        examples = [json.loads(line) for line in f]

    print(f"Tokenizing {len(examples)} examples (max_seq_len={MAX_SEQ_LEN})...")

    all_input_ids = []
    all_labels = []
    skipped = 0

    for ex in examples:
        input_ids, labels_list = _tokenize_chat(ex["messages"], tokenizer)

        if len(input_ids) > MAX_SEQ_LEN:
            input_ids = input_ids[:MAX_SEQ_LEN]
            labels_list = labels_list[:MAX_SEQ_LEN]
        else:
            pad_len = MAX_SEQ_LEN - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels_list = labels_list + [IGNORE_INDEX] * pad_len

        trainable = sum(1 for l in labels_list if l != IGNORE_INDEX)
        if trainable == 0:
            skipped += 1
            continue

        all_input_ids.append(input_ids)
        all_labels.append(labels_list)

    input_ids_arr = np.array(all_input_ids, dtype=np.int32)
    labels_arr = np.array(all_labels, dtype=np.int32)

    os.makedirs(TOKENIZED_DIR, exist_ok=True)
    np.save(f"{TOKENIZED_DIR}/input_ids.npy", input_ids_arr)
    np.save(f"{TOKENIZED_DIR}/labels.npy", labels_arr)
    volume.commit()

    print(f"Saved {len(all_input_ids)} examples ({skipped} skipped).")
    print(f"  input_ids shape: {input_ids_arr.shape}")
    print(f"  labels shape:    {labels_arr.shape}")

    sample_labels = labels_arr[0]
    trainable_count = int((sample_labels != IGNORE_INDEX).sum())
    print(f"  Sample: {trainable_count}/{MAX_SEQ_LEN} tokens trainable")

    return len(all_input_ids)


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu=f"{GPU_TYPE}:{NUM_GPUS}",
    timeout=3600,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def train(n_epochs: int = 1, lr: float = 2e-4, batch_size: int = 4, grad_accum: int = 2):
    """QLoRA fine-tuning on pre-tokenized data across multiple GPUs."""
    import math
    import os
    import time

    import numpy as np
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader, Dataset, random_split
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision="bf16",
    )

    input_ids = np.load(f"{TOKENIZED_DIR}/input_ids.npy")
    labels = np.load(f"{TOKENIZED_DIR}/labels.npy")
    accelerator.print(f"Loaded {len(input_ids)} tokenized examples.")

    class SFTDataset(Dataset):
        def __init__(self, ids, lbls):
            self.input_ids = torch.from_numpy(ids).long()
            self.labels = torch.from_numpy(lbls).long()

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

    full_ds = SFTDataset(input_ids, labels)
    val_size = int(0.05 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    accelerator.print(f"Train: {train_size}, Val: {val_size}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": accelerator.local_process_index},
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    total_steps = math.ceil(len(train_loader) / grad_accum) * n_epochs
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    accelerator.print(f"Training {n_epochs} epoch(s), ~{total_steps} optimizer steps, {NUM_GPUS} GPUs")
    start = time.time()
    global_step = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = outputs.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

            if accelerator.sync_gradients:
                global_step += 1

            if accelerator.sync_gradients and global_step % 20 == 0:
                elapsed = time.time() - start
                accelerator.print(
                    f"  step {global_step}/{total_steps}  loss={loss.item():.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={elapsed:.0f}s"
                )

        avg_train = epoch_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
                val_loss += outputs.loss.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        perplexity = math.exp(min(avg_val, 20))
        elapsed = time.time() - start
        accelerator.print(
            f"Epoch {epoch + 1}/{n_epochs}  train_loss={avg_train:.4f}  "
            f"val_loss={avg_val:.4f}  ppl={perplexity:.2f}  elapsed={elapsed:.0f}s"
        )

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
        unwrapped.save_pretrained(FINAL_MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.save_pretrained(FINAL_MODEL_DIR)
        volume.commit()
        elapsed = time.time() - start
        print(f"\nDone. Adapter saved to {FINAL_MODEL_DIR} in {elapsed:.0f}s")


@app.local_entrypoint()
def main(n_epochs: int = 1, lr: float = 2e-4, batch_size: int = 4, grad_accum: int = 2, skip_tokenize: bool = False):
    if not skip_tokenize:
        n = tokenize.remote()
        print(f"Tokenized {n} examples.")
    train.remote(n_epochs=n_epochs, lr=lr, batch_size=batch_size, grad_accum=grad_accum)
