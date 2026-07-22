"""Shared training helpers: device detection, Gemma chat tokenization with
loss masking, dynamic-padding collation, and a model-agnostic training loop.

Hoisted from gemma2b/finetune_local.py (_tokenize_chat, IGNORE_INDEX masking,
collate/evaluate/loop) and slm125m/train_dpo.py (get_device) so later stages
(instruction_tune, train_raft) reuse a single implementation. The originals
are left untouched.
"""
from __future__ import annotations

import json
import math
import os
import time
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset, random_split

IGNORE_INDEX = -100
MAX_SEQ_LEN = 1024


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _real_token_id(tokenizer, token: str) -> int | None:
    """Token id, or None if the token is absent / maps to unk."""
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None or tid < 0:
        return None
    if tokenizer.unk_token_id is not None and tid == tokenizer.unk_token_id:
        return None
    return tid


def detect_chat_format(tokenizer) -> str:
    """Which chat-token family this tokenizer supports.

    "gemma"  -> <start_of_turn>/<end_of_turn> (Gemma 2)
    "pipe"   -> <|system|>/<|user|>/<|assistant|>/<|eos|> (SLM 125M)
    """
    if _real_token_id(tokenizer, "<start_of_turn>") is not None:
        return "gemma"
    if all(_real_token_id(tokenizer, t) is not None
           for t in ("<|user|>", "<|assistant|>", "<|eos|>")):
        return "pipe"
    raise ValueError(
        "Tokenizer has neither Gemma <start_of_turn> tokens nor "
        "<|user|>/<|assistant|>/<|eos|> tokens — unsupported chat format."
    )


def tokenize_chat(messages, tokenizer):
    """Tokenize a chat with assistant-only loss masking.

    Dispatches on the tokenizer's token family: Gemma 2's
    <start_of_turn>/<end_of_turn> format, or the SLM 125M
    <|bos|><|system|>...<|eos|> format (per that model's README).
    Loss is computed only on assistant content tokens in both cases.
    """
    if detect_chat_format(tokenizer) == "gemma":
        return _tokenize_chat_gemma(messages, tokenizer)
    return _tokenize_chat_pipe(messages, tokenizer)


def _tokenize_chat_pipe(messages, tokenizer):
    """SLM 125M format:
    <|bos|><|system|>\\n{SYSTEM}<|eos|>\\n<|user|>\\n{Q}<|eos|>\\n<|assistant|>\\n{A}<|eos|>
    """
    role_ids = {
        "system": _real_token_id(tokenizer, "<|system|>"),
        "user": _real_token_id(tokenizer, "<|user|>"),
        "assistant": _real_token_id(tokenizer, "<|assistant|>"),
    }
    eos_id = _real_token_id(tokenizer, "<|eos|>")
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)

    input_ids = []
    labels = []
    if tokenizer.bos_token_id is not None:
        input_ids.append(tokenizer.bos_token_id)
        labels.append(IGNORE_INDEX)

    for i, msg in enumerate(messages):
        role = msg["role"]
        content_ids = tokenizer.encode(msg["content"], add_special_tokens=False)

        header = [role_ids[role]] + nl_ids
        turn_ids = header + content_ids + [eos_id]
        if role == "assistant":
            turn_labels = [IGNORE_INDEX] * len(header) + content_ids + [eos_id]
        else:
            turn_labels = [IGNORE_INDEX] * len(turn_ids)

        input_ids.extend(turn_ids)
        labels.extend(turn_labels)
        if i < len(messages) - 1:
            input_ids.extend(nl_ids)
            labels.extend([IGNORE_INDEX] * len(nl_ids))

    return input_ids, labels


def _tokenize_chat_gemma(messages, tokenizer):
    """Gemma 2 format. System content is folded into the first user turn
    (Gemma has no system role)."""
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


def tokenize_examples(examples, tokenizer, seq_len: int = MAX_SEQ_LEN):
    """Tokenize chat examples into (input_ids, labels) tensor pairs.

    Truncates to seq_len and drops examples with fewer than 5 trainable
    (assistant) tokens. Padding is deferred to collate_fn.
    """
    features = []
    skipped = 0
    for ex in examples:
        input_ids, labels = tokenize_chat(ex["messages"], tokenizer)
        if len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
        trainable = sum(1 for l in labels if l != IGNORE_INDEX)
        if trainable < 5:
            skipped += 1
            continue
        features.append((
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        ))
    if skipped:
        print(f"Skipped {skipped} examples with <5 trainable tokens")
    return features


class ChatDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


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


def train(model, tokenizer, features, output_dir, device, *,
          epochs: int = 2, lr: float = 1e-5, min_lr: float | None = None,
          batch_size: int = 8, grad_accum: int = 2, weight_decay: float = 0.01,
          grad_clip: float = 1.0, log_every: int = 50, eval_every: int = 0,
          seed: int = 42):
    """AdamW + cosine-warmup loop; saves best-val-loss weights to output_dir.

    Works for both full-parameter models and PEFT-wrapped models —
    save_pretrained writes full weights or just the adapter accordingly.
    """
    if min_lr is None:
        min_lr = lr / 10
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    full_ds = ChatDataset(features)
    val_size = max(10, min(100, int(len(full_ds) * 0.05)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Train: {train_size}, Val: {val_size}")

    collate = partial(collate_fn, pad_id=pad_id)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=False, drop_last=True,
                          collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                        num_workers=0, pin_memory=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay,
    )

    steps_per_epoch = max(1, len(train_dl) // grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = min(100, total_steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return lr * step / max(1, warmup_steps)
        ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * ratio))

    os.makedirs(output_dir, exist_ok=True)
    mf = open(os.path.join(output_dir, "metrics.jsonl"), "a", encoding="utf-8")

    best_val_loss = float("inf")
    step = micro = 0
    running_loss = 0.0
    t0 = t_global = time.time()
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_train_loss = 0.0
        epoch_train_steps = 0

        for ids, labs, mask in train_dl:
            ids, labs, mask = ids.to(device), labs.to(device), mask.to(device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss / grad_accum
            loss.backward()
            running_loss += loss.item()
            epoch_train_loss += loss.item()
            micro += 1

            if micro % grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            current_lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            epoch_train_steps += 1

            if step % log_every == 0:
                avg = running_loss / log_every
                sec_per_step = (time.time() - t0) / log_every
                print(f"step {step:>5}/{total_steps} | train_loss {avg:.4f} | "
                      f"lr {current_lr:.2e} | {sec_per_step:.2f}s/step | "
                      f"{(time.time() - t_global) / 60:.1f}min")
                mf.write(json.dumps({"epoch": epoch, "step": step,
                                     "train_loss": round(avg, 4),
                                     "lr": current_lr}) + "\n")
                mf.flush()
                running_loss = 0.0
                t0 = time.time()

            if eval_every and step % eval_every == 0:
                vl = evaluate(model, val_dl, device)
                ppl = math.exp(min(vl, 20))
                improving = "+" if vl < best_val_loss else "-"
                print(f"  [eval step {step:>5}/{total_steps}] val_loss={vl:.4f} "
                      f"ppl={ppl:.2f} {improving} | "
                      f"{(time.time() - t_global) / 60:.1f}min")
                mf.write(json.dumps({"epoch": epoch, "step": step,
                                     "val_loss": round(vl, 4),
                                     "ppl": round(ppl, 2)}) + "\n")
                mf.flush()
                if vl < best_val_loss:
                    best_val_loss = vl
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    print(f"  [best] saved to {output_dir} (step {step})")
                model.train()
                t0 = time.time()

        avg_train_loss = epoch_train_loss / max(1, epoch_train_steps)
        vl = evaluate(model, val_dl, device)
        ppl = math.exp(min(vl, 20))
        improving = "+" if vl < best_val_loss else "-"
        print(f"  [epoch {epoch}] train_loss={avg_train_loss:.4f} "
              f"val_loss={vl:.4f} ppl={ppl:.2f} {improving} | "
              f"{(time.time() - t_global) / 60:.1f}min")
        mf.write(json.dumps({"epoch": epoch, "train_loss": round(avg_train_loss, 4),
                             "val_loss": round(vl, 4), "ppl": round(ppl, 2)}) + "\n")
        mf.flush()

        if vl < best_val_loss:
            best_val_loss = vl
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  [best] saved to {output_dir}")

    mf.close()
    return best_val_loss
