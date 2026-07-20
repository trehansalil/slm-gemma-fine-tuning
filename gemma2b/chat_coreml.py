"""Chat with Gemma 2 2B via CoreML (Apple Silicon, KV cached)."""

import argparse
import json
import sys
import time

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

MODEL_DIR = "models/coreml_local"
SYSTEM_MSG = "You are a helpful legal and financial assistant. Answer based only on provided context."


def load_model(model_dir: str):
    print("Loading CoreML model...")
    t0 = time.time()

    with open(f"{model_dir}/model_meta.json") as f:
        meta = json.load(f)

    model = ct.models.MLModel(f"{model_dir}/gemma2-2b-sft.mlpackage")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    rotary_cos = np.load(f"{model_dir}/rotary_cos.npy")
    rotary_sin = np.load(f"{model_dir}/rotary_sin.npy")
    causal_mask = np.load(f"{model_dir}/causal_mask.npy")

    print(f"Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer, meta, rotary_cos, rotary_sin, causal_mask


def build_prompt(tokenizer, system: str, user: str) -> list[int]:
    """Build ChatML-formatted prompt for Gemma 2."""
    prompt = (
        f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return tokenizer.encode(prompt, add_special_tokens=True)


def generate(model, tokenizer, meta, rotary_cos, rotary_sin, causal_mask,
             prompt_ids: list[int], max_new_tokens: int = 256,
             temperature: float = 0.7, top_k: int = 50,
             stream: bool = True) -> tuple[str, float, float]:
    eos_id = tokenizer.eos_token_id
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    num_layers = meta["num_layers"]
    num_kv_heads = meta["num_kv_heads"]
    head_dim = meta["head_dim"]
    max_seq = meta["max_seq_len"]

    k_cache = np.zeros((num_layers, num_kv_heads, max_seq, head_dim), dtype=np.float16)
    v_cache = np.zeros_like(k_cache)

    def step(token_id: int, pos: int):
        nonlocal k_cache, v_cache
        cos = rotary_cos[pos:pos + 1][np.newaxis].astype(np.float16)
        sin = rotary_sin[pos:pos + 1][np.newaxis].astype(np.float16)
        mask = causal_mask[pos:pos + 1][np.newaxis, np.newaxis].astype(np.float16)
        scatter = np.full((1, num_kv_heads, 1, head_dim), pos, dtype=np.int32)

        out = model.predict({
            "input_ids": np.array([[token_id]], dtype=np.int32),
            "cos": cos,
            "sin": sin,
            "attn_mask": mask,
            "scatter_pos": scatter,
            "k_cache": k_cache,
            "v_cache": v_cache,
        })

        k_cache = out["new_k_cache"]
        v_cache = out["new_v_cache"]
        return out["logits"][0, 0, :]

    # Prefill
    if len(prompt_ids) >= max_seq:
        print(f"Warning: prompt ({len(prompt_ids)} tokens) exceeds context ({max_seq}), truncating")
        prompt_ids = prompt_ids[:max_seq - 1]
    if not prompt_ids:
        return "", 0.0, 0.0
    t_prefill = time.time()
    for pos, token_id in enumerate(prompt_ids):
        logits = step(token_id, pos)
    t_prefill = time.time() - t_prefill

    # Decode
    t_decode = time.time()
    generated_ids = []
    pos = len(prompt_ids)

    for _ in range(max_new_tokens):
        if pos >= max_seq:
            break

        if temperature > 0:
            scaled = logits / temperature
            if top_k > 0:
                top_idx = np.argpartition(scaled, -top_k)[-top_k:]
                filtered = np.full_like(scaled, -np.inf)
                filtered[top_idx] = scaled[top_idx]
                scaled = filtered
            probs = np.exp(scaled - np.max(scaled))
            probs /= probs.sum()
            next_id = int(np.random.choice(len(probs), p=probs))
        else:
            next_id = int(np.argmax(logits))

        if next_id in (eos_id, eot_id):
            break

        generated_ids.append(next_id)
        token_text = tokenizer.decode([next_id], skip_special_tokens=True)
        if stream:
            sys.stdout.write(token_text)
            sys.stdout.flush()
        logits = step(next_id, pos)
        pos += 1

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    t_decode = time.time() - t_decode

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    decode_tps = len(generated_ids) / t_decode if t_decode > 0 else 0
    prefill_tps = len(prompt_ids) / t_prefill if t_prefill > 0 else 0
    return text, prefill_tps, decode_tps


def interactive(model, tokenizer, meta, rotary_cos, rotary_sin, causal_mask,
                system_prompt: str, max_tokens: int, temperature: float):
    print("\n--- Gemma 2 2B CoreML Chat (KV cached) ---")
    print("Type your question (or 'quit' to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        prompt_ids = build_prompt(tokenizer, system_prompt, user_input)
        print(f"[{len(prompt_ids)} prompt tokens]")

        sys.stdout.write("Gemma: ")
        sys.stdout.flush()
        _, prefill_tps, decode_tps = generate(
            model, tokenizer, meta, rotary_cos, rotary_sin, causal_mask,
            prompt_ids, max_new_tokens=max_tokens, temperature=temperature,
            stream=True,
        )
        print(f"[prefill: {prefill_tps:.0f} tok/s | decode: {decode_tps:.0f} tok/s]\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with Gemma 2 2B CoreML (KV cached)")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--system", type=str, default=SYSTEM_MSG)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer, meta, rot_cos, rot_sin, mask = load_model(args.model_dir)

    if args.prompt:
        prompt_ids = build_prompt(tokenizer, args.system, args.prompt)
        _, prefill_tps, decode_tps = generate(
            model, tokenizer, meta, rot_cos, rot_sin, mask,
            prompt_ids, max_new_tokens=args.max_tokens,
            temperature=args.temperature, stream=True,
        )
        print(f"[prefill: {prefill_tps:.0f} tok/s | decode: {decode_tps:.0f} tok/s]")
    else:
        interactive(model, tokenizer, meta, rot_cos, rot_sin, mask,
                    args.system, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
