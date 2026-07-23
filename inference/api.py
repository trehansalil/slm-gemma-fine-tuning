"""Modal web API for inference benchmarks.

Exposes two endpoints:
  POST /kv-cache-benchmark   — KV cache vs no-cache comparison (SLM 125M)
  POST /speculative-decoding — Speculative decoding vs autoregressive (Qwen)
"""
from __future__ import annotations

import modal

PROJECT = "inference-benchmarks"
VOLUME_NAME = "inference-benchmarks"
DATA_ROOT = "/data"
SLM_MODEL_DIR = f"{DATA_ROOT}/slm125m"
HF_SECRET_NAME = "huggingface-secret"

app = modal.App(PROJECT)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3",
        "transformers>=4.42",
        "accelerate>=0.30",
        "huggingface_hub",
        "fastapi[standard]",
    )
)


@app.function(
    image=gpu_image,
    volumes={DATA_ROOT: volume},
    gpu="A10G",
    timeout=600,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
@modal.concurrent(max_inputs=4)
@modal.fastapi_endpoint(method="POST")
def kv_cache_benchmark(item: dict) -> dict:
    """Run KV cache benchmark on the SLM 125M model."""
    import os

    from huggingface_hub import snapshot_download

    model_path = SLM_MODEL_DIR
    if not os.path.exists(os.path.join(model_path, "config.json")):
        snapshot_download(
            "thesreedath/slm-125m-qa",
            local_dir=model_path,
        )
        volume.commit()

    # Inline the benchmark logic to avoid needing to mount project code
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt = item.get("prompt", "The capital of France is")
    max_new_tokens = item.get("max_new_tokens", 256)

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Warmup both code paths
    with torch.no_grad():
        model(input_ids)
        model(input_ids)
    torch.cuda.synchronize()

    def step_by_step(use_cache: bool):
        """Generate token-by-token, recording per-step latency."""
        current_ids = input_ids.clone()
        past_kv = None
        step_times_ms = []
        generated_token_ids = []

        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(max_new_tokens):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                if use_cache:
                    model_input = current_ids if past_kv is None else current_ids[:, -1:]
                    out = model(model_input, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                else:
                    out = model(current_ids, use_cache=False)

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) * 1000
            step_times_ms.append(round(step_ms, 3))

            generated_token_ids.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        total_time = sum(step_times_ms) / 1000
        n_tokens = len(generated_token_ids)
        text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        return {
            "total_tokens": n_tokens,
            "elapsed_s": round(total_time, 4),
            "tokens_per_sec": round(n_tokens / total_time, 2) if total_time > 0 else 0,
            "peak_memory_mb": round(peak_mem, 1),
            "generated_text": text,
            "step_times_ms": step_times_ms,
        }

    without_kv = step_by_step(use_cache=False)
    with_kv = step_by_step(use_cache=True)

    speedup = (
        round(with_kv["tokens_per_sec"] / without_kv["tokens_per_sec"], 2)
        if without_kv["tokens_per_sec"] > 0
        else 0
    )

    return {
        "benchmark": "kv_cache",
        "with_kv_cache": with_kv,
        "without_kv_cache": without_kv,
        "speedup": speedup,
    }


@app.function(
    image=gpu_image,
    volumes={DATA_ROOT: volume},
    gpu="A100",
    timeout=900,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
@modal.fastapi_endpoint(method="POST")
def speculative_decoding_benchmark(item: dict) -> dict:
    """Run speculative decoding benchmark with Qwen models."""
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_id = item.get("target_model", "Qwen/Qwen2.5-7B-Instruct")
    draft_id = item.get("draft_model", "Qwen/Qwen2.5-0.5B-Instruct")
    prompt = item.get("prompt", "Explain the theory of relativity in simple terms.")
    max_new_tokens = item.get("max_new_tokens", 128)
    gamma = item.get("gamma", 4)

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(target_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map="auto"
    )
    target_model.eval()

    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_id, torch_dtype=torch.float16
    ).to(device)
    draft_model.eval()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # Warmup
    _ = target_model(input_ids[:, :1])
    _ = draft_model(input_ids[:, :1])

    # --- Autoregressive baseline ---
    torch.cuda.synchronize()
    start = time.perf_counter()
    baseline_ids = target_model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize()
    baseline_elapsed = time.perf_counter() - start
    baseline_new = baseline_ids[0, input_ids.shape[1]:].tolist()
    baseline_text = tokenizer.decode(baseline_new, skip_special_tokens=True)
    baseline_tps = round(len(baseline_new) / baseline_elapsed, 2)

    # --- Speculative decoding ---
    generated_ids = []
    token_details = []
    draft_accepted = 0
    target_resampled = 0
    current_ids = input_ids.clone()

    torch.cuda.synchronize()
    start = time.perf_counter()

    while len(generated_ids) < max_new_tokens:
        draft_ids = current_ids.clone()
        draft_tokens = []
        draft_logits_list = []

        for _ in range(gamma):
            if len(generated_ids) + len(draft_tokens) >= max_new_tokens:
                break
            with torch.no_grad():
                out = draft_model(draft_ids)
            logits = out.logits[:, -1, :]
            draft_logits_list.append(logits)
            next_token = logits.argmax(dim=-1)
            draft_tokens.append(next_token.item())
            draft_ids = torch.cat([draft_ids, next_token.unsqueeze(0)], dim=1)

        if not draft_tokens:
            break

        with torch.no_grad():
            target_out = target_model(draft_ids)
        target_logits = target_out.logits

        n_accepted = 0
        prompt_plus = current_ids.shape[1]

        for i, draft_tok in enumerate(draft_tokens):
            pos = prompt_plus + i - 1
            t_logits = target_logits[:, pos, :]
            target_tok = t_logits.argmax(dim=-1).item()

            if draft_tok == target_tok:
                generated_ids.append(draft_tok)
                token_details.append({
                    "token": tokenizer.decode([draft_tok]),
                    "source": "draft",
                })
                n_accepted += 1
                draft_accepted += 1
            else:
                generated_ids.append(target_tok)
                token_details.append({
                    "token": tokenizer.decode([target_tok]),
                    "source": "target",
                })
                target_resampled += 1
                break

        if n_accepted == len(draft_tokens) and len(generated_ids) < max_new_tokens:
            bonus_logits = target_logits[:, prompt_plus + len(draft_tokens) - 1, :]
            bonus_tok = bonus_logits.argmax(dim=-1).item()
            generated_ids.append(bonus_tok)
            token_details.append({
                "token": tokenizer.decode([bonus_tok]),
                "source": "target",
            })
            target_resampled += 1

        accepted_this_round = n_accepted + (1 if n_accepted == len(draft_tokens) else 1)
        new_tokens = torch.tensor(
            [generated_ids[-accepted_this_round:]], device=device
        )
        current_ids = torch.cat([current_ids, new_tokens], dim=1)

        eos_id = tokenizer.eos_token_id
        if eos_id is not None and generated_ids[-1] == eos_id:
            break

    torch.cuda.synchronize()
    spec_elapsed = time.perf_counter() - start
    total = len(generated_ids)
    spec_tps = round(total / spec_elapsed, 2) if spec_elapsed > 0 else 0
    acceptance_rate = round(
        draft_accepted / max(draft_accepted + target_resampled, 1), 4
    )
    speedup = round(spec_tps / baseline_tps, 2) if baseline_tps > 0 else 0

    return {
        "benchmark": "speculative_decoding",
        "prompt": prompt,
        "target_model": target_id,
        "draft_model": draft_id,
        "gamma": gamma,
        "autoregressive": {
            "total_tokens": len(baseline_new),
            "elapsed_s": round(baseline_elapsed, 4),
            "tokens_per_sec": baseline_tps,
            "generated_text": baseline_text,
        },
        "speculative": {
            "total_tokens": total,
            "elapsed_s": round(spec_elapsed, 4),
            "tokens_per_sec": spec_tps,
            "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
            "acceptance_rate": acceptance_rate,
            "draft_accepted": draft_accepted,
            "target_resampled": target_resampled,
            "token_details": token_details,
        },
        "speedup": speedup,
    }
