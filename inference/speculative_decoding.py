"""Speculative decoding benchmark: Qwen 2.5 7B (target) + Qwen 2.5 0.5B (draft).

Implements speculative decoding from scratch so we can track which tokens
came from the draft model vs. the target model, and report acceptance rates.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class TokenInfo:
    token_id: int
    token_text: str
    source: str  # "draft" (accepted) or "target" (rejected & resampled)


@dataclass
class SpecDecodingResult:
    mode: str  # "speculative" or "autoregressive"
    total_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    generated_text: str
    token_details: list[TokenInfo] = field(default_factory=list)
    acceptance_rate: float = 0.0
    draft_accepted: int = 0
    target_resampled: int = 0


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def speculative_decode(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    gamma: int = 4,
    temperature: float = 0.0,
) -> SpecDecodingResult:
    """Speculative decoding with token-level attribution.

    Args:
        gamma: number of draft tokens to generate before verification.
        temperature: 0 for greedy, >0 for sampling.
    """
    device = input_ids.device
    generated_ids: list[int] = []
    token_details: list[TokenInfo] = []
    draft_accepted = 0
    target_resampled = 0

    current_ids = input_ids.clone()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    while len(generated_ids) < max_new_tokens:
        # Step 1: Draft model generates gamma tokens greedily
        draft_ids = current_ids.clone()
        draft_tokens: list[int] = []
        draft_logits_list: list[torch.Tensor] = []

        for _ in range(gamma):
            if len(generated_ids) + len(draft_tokens) >= max_new_tokens:
                break
            out = draft_model(draft_ids)
            logits = out.logits[:, -1, :]
            draft_logits_list.append(logits)
            if temperature == 0:
                next_token = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)
            draft_tokens.append(next_token.item())
            draft_ids = torch.cat(
                [draft_ids, next_token.unsqueeze(0)], dim=1
            )

        if not draft_tokens:
            break

        # Step 2: Target model scores all draft tokens in one forward pass
        target_out = target_model(draft_ids)
        target_logits = target_out.logits  # (1, seq_len, vocab)

        # Verify each draft token
        n_accepted = 0
        prompt_plus_generated = current_ids.shape[1]

        for i, draft_tok in enumerate(draft_tokens):
            pos = prompt_plus_generated + i - 1
            t_logits = target_logits[:, pos, :]

            if temperature == 0:
                target_tok = t_logits.argmax(dim=-1).item()
                accepted = draft_tok == target_tok
            else:
                t_probs = torch.softmax(t_logits / temperature, dim=-1)
                d_probs = torch.softmax(draft_logits_list[i] / temperature, dim=-1)
                ratio = t_probs[0, draft_tok] / (d_probs[0, draft_tok] + 1e-10)
                accepted = torch.rand(1, device=device).item() < ratio.item()

            if accepted:
                generated_ids.append(draft_tok)
                token_details.append(
                    TokenInfo(
                        token_id=draft_tok,
                        token_text=tokenizer.decode([draft_tok]),
                        source="draft",
                    )
                )
                n_accepted += 1
                draft_accepted += 1
            else:
                # Resample from target distribution at this position
                if temperature == 0:
                    new_tok = t_logits.argmax(dim=-1).item()
                else:
                    t_probs = torch.softmax(t_logits / temperature, dim=-1)
                    new_tok = torch.multinomial(t_probs, 1).squeeze(-1).item()

                generated_ids.append(new_tok)
                token_details.append(
                    TokenInfo(
                        token_id=new_tok,
                        token_text=tokenizer.decode([new_tok]),
                        source="target",
                    )
                )
                target_resampled += 1
                break

        # If ALL draft tokens accepted, sample one bonus token from target
        if n_accepted == len(draft_tokens) and len(generated_ids) < max_new_tokens:
            bonus_logits = target_logits[:, prompt_plus_generated + len(draft_tokens) - 1, :]
            if temperature == 0:
                bonus_tok = bonus_logits.argmax(dim=-1).item()
            else:
                probs = torch.softmax(bonus_logits / temperature, dim=-1)
                bonus_tok = torch.multinomial(probs, 1).squeeze(-1).item()
            generated_ids.append(bonus_tok)
            token_details.append(
                TokenInfo(
                    token_id=bonus_tok,
                    token_text=tokenizer.decode([bonus_tok]),
                    source="target",
                )
            )
            target_resampled += 1

        # Update current_ids
        new_tokens = torch.tensor(
            [generated_ids[-n_accepted - (1 if n_accepted == len(draft_tokens) else 1):]],
            device=device,
        )
        current_ids = torch.cat(
            [current_ids, new_tokens], dim=1
        )

        # Stop on EOS
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and generated_ids[-1] == eos_id:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total = len(generated_ids)
    acceptance_rate = draft_accepted / max(draft_accepted + target_resampled, 1)

    return SpecDecodingResult(
        mode="speculative",
        total_tokens=total,
        elapsed_seconds=round(elapsed, 4),
        tokens_per_second=round(total / elapsed, 2) if elapsed > 0 else 0,
        generated_text=tokenizer.decode(generated_ids, skip_special_tokens=True),
        token_details=token_details,
        acceptance_rate=round(acceptance_rate, 4),
        draft_accepted=draft_accepted,
        target_resampled=target_resampled,
    )


@torch.no_grad()
def autoregressive_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> SpecDecodingResult:
    """Standard autoregressive generation from the target model (baseline)."""
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        use_cache=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    new_ids = output_ids[0, input_ids.shape[1] :].tolist()
    total = len(new_ids)

    return SpecDecodingResult(
        mode="autoregressive",
        total_tokens=total,
        elapsed_seconds=round(elapsed, 4),
        tokens_per_second=round(total / elapsed, 2) if elapsed > 0 else 0,
        generated_text=tokenizer.decode(new_ids, skip_special_tokens=True),
    )


def run_benchmark(
    prompt: str = "Explain the theory of relativity in simple terms.",
    max_new_tokens: int = 128,
    gamma: int = 4,
    target_model_id: str = TARGET_MODEL_ID,
    draft_model_id: str = DRAFT_MODEL_ID,
    device: torch.device | None = None,
) -> dict:
    """Run speculative decoding vs. autoregressive baseline."""
    if device is None:
        device = _get_device()

    print(f"Loading target model: {target_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    target_model.eval()

    print(f"Loading draft model: {draft_model_id}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        torch_dtype=torch.float16,
    ).to(device)
    draft_model.eval()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # Warmup
    print("Warming up...")
    _ = target_model(input_ids[:, :1])
    _ = draft_model(input_ids[:, :1])

    # Autoregressive baseline
    print("Running autoregressive baseline...")
    baseline = autoregressive_baseline(
        target_model, tokenizer, input_ids, max_new_tokens
    )

    # Speculative decoding
    print("Running speculative decoding...")
    spec = speculative_decode(
        target_model, draft_model, tokenizer, input_ids, max_new_tokens, gamma
    )

    speedup = (
        round(spec.tokens_per_second / baseline.tokens_per_second, 2)
        if baseline.tokens_per_second > 0
        else 0
    )

    return {
        "benchmark": "speculative_decoding",
        "prompt": prompt,
        "target_model": target_model_id,
        "draft_model": draft_model_id,
        "gamma": gamma,
        "autoregressive": {
            "total_tokens": baseline.total_tokens,
            "elapsed_s": baseline.elapsed_seconds,
            "tokens_per_sec": baseline.tokens_per_second,
            "generated_text": baseline.generated_text,
        },
        "speculative": {
            "total_tokens": spec.total_tokens,
            "elapsed_s": spec.elapsed_seconds,
            "tokens_per_sec": spec.tokens_per_second,
            "generated_text": spec.generated_text,
            "acceptance_rate": spec.acceptance_rate,
            "draft_accepted": spec.draft_accepted,
            "target_resampled": spec.target_resampled,
            "token_details": [
                {
                    "token": t.token_text,
                    "source": t.source,
                }
                for t in spec.token_details
            ],
        },
        "speedup": speedup,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Speculative decoding benchmark")
    parser.add_argument("--prompt", default="Explain the theory of relativity in simple terms.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--target-model", default=TARGET_MODEL_ID)
    parser.add_argument("--draft-model", default=DRAFT_MODEL_ID)
    args = parser.parse_args()

    result = run_benchmark(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
        target_model_id=args.target_model,
        draft_model_id=args.draft_model,
    )
    print(json.dumps(result, indent=2))
