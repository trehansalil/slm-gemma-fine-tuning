"""KV cache vs no-KV-cache inference benchmark for SLM 125M.

Compares generation speed (tokens/sec) with and without KV cache
at varying batch sizes to demonstrate the impact of KV caching.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BenchmarkResult:
    mode: str  # "with_kv_cache" or "without_kv_cache"
    batch_size: int
    total_tokens_generated: int
    elapsed_seconds: float
    tokens_per_second: float
    generated_texts: list[str] = field(default_factory=list)
    peak_memory_mb: float = 0.0


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return 0.0


def generate_with_kv_cache(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, float]:
    """Standard autoregressive generation with KV cache (default)."""
    if input_ids.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(input_ids.device)
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
        )
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return output_ids, elapsed


def generate_without_kv_cache(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, float]:
    """Autoregressive generation WITHOUT KV cache — recomputes all attention each step."""
    if input_ids.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(input_ids.device)
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=False,
            do_sample=False,
        )
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return output_ids, elapsed


def run_benchmark(
    model_path: str,
    prompt: str = "The capital of France is",
    max_new_tokens: int = 100,
    batch_sizes: list[int] | None = None,
    device: torch.device | None = None,
) -> list[BenchmarkResult]:
    """Run the KV cache benchmark across batch sizes.

    Returns a list of BenchmarkResult for each (mode, batch_size) combination.
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]
    if device is None:
        device = _get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
    ).to(device)
    model.eval()

    results: list[BenchmarkResult] = []

    for bs in batch_sizes:
        prompts = [prompt] * bs
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        prompt_len = input_ids.shape[1]

        # --- With KV cache ---
        output_ids, elapsed = generate_with_kv_cache(
            model, input_ids, attention_mask, max_new_tokens
        )
        new_tokens = (output_ids.shape[1] - prompt_len) * bs
        texts = tokenizer.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )
        results.append(
            BenchmarkResult(
                mode="with_kv_cache",
                batch_size=bs,
                total_tokens_generated=new_tokens,
                elapsed_seconds=round(elapsed, 4),
                tokens_per_second=round(new_tokens / elapsed, 2),
                generated_texts=texts,
                peak_memory_mb=round(_peak_memory_mb(device), 1),
            )
        )

        # --- Without KV cache ---
        output_ids, elapsed = generate_without_kv_cache(
            model, input_ids, attention_mask, max_new_tokens
        )
        new_tokens = (output_ids.shape[1] - prompt_len) * bs
        texts = tokenizer.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )
        results.append(
            BenchmarkResult(
                mode="without_kv_cache",
                batch_size=bs,
                total_tokens_generated=new_tokens,
                elapsed_seconds=round(elapsed, 4),
                tokens_per_second=round(new_tokens / elapsed, 2),
                generated_texts=texts,
                peak_memory_mb=round(_peak_memory_mb(device), 1),
            )
        )

    return results


def format_results(results: list[BenchmarkResult]) -> dict:
    """Format benchmark results into a JSON-serializable dict."""
    rows = []
    for r in results:
        rows.append(
            {
                "mode": r.mode,
                "batch_size": r.batch_size,
                "total_tokens": r.total_tokens_generated,
                "elapsed_s": r.elapsed_seconds,
                "tokens_per_sec": r.tokens_per_second,
                "peak_memory_mb": r.peak_memory_mb,
                "generated_text": r.generated_texts[0] if r.generated_texts else "",
            }
        )

    grouped: dict[int, dict] = {}
    for row in rows:
        bs = row["batch_size"]
        if bs not in grouped:
            grouped[bs] = {}
        grouped[bs][row["mode"]] = row

    comparisons = []
    for bs in sorted(grouped):
        with_kv = grouped[bs].get("with_kv_cache", {})
        without_kv = grouped[bs].get("without_kv_cache", {})
        speedup = (
            round(with_kv["tokens_per_sec"] / without_kv["tokens_per_sec"], 2)
            if without_kv.get("tokens_per_sec", 0) > 0
            else 0
        )
        comparisons.append(
            {
                "batch_size": bs,
                "with_kv_cache": with_kv,
                "without_kv_cache": without_kv,
                "speedup": speedup,
            }
        )

    return {"benchmark": "kv_cache", "comparisons": comparisons}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="KV cache benchmark")
    parser.add_argument(
        "--model", default="models/slm125m/base", help="Path to model"
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8]
    )
    args = parser.parse_args()

    results = run_benchmark(
        model_path=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        batch_sizes=args.batch_sizes,
    )
    output = format_results(results)
    print(json.dumps(output, indent=2))
