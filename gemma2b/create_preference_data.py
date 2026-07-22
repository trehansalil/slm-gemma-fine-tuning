"""Create a preference dataset for DPO using Azure OpenAI as the AI judge.

Samples prompts from the SFT training data, generates chosen (strong) and
rejected (weak) responses via Azure GPT, then validates with an LLM judge.

Usage:
    python -m gemma2b.create_preference_data
    python -m gemma2b.create_preference_data --n-samples 200
    python -m gemma2b.create_preference_data --concurrency 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time

from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

STRONG_SYSTEM = (
    "You are an expert legal and financial assistant. "
    "Provide detailed, accurate, well-structured answers with clear reasoning."
)

WEAK_SYSTEM = (
    "You are a general assistant. Give a brief, surface-level answer. "
    "Do not provide detailed reasoning, citations, or structured formatting."
)

JUDGE_SYSTEM = (
    "You are an impartial quality judge. You will be given a question, "
    "Response A, and Response B. Decide which response is better for a "
    "legal/financial professional seeking a thorough answer.\n\n"
    "Output ONLY valid JSON: {\"winner\": \"A\" or \"B\", \"score_a\": 1-5, \"score_b\": 1-5, \"reason\": \"...\"}"
)

JUDGE_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}\n\n"
    "Which response is better? Output JSON only."
)


def load_prompts(path: str, n_samples: int, seed: int = 42) -> list[str]:
    """Sample unique user prompts from the SFT dataset."""
    with open(path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    prompts = []
    seen = set()
    for ex in examples:
        for msg in ex["messages"]:
            if msg["role"] == "user":
                text = msg["content"].strip()
                if text and text not in seen:
                    seen.add(text)
                    prompts.append(text)
                break

    random.seed(seed)
    random.shuffle(prompts)
    return prompts[:n_samples]


async def generate_response(
    client: AsyncAzureOpenAI,
    model: str,
    system: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
    max_completion_tokens: int = 512,
) -> str | None:
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_completion_tokens,
                temperature=0.7,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            print(f"  [error] generate: {e}")
            return None


async def judge_pair(
    client: AsyncAzureOpenAI,
    model: str,
    question: str,
    response_a: str,
    response_b: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Ask the LLM judge which response is better. Randomly swaps order to reduce position bias."""
    swapped = random.random() < 0.5
    if swapped:
        ra, rb = response_b, response_a
    else:
        ra, rb = response_a, response_b

    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question, response_a=ra, response_b=rb
    )

    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=200,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content
            text = raw.strip() if raw else ""
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                return None
            result = json.loads(text[start:end])
            if swapped:
                actual_winner = "B" if result["winner"] == "A" else "A"
                result["winner"] = actual_winner
                result["score_a"], result["score_b"] = result["score_b"], result["score_a"]
            return result
        except Exception as e:
            print(f"  [error] judge: {e}")
            return None


async def create_dataset(args):
    client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_API_KEY"],
        api_version=os.environ.get("AZURE_API_VERSION", "2025-01-01-preview"),
        azure_endpoint=os.environ["AZURE_BASE_URL"],
    )
    model = os.environ.get("AZURE_MODEL", "gpt-5.4-mini")
    if model.startswith("azure/"):
        model = model[len("azure/"):]

    existing = []
    existing_prompts: set[str] = set()
    out_path = args.output
    if args.append and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing.append(item)
                    existing_prompts.add(item["prompt"].strip().lower())
        print(f"Loaded {len(existing)} existing pairs (append mode)")

    sft_path = os.path.join(os.path.dirname(__file__), "..", "data", "sft_train.jsonl")
    prompts = load_prompts(sft_path, args.n_samples, seed=args.seed)
    if args.append:
        prompts = [p for p in prompts if p.strip().lower() not in existing_prompts]
    print(f"Sampled {len(prompts)} unique new prompts")

    sem = asyncio.Semaphore(args.concurrency)

    # Step 1: Generate chosen (strong) responses
    print("Generating chosen (strong) responses...")
    t0 = time.time()
    chosen_tasks = [
        generate_response(client, model, STRONG_SYSTEM, p, sem) for p in prompts
    ]
    chosen_responses = await asyncio.gather(*chosen_tasks)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Step 2: Generate rejected (weak) responses
    print("Generating rejected (weak) responses...")
    t0 = time.time()
    rejected_tasks = [
        generate_response(client, model, WEAK_SYSTEM, p, sem) for p in prompts
    ]
    rejected_responses = await asyncio.gather(*rejected_tasks)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Filter out failures
    valid = []
    for prompt, chosen, rejected in (
        zip(prompts, chosen_responses, rejected_responses)
    ):
        if chosen and rejected and len(chosen) > 20 and len(rejected) > 10:
            valid.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    print(f"Valid pairs after generation: {len(valid)}/{len(prompts)}")

    # Step 3: LLM judge validation
    print("Running LLM judge validation...")
    t0 = time.time()
    judge_tasks = [
        judge_pair(client, model, v["prompt"], v["chosen"], v["rejected"], sem)
        for v in valid
    ]
    judge_results = await asyncio.gather(*judge_tasks)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Filter: keep only where judge confirms chosen > rejected
    final = []
    judge_pass = judge_fail = judge_err = 0
    for item, result in zip(valid, judge_results):
        if result is None:
            judge_err += 1
            continue
        if result.get("winner") == "A" and result.get("score_a", 0) > result.get("score_b", 0):
            final.append(item)
            judge_pass += 1
        else:
            judge_fail += 1

    print(f"Judge results: {judge_pass} pass, {judge_fail} fail, {judge_err} error")

    # Deduplicate by prompt (including against existing data)
    seen = set(existing_prompts)
    deduped = []
    for item in final:
        key = item["prompt"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    final = deduped

    combined = existing + final

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(combined)} total pairs ({len(final)} new) to {out_path}")

    # Stats
    avg_chosen_len = sum(len(x["chosen"]) for x in combined) / max(len(combined), 1)
    avg_rejected_len = sum(len(x["rejected"]) for x in combined) / max(len(combined), 1)
    print(f"Avg chosen length: {avg_chosen_len:.0f} chars")
    print(f"Avg rejected length: {avg_rejected_len:.0f} chars")

    return final


def main():
    parser = argparse.ArgumentParser(description="Create preference dataset via Azure OpenAI")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "preference_train.jsonl"))
    parser.add_argument("--append", action="store_true",
                        help="Append to existing dataset instead of overwriting")
    args = parser.parse_args()

    asyncio.run(create_dataset(args))


if __name__ == "__main__":
    main()
