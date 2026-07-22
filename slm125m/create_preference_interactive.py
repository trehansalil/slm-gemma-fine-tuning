"""Interactive preference dataset builder for live demos.

Prompts you to type questions one at a time. For each question, generates
a chosen (strong) and rejected (weak) response via Azure GPT, displays both,
and lets you accept, reject, or edit the pair. When done, saves the dataset
and optionally kicks off DPO training.

Usage:
    python -m slm125m.create_preference_interactive
    python -m slm125m.create_preference_interactive --output data/preference_interactive.jsonl
    python -m slm125m.create_preference_interactive --batch  # also load from sft_train.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
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


async def generate_pair(
    client: AsyncAzureOpenAI,
    model: str,
    prompt: str,
) -> tuple[str | None, str | None]:
    """Generate a chosen and rejected response for a prompt."""
    try:
        chosen_resp, rejected_resp = await asyncio.gather(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": STRONG_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.7,
            ),
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": WEAK_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
                temperature=0.7,
            ),
        )
        chosen = chosen_resp.choices[0].message.content
        rejected = rejected_resp.choices[0].message.content
        return (
            chosen.strip() if chosen else None,
            rejected.strip() if rejected else None,
        )
    except Exception as e:
        print(f"\n  [error] {e}")
        return None, None


def print_pair(prompt: str, chosen: str, rejected: str):
    print("\n" + "=" * 60)
    print(f"PROMPT: {prompt}")
    print("-" * 60)
    print(f"CHOSEN (strong):\n{chosen}")
    print("-" * 60)
    print(f"REJECTED (weak):\n{rejected}")
    print("=" * 60)


def load_sft_prompts(path: str, n: int = 50) -> list[str]:
    """Load a random sample of prompts from the SFT dataset."""
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            for msg in ex["messages"]:
                if msg["role"] == "user":
                    prompts.append(msg["content"].strip())
                    break
    random.shuffle(prompts)
    return prompts[:n]


async def interactive_loop(args):
    client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_API_KEY"],
        api_version=os.environ.get("AZURE_API_VERSION", "2025-01-01-preview"),
        azure_endpoint=os.environ["AZURE_BASE_URL"],
    )
    model = os.environ.get("AZURE_MODEL", "gpt-5.4-mini")
    if model.startswith("azure/"):
        model = model[len("azure/"):]

    dataset: list[dict] = []

    # Load existing dataset if appending
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(dataset)} existing pairs from {args.output}")

    # Optionally pre-load SFT prompts for batch mode
    sft_prompts: list[str] = []
    if args.batch:
        sft_path = os.path.join(os.path.dirname(__file__), "..", "data", "sft_train.jsonl")
        if os.path.exists(sft_path):
            sft_prompts = load_sft_prompts(sft_path, args.batch_size)
            print(f"Loaded {len(sft_prompts)} SFT prompts for batch processing")

    print("\n" + "=" * 60)
    print("  INTERACTIVE PREFERENCE DATASET BUILDER")
    print("=" * 60)
    print("\nCommands:")
    print("  Type a question  → generate chosen/rejected pair")
    print("  'batch'          → process pre-loaded SFT prompts")
    print("  'stats'          → show dataset statistics")
    print("  'done'           → save and exit")
    print("  'quit'           → exit without saving new entries")
    print()

    new_count = 0

    while True:
        try:
            user_input = input(f"[{len(dataset)} pairs] Enter prompt (or command): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Exiting without saving new entries.")
            return dataset

        if user_input.lower() == "done":
            break

        if user_input.lower() == "stats":
            if dataset:
                avg_c = sum(len(d["chosen"]) for d in dataset) / len(dataset)
                avg_r = sum(len(d["rejected"]) for d in dataset) / len(dataset)
                print(f"\nDataset: {len(dataset)} pairs")
                print(f"Avg chosen length: {avg_c:.0f} chars")
                print(f"Avg rejected length: {avg_r:.0f} chars")
            else:
                print("\nNo pairs yet.")
            continue

        if user_input.lower() == "batch":
            if not sft_prompts:
                print("No SFT prompts loaded. Use --batch flag or provide prompts manually.")
                continue
            print(f"\nProcessing {len(sft_prompts)} prompts in batch...")
            for i, prompt in enumerate(sft_prompts):
                print(f"\n[{i+1}/{len(sft_prompts)}] Generating for: {prompt[:80]}...")
                chosen, rejected = await generate_pair(client, model, prompt)
                if chosen and rejected:
                    print_pair(prompt, chosen, rejected)
                    while True:
                        action = input("  [a]ccept / [s]kip / [q]uit batch? ").strip().lower()
                        if action in ("a", "s", "q"):
                            break
                    if action == "a":
                        dataset.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                        new_count += 1
                        print(f"  Added. ({len(dataset)} total)")
                    elif action == "q":
                        break
                else:
                    print("  Failed to generate, skipping.")
            continue

        # Regular prompt: generate pair
        print("Generating responses...")
        chosen, rejected = await generate_pair(client, model, user_input)

        if not chosen or not rejected:
            print("Failed to generate responses. Try again.")
            continue

        print_pair(user_input, chosen, rejected)

        while True:
            action = input("  [a]ccept / [r]egenerate / [s]kip? ").strip().lower()
            if action in ("a", "r", "s"):
                break

        if action == "a":
            dataset.append({"prompt": user_input, "chosen": chosen, "rejected": rejected})
            new_count += 1
            print(f"  Added. ({len(dataset)} total)")
        elif action == "r":
            print("Regenerating...")
            chosen, rejected = await generate_pair(client, model, user_input)
            if chosen and rejected:
                print_pair(user_input, chosen, rejected)
                confirm = input("  [a]ccept / [s]kip? ").strip().lower()
                if confirm == "a":
                    dataset.append({"prompt": user_input, "chosen": chosen, "rejected": rejected})
                    new_count += 1
                    print(f"  Added. ({len(dataset)} total)")
            else:
                print("  Failed on retry, skipping.")

    # Save
    if new_count > 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(dataset)} pairs ({new_count} new) to {args.output}")
    else:
        print(f"\nNo new pairs added. Dataset unchanged ({len(dataset)} pairs).")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Interactive preference dataset builder")
    parser.add_argument("--output", type=str, default="data/preference_train.jsonl")
    parser.add_argument("--batch", action="store_true",
                        help="Pre-load SFT prompts for batch review")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Number of SFT prompts to pre-load for batch mode")
    args = parser.parse_args()

    asyncio.run(interactive_loop(args))


if __name__ == "__main__":
    main()
