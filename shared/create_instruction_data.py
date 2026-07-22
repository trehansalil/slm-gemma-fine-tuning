"""Create a diverse in-domain instruction-tuning dataset via Azure OpenAI.

Stays inside the legal/financial domain but spans diverse TASK TYPES:
extractive QA, summarization of contracts/filings, drafting, structured
extraction, multi-step reasoning over financial statements, and
comparison/analysis. Seeds from data/sft_train.jsonl user prompts and
generates fresh synthetic instructions on top.

Quality bar matches the preference pipeline: each instruction gets two
candidate responses; an LLM judge picks the better one (same
swap-to-reduce-position-bias trick as judge_pair in
gemma2b/create_preference_data.py) and must score the winner >= 3/5.
Prompts are deduplicated (lowercased/stripped) and junk is filtered.

Output: chat-messages JSONL matching data/sft_train.jsonl, one object/line:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
    python -m shared.create_instruction_data
    python -m shared.create_instruction_data --n-samples 2000
    python -m shared.create_instruction_data --concurrency 30 --append
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time

from shared.azure_client import make_client

OUTPUT_SYSTEM = "You are a helpful legal and financial assistant."

STRONG_SYSTEM = (
    "You are an expert legal and financial assistant. Provide detailed, "
    "accurate, well-structured answers with clear reasoning."
)

TASK_TYPES = {
    "extractive_qa": (
        "extractive question answering: a short legal or financial passage "
        "(court opinion excerpt, contract clause, filing paragraph) followed "
        "by a question answerable from that passage alone"
    ),
    "summarization": (
        "summarization: a contract section, court opinion excerpt, or SEC "
        "filing passage (150-400 words, included in the instruction) with a "
        "request to summarize it for a stated audience"
    ),
    "drafting": (
        "drafting: a request to draft a contract clause, client email, or "
        "internal memo, with concrete requirements (parties, terms, tone) "
        "stated in the instruction"
    ),
    "structured_extraction": (
        "structured extraction: a passage with several entities, dates, or "
        "figures (included in the instruction) and a request to extract them "
        "into a specified table or JSON structure"
    ),
    "multi_step_reasoning": (
        "multi-step reasoning: a small financial statement or set of figures "
        "(included in the instruction) requiring 2-4 chained calculations or "
        "inferences to answer"
    ),
    "comparison_analysis": (
        "comparison/analysis: two short clauses, terms, or financial options "
        "(included in the instruction) with a request to compare them and "
        "recommend or analyze trade-offs"
    ),
}

INSTRUCTION_GEN_SYSTEM = (
    "You generate training instructions for a legal/financial AI assistant. "
    "Each instruction must be fully self-contained: include any passage, "
    "clause, or figures needed to complete the task inside the instruction "
    "itself. Output ONLY a valid JSON array of strings, nothing else."
)

INSTRUCTION_GEN_TEMPLATE = (
    "Generate {k} diverse instructions of this task type:\n{task_desc}\n\n"
    "For domain grounding, here are example user prompts from our existing "
    "legal/financial dataset (match their domain, NOT their task type):\n"
    "{seeds}\n\n"
    "Vary the sub-domain (contracts, litigation, securities filings, "
    "accounting, tax, banking), length, and difficulty. "
    "Output a JSON array of {k} instruction strings."
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

REFUSAL_MARKERS = (
    "i'm sorry", "i am sorry", "i cannot", "i can't", "as an ai",
    "i'm unable", "i am unable", "i apologize",
)

MIN_WINNER_SCORE = 3


def is_junk(text: str | None, min_len: int = 30) -> bool:
    if not text or len(text.strip()) < min_len:
        return True
    return text.strip().lower().startswith(REFUSAL_MARKERS)


def load_seed_prompts(path: str) -> list[str]:
    """Unique user prompts from the SFT dataset."""
    with open(path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

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
    return prompts


async def chat(client, model, system, user, semaphore, *,
               max_completion_tokens=512, temperature=0.7) -> str | None:
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            print(f"  [error] chat: {e}")
            return None


async def gen_instructions(client, model, task_desc, seeds, k, semaphore) -> list[str]:
    user = INSTRUCTION_GEN_TEMPLATE.format(
        k=k, task_desc=task_desc,
        seeds="\n".join(f"- {s[:300]}" for s in seeds),
    )
    text = await chat(client, model, INSTRUCTION_GEN_SYSTEM, user, semaphore,
                      max_completion_tokens=4000, temperature=1.0)
    if not text:
        return []
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        items = json.loads(text[start:end])
    except json.JSONDecodeError:
        return []
    return [x.strip() for x in items if isinstance(x, str) and x.strip()]


async def judge_pair(client, model, question, response_a, response_b,
                     semaphore) -> dict | None:
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
    client, model = make_client()
    random.seed(args.seed)

    existing = []
    existing_prompts: set[str] = set()
    out_path = args.output
    if args.append and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing.append(item)
                    user = next(m["content"] for m in item["messages"]
                                if m["role"] == "user")
                    existing_prompts.add(user.strip().lower())
        print(f"Loaded {len(existing)} existing examples (append mode)")

    sft_path = os.path.join(os.path.dirname(__file__), "..", "data", "sft_train.jsonl")
    seeds = load_seed_prompts(sft_path)
    print(f"Loaded {len(seeds)} seed prompts from SFT data")

    sem = asyncio.Semaphore(args.concurrency)

    # Step 1: assemble instructions — a slice of real SFT prompts as
    # extractive QA, plus fresh synthetic instructions across all task types.
    # Over-generate ~25% so filtering/judging still lands near n_samples.
    n_seed_qa = min(len(seeds), args.n_samples // 6)
    instructions = [
        {"task": "extractive_qa", "instruction": p}
        for p in random.sample(seeds, n_seed_qa)
    ]

    n_synthetic = int(args.n_samples * 1.25) - n_seed_qa
    per_call = 10
    task_keys = list(TASK_TYPES)
    n_calls = max(1, (n_synthetic + per_call - 1) // per_call)
    print(f"Generating ~{n_synthetic} synthetic instructions "
          f"({n_calls} calls across {len(task_keys)} task types)...")
    t0 = time.time()
    call_tasks = []
    call_types = []
    for i in range(n_calls):
        task_key = task_keys[i % len(task_keys)]
        call_types.append(task_key)
        call_tasks.append(gen_instructions(
            client, model, TASK_TYPES[task_key],
            random.sample(seeds, min(3, len(seeds))), per_call, sem,
        ))
    results = await asyncio.gather(*call_tasks)
    for task_key, batch in zip(call_types, results):
        for instr in batch:
            instructions.append({"task": task_key, "instruction": instr})
    print(f"  Done in {time.time() - t0:.1f}s — {len(instructions)} raw instructions")

    # Dedup by prompt text (lowercased/stripped), incl. against existing data
    seen = set(existing_prompts)
    unique = []
    for item in instructions:
        key = item["instruction"].strip().lower()
        if key not in seen and not is_junk(item["instruction"], min_len=20):
            seen.add(key)
            unique.append(item)
    random.shuffle(unique)
    print(f"Unique instructions after dedup/filter: {len(unique)}")

    # Step 2: two candidate responses per instruction (best-of-2)
    print("Generating candidate responses (2 per instruction)...")
    t0 = time.time()
    resp_a = await asyncio.gather(*[
        chat(client, model, STRONG_SYSTEM, it["instruction"], sem)
        for it in unique
    ])
    resp_b = await asyncio.gather(*[
        chat(client, model, STRONG_SYSTEM, it["instruction"], sem)
        for it in unique
    ])
    print(f"  Done in {time.time() - t0:.1f}s")

    valid = []
    for item, a, b in zip(unique, resp_a, resp_b):
        if not is_junk(a) and not is_junk(b):
            valid.append({**item, "resp_a": a, "resp_b": b})
    print(f"Valid after junk filter: {len(valid)}/{len(unique)}")

    # Step 3: LLM judge picks the better response; winner must score >= 3
    print("Running LLM judge validation...")
    t0 = time.time()
    judge_results = await asyncio.gather(*[
        judge_pair(client, model, v["instruction"], v["resp_a"], v["resp_b"], sem)
        for v in valid
    ])
    print(f"  Done in {time.time() - t0:.1f}s")

    final = []
    judge_pass = judge_fail = judge_err = 0
    for item, result in zip(valid, judge_results):
        if result is None:
            judge_err += 1
            continue
        if result.get("winner") == "B":
            response, score = item["resp_b"], result.get("score_b", 0)
        else:
            response, score = item["resp_a"], result.get("score_a", 0)
        if score >= MIN_WINNER_SCORE:
            final.append({"task": item["task"],
                          "instruction": item["instruction"],
                          "response": response})
            judge_pass += 1
        else:
            judge_fail += 1
    print(f"Judge results: {judge_pass} pass, {judge_fail} fail, {judge_err} error")

    final = final[:args.n_samples]

    records = [
        {"messages": [
            {"role": "system", "content": OUTPUT_SYSTEM},
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["response"]},
        ]}
        for item in final
    ]
    combined = existing + records

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(combined)} total examples ({len(records)} new) to {out_path}")

    by_task: dict[str, int] = {}
    for item in final:
        by_task[item["task"]] = by_task.get(item["task"], 0) + 1
    for task, count in sorted(by_task.items()):
        print(f"  {task}: {count}")
    avg_len = sum(len(x["response"]) for x in final) / max(len(final), 1)
    print(f"Avg response length: {avg_len:.0f} chars")

    return records


def main():
    parser = argparse.ArgumentParser(description="Create instruction dataset via Azure OpenAI")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "instruction_train.jsonl"))
    parser.add_argument("--append", action="store_true",
                        help="Append to existing dataset instead of overwriting")
    args = parser.parse_args()

    asyncio.run(create_dataset(args))


if __name__ == "__main__":
    main()
