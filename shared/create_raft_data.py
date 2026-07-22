"""Create a RAFT dataset for DPO and RLAIF training.

RAFT (Retrieval Augmented Fine-Tuning) teaches models to:
- Answer from oracle passages in a mixed-context window
- Ignore topically similar distractor passages
- Refuse when no oracle passage is present

Pipeline:
  1. Generate domain passages with QA pairs (Azure OpenAI)
  2. Assemble RAFT contexts: oracle + distractors (local)
  3. Generate rejected responses (Azure OpenAI)
  4. Filter and deduplicate

Output: JSONL with prompt, chosen, rejected, and metadata — ready for
both DPO (direct) and RLAIF (use prompt to elicit new responses + judge).

Usage:
    python -m shared.create_raft_data
    python -m shared.create_raft_data --n-samples 18000
    python -m shared.create_raft_data --concurrency 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time

from shared.azure_client import make_client

# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------

DOMAINS = [
    ("contracts",
     "employment agreements, NDAs, license agreements, merger/acquisition "
     "agreements, lease contracts, service level agreements, supply chain "
     "contracts, joint venture agreements, indemnification clauses"),
    ("litigation",
     "court opinions, motions to dismiss, summary judgment orders, settlement "
     "agreements, arbitration awards, appellate decisions, class action "
     "complaints, injunction rulings"),
    ("securities",
     "10-K annual reports, 10-Q quarterly filings, proxy statements, "
     "S-1 registrations, 8-K current reports, prospectus supplements, "
     "shareholder letters, Form 4 insider-trading filings"),
    ("accounting",
     "audit reports, financial statement notes, MD&A sections, internal "
     "control assessments, revenue recognition disclosures, goodwill "
     "impairment analyses, lease accounting summaries"),
    ("tax",
     "IRS revenue rulings, tax court opinions, private letter rulings, "
     "transfer pricing reports, tax advisory memoranda, state/local tax "
     "compliance guidance, OECD BEPS commentary"),
    ("banking",
     "loan agreements, credit facility term sheets, FDIC examination "
     "reports, Basel III compliance documents, AML/KYC policies, mortgage "
     "servicing guidelines, CFPB enforcement actions"),
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PASSAGE_QA_SYSTEM = (
    "You generate training data for a legal/financial AI. Output ONLY "
    "valid JSON, nothing else."
)

PASSAGE_QA_TEMPLATE = (
    "Generate a realistic legal/financial passage and question-answer pairs.\n\n"
    "Domain: {domain}\n"
    "Sub-areas: {sub_areas}\n\n"
    "Requirements:\n"
    "- Passage: 80-150 words, realistic, with specific names, dates, "
    "dollar amounts, percentages, and defined terms\n"
    "- Generate {n_qa} questions answerable ONLY from this passage\n"
    "- Answers: 1-4 sentences, precise, grounded in the passage\n"
    "- Vary question types: factual recall, numerical, definitional, "
    "conditional, comparative\n\n"
    "Output JSON:\n"
    '{{"passage": "...", "questions": [{{"q": "...", "a": "..."}}]}}'
)

REJECTED_SYSTEM = (
    "You generate a plausible but WRONG answer. Sound confident and "
    "professional. Output ONLY the wrong answer text."
)

REJECTED_ANSWERABLE_TEMPLATE = (
    "A user was given several context passages and asked:\n"
    "{question}\n\n"
    "The correct answer (from the oracle passage) is:\n"
    "{correct_answer}\n\n"
    "Distractor passages the user also saw:\n"
    "{distractors}\n\n"
    "Generate a WRONG answer that either:\n"
    "- Cites information from a distractor passage instead of the oracle\n"
    "- Fabricates plausible-sounding but incorrect details\n"
    "It should sound confident and be similar in length to the correct answer."
)

REJECTED_UNANSWERABLE_TEMPLATE = (
    "A user was given several context passages and asked:\n"
    "{question}\n\n"
    "None of the passages contain the answer, but generate a confident, "
    "professional-sounding answer anyway. Include fabricated but plausible "
    "details (names, dates, figures). 2-4 sentences. Do NOT acknowledge "
    "that the information is unavailable."
)

REFUSAL_TEMPLATES = [
    "Based on the provided passages, I cannot find sufficient information "
    "to answer this question. The passages discuss related legal/financial "
    "topics but do not contain the specific details needed.",
    "The provided context passages do not contain information relevant to "
    "this question. I cannot answer without risking inaccuracy.",
    "After reviewing all provided passages, none contain the information "
    "needed to answer this question accurately.",
    "This question cannot be answered from the given passages. While the "
    "passages cover related subject matter, they do not address the "
    "specific point being asked about.",
    "I don't have enough information in the provided passages to answer "
    "this question. Answering would require speculation beyond what the "
    "context supports.",
]

RAFT_PROMPT_TEMPLATE = (
    "Given the following passages, answer the question. If the answer "
    "cannot be found in any passage, state that clearly.\n\n"
    "{passages_text}\n"
    "Question: {question}"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def chat(client, model, system, user, semaphore, *,
               max_tokens=1024, temperature=0.7) -> str | None:
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            print(f"  [error] {e}")
            return None


def parse_json(text: str | None):
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        start = text.find("[")
    end = max(text.rfind("}"), text.rfind("]")) + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def format_passages_block(passages: list[str]) -> str:
    parts = []
    for i, p in enumerate(passages, 1):
        parts.append(f"Passage {i}:\n{p}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Phase 1: Generate passage + QA bundles
# ---------------------------------------------------------------------------

async def gen_passage_qa(client, model, domain, sub_areas, n_qa, sem):
    user = PASSAGE_QA_TEMPLATE.format(
        domain=domain, sub_areas=sub_areas, n_qa=n_qa,
    )
    text = await chat(client, model, PASSAGE_QA_SYSTEM, user, sem,
                      max_tokens=2000, temperature=1.0)
    obj = parse_json(text)
    if not obj or "passage" not in obj or "questions" not in obj:
        return None
    passage = obj["passage"].strip()
    if len(passage) < 80:
        return None
    qas = []
    for item in obj["questions"]:
        if isinstance(item, dict) and "q" in item and "a" in item:
            q = item["q"].strip()
            a = item["a"].strip()
            if len(q) > 10 and len(a) > 10:
                qas.append({"q": q, "a": a})
    if not qas:
        return None
    return {"passage": passage, "domain": domain, "qas": qas}


async def build_passage_bank(client, model, sem, n_bundles: int):
    """Generate passage bundles across all domains."""
    print(f"Phase 1: Generating ~{n_bundles} passage-QA bundles "
          f"across {len(DOMAINS)} domains...")
    t0 = time.time()

    tasks = []
    for i in range(n_bundles):
        domain, sub_areas = DOMAINS[i % len(DOMAINS)]
        tasks.append(gen_passage_qa(client, model, domain, sub_areas, 4, sem))

    results = await asyncio.gather(*tasks)
    bank = [r for r in results if r is not None]
    elapsed = time.time() - t0
    total_qas = sum(len(b["qas"]) for b in bank)
    print(f"  Done in {elapsed:.1f}s — {len(bank)} passages, "
          f"{total_qas} QA pairs")
    return bank


# ---------------------------------------------------------------------------
# Phase 2: Assemble RAFT examples (local, no API)
# ---------------------------------------------------------------------------

def assemble_raft_examples(bank, n_samples, unanswerable_frac=0.25,
                           n_distractors=2, max_distractor_len=200):
    """Build RAFT prompts from the passage bank."""
    print(f"Phase 2: Assembling {n_samples} RAFT examples "
          f"({unanswerable_frac:.0%} unanswerable)...")

    n_unanswerable = int(n_samples * unanswerable_frac)
    n_answerable = n_samples - n_unanswerable

    # Flatten all (passage_idx, qa_idx) pairs
    qa_index = []
    for pi, bundle in enumerate(bank):
        for qi in range(len(bundle["qas"])):
            qa_index.append((pi, qi))
    random.shuffle(qa_index)

    examples = []

    # --- Answerable examples ---
    for idx in range(min(n_answerable, len(qa_index))):
        pi, qi = qa_index[idx]
        bundle = bank[pi]
        qa = bundle["qas"][qi]

        # Pick distractors from other passages, prefer same-domain for ~half
        same_domain = [i for i in range(len(bank))
                       if i != pi and bank[i]["domain"] == bundle["domain"]]
        diff_domain = [i for i in range(len(bank))
                       if i != pi and bank[i]["domain"] != bundle["domain"]]

        n_same = min(n_distractors // 2, len(same_domain))
        n_diff = n_distractors - n_same
        distractor_idxs = (random.sample(same_domain, n_same) +
                           random.sample(diff_domain, min(n_diff, len(diff_domain))))

        # Build passage list with oracle in random position
        all_passages = [bank[di]["passage"] for di in distractor_idxs]
        oracle_pos = random.randint(0, len(all_passages))
        all_passages.insert(oracle_pos, bundle["passage"])

        prompt = RAFT_PROMPT_TEMPLATE.format(
            passages_text=format_passages_block(all_passages),
            question=qa["q"],
        )
        examples.append({
            "prompt": prompt,
            "chosen": qa["a"],
            "answerable": True,
            "domain": bundle["domain"],
            "oracle_index": oracle_pos,
            "question": qa["q"],
            "distractor_passages": [bank[di]["passage"][:max_distractor_len]
                                    for di in distractor_idxs],
        })

    # --- Unanswerable examples ---
    # Reuse QA pairs but assemble context WITHOUT the oracle passage
    remaining = qa_index[n_answerable:]
    if len(remaining) < n_unanswerable:
        remaining = remaining + qa_index[:n_unanswerable - len(remaining)]

    for idx in range(min(n_unanswerable, len(remaining))):
        pi, qi = remaining[idx]
        bundle = bank[pi]
        qa = bundle["qas"][qi]

        others = [i for i in range(len(bank)) if i != pi]
        distractor_idxs = random.sample(
            others, min(n_distractors + 1, len(others))
        )

        all_passages = [bank[di]["passage"] for di in distractor_idxs]

        prompt = RAFT_PROMPT_TEMPLATE.format(
            passages_text=format_passages_block(all_passages),
            question=qa["q"],
        )
        examples.append({
            "prompt": prompt,
            "chosen": random.choice(REFUSAL_TEMPLATES),
            "answerable": False,
            "domain": bundle["domain"],
            "oracle_index": -1,
            "question": qa["q"],
            "distractor_passages": [p[:max_distractor_len] for p in all_passages[:2]],
        })

    random.shuffle(examples)
    print(f"  Assembled {len(examples)} examples "
          f"({sum(1 for e in examples if e['answerable'])} answerable, "
          f"{sum(1 for e in examples if not e['answerable'])} unanswerable)")
    return examples


# ---------------------------------------------------------------------------
# Phase 3: Generate rejected responses
# ---------------------------------------------------------------------------

async def gen_rejected(client, model, example, sem):
    if example["answerable"]:
        distractors_text = "\n\n".join(
            f"- {d}" for d in example["distractor_passages"]
        )
        user = REJECTED_ANSWERABLE_TEMPLATE.format(
            question=example["question"],
            correct_answer=example["chosen"],
            distractors=distractors_text,
        )
    else:
        user = REJECTED_UNANSWERABLE_TEMPLATE.format(
            question=example["question"],
        )
    return await chat(client, model, REJECTED_SYSTEM, user, sem,
                      max_tokens=512, temperature=0.8)


async def fill_rejected(client, model, sem, examples):
    """Generate rejected responses for all examples."""
    print(f"Phase 3: Generating {len(examples)} rejected responses...")
    t0 = time.time()
    results = await asyncio.gather(*[
        gen_rejected(client, model, ex, sem) for ex in examples
    ])
    filled = 0
    for ex, rej in zip(examples, results):
        if rej and len(rej.strip()) > 15:
            ex["rejected"] = rej.strip()
            filled += 1
        else:
            ex["rejected"] = None
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {filled}/{len(examples)} filled")
    return [ex for ex in examples if ex["rejected"] is not None]


# ---------------------------------------------------------------------------
# Phase 4: Dedup and write
# ---------------------------------------------------------------------------

def dedup_and_write(examples, out_path, n_target):
    seen = set()
    deduped = []
    for ex in examples:
        key = ex["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(ex)

    deduped = deduped[:n_target]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in deduped:
            row = {
                "prompt": ex["prompt"],
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
                "meta": {
                    "answerable": ex["answerable"],
                    "domain": ex["domain"],
                    "oracle_index": ex["oracle_index"],
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Stats
    if not deduped:
        print(f"\nWARNING: No examples survived deduplication. "
              f"Output file {out_path} is empty.")
        return deduped

    answerable = sum(1 for e in deduped if e["answerable"])
    unanswerable = len(deduped) - answerable
    by_domain: dict[str, int] = {}
    for e in deduped:
        by_domain[e["domain"]] = by_domain.get(e["domain"], 0) + 1

    print(f"\nSaved {len(deduped)} examples to {out_path}")
    print(f"  Answerable: {answerable} ({answerable/len(deduped):.1%})")
    print(f"  Unanswerable: {unanswerable} ({unanswerable/len(deduped):.1%})")
    for domain, count in sorted(by_domain.items()):
        print(f"  {domain}: {count}")
    return deduped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def create_dataset(args):
    client, model = make_client()
    random.seed(args.seed)

    # Over-generate by 30% to account for filtering
    n_target = args.n_samples
    n_bundles = int((n_target * 1.3) / 3.5)  # ~3.5 usable QAs per bundle

    sem = asyncio.Semaphore(args.concurrency)

    bank = await build_passage_bank(client, model, sem, n_bundles)
    if not bank:
        print("ERROR: No passages generated. Check Azure OpenAI credentials.")
        return

    examples = assemble_raft_examples(
        bank, int(n_target * 1.15),
        unanswerable_frac=args.unanswerable_frac,
        max_distractor_len=args.max_distractor_len,
    )

    examples = await fill_rejected(client, model, sem, examples)

    dedup_and_write(examples, args.output, n_target)


def main():
    parser = argparse.ArgumentParser(
        description="Create RAFT dataset for DPO/RLAIF")
    parser.add_argument("--n-samples", type=int, default=18000)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unanswerable-frac", type=float, default=0.25)
    parser.add_argument("--max-distractor-len", type=int, default=200,
                        help="Max characters to keep per distractor passage "
                             "in metadata (default: 200)")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(os.path.dirname(__file__), "..",
                             "data", "raft_train.jsonl"),
    )
    args = parser.parse_args()
    asyncio.run(create_dataset(args))


if __name__ == "__main__":
    main()
