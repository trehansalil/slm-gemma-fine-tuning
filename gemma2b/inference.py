"""Run inference with the fine-tuned Gemma 2 2B LoRA adapter.

Usage (Modal):
    modal run gemma2b/inference.py
    modal run gemma2b/inference.py --prompt "What is a tort?"
"""
from __future__ import annotations

import modal

PROJECT = "gemma2-2b-sft"
VOLUME_NAME = "gemma2-2b-sft"
DATA_ROOT = "/data"
FINAL_MODEL_DIR = f"{DATA_ROOT}/final"
BASE_MODEL = "google/gemma-2-2b"
HF_SECRET_NAME = "huggingface-secret"
GPU_TYPE = "A100"

SYSTEM_MSG = "You helpful legal financial assistant. Answer based only on provided context."

DEFAULT_PROMPTS = [
    "Under what Alabama rule did court grant Blockbuster, Inc. permission appeal interlocutory order?",
    "What is the standard of review for summary judgment in federal courts?",
    "Explain the doctrine of res judicata in simple terms.",
]

app = modal.App(PROJECT)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3",
        "transformers>=4.42",
        "peft>=0.11",
        "bitsandbytes>=0.43",
        "accelerate>=0.30",
    )
)


@app.function(
    image=image,
    volumes={DATA_ROOT: volume},
    gpu=GPU_TYPE,
    timeout=600,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def generate(prompt: str, max_new_tokens: int = 256) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, FINAL_MODEL_DIR)
    model.eval()

    messages = [
        {"role": "user", "content": f"{SYSTEM_MSG}\n\n{prompt}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@app.local_entrypoint()
def main(prompt: str = None):
    prompts = [prompt] if prompt else DEFAULT_PROMPTS
    for p in prompts:
        print(f"\n{'=' * 60}")
        print(f"Q: {p}")
        print(f"{'=' * 60}")
        answer = generate.remote(p)
        print(f"A: {answer}")
