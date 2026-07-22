# Push to Hugging Face Hub

## When to use
When the user says `/push-to-hf`, wants to upload a model to Hugging Face, or asks to publish/share their fine-tuned model.

## Arguments
- `MODEL_PATH` (optional): Path to the model directory (e.g., `models/slm125m/instruction`). If omitted, ask the user which model stage to push.
- `REPO_NAME` (optional): HuggingFace repo name (e.g., `username/slm-125m-instruction-tuned`). If omitted, ask the user.
- `--private` (optional): Create a private repo. Default is public.

## Instructions

Follow these steps exactly:

### 1. Identify the model to push

Check which model directories exist:
```bash
ls models/slm125m/
ls models/instruction_local/ 2>/dev/null
ls models/gemma2b/ 2>/dev/null
```

If no MODEL_PATH argument was given, list the available stages and ask the user which one to push. Each stage directory should contain at minimum: `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`.

### 2. Verify HuggingFace authentication

```bash
python -c "from huggingface_hub import HfApi; api = HfApi(); print(f'Authenticated as: {api.whoami()[\"name\"]}')"
```

If this fails, tell the user to run:
```bash
huggingface-cli login
```

### 3. Check for training metrics

Look for `metrics.jsonl` in the model directory. If found, parse it to extract:
- Final train loss
- Final val loss
- Final perplexity
- Number of epochs completed
- Total training steps

### 4. Generate a model card

Create a `README.md` in the model directory with this structure:

```markdown
---
license: apache-2.0
tags:
  - text-generation
  - fine-tuned
  - slm
  - instruction-tuning
  - dpo
  - rlaif
library_name: transformers
pipeline_tag: text-generation
---

# {Model Name}

## Model Description

Fine-tuned small language model from the [slm-gemma-fine-tuning](https://github.com/{user}/slm-gemma-fine-tuning) project.

**Base model:** {read from config.json → _name_or_path or architectures}
**Training stage:** {stage name from directory path}
**Pipeline:** SFT → Instruction Tuning → DPO → RLAIF

## Training Metrics

{Insert table from metrics.jsonl if available}

| Metric | Value |
|--------|-------|
| Final Train Loss | {value} |
| Final Val Loss | {value} |
| Perplexity | {value} |
| Epochs | {value} |
| Steps | {value} |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

prompt = "Your question here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
```

### 5. Upload to Hugging Face

Write and run a Python script:

```python
from huggingface_hub import HfApi, create_repo
import os

repo_id = "{REPO_NAME}"
model_dir = "{MODEL_PATH}"
private = {True/False}

# Create repo if it doesn't exist
try:
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
except Exception as e:
    print(f"Repo creation note: {e}")

api = HfApi()

# Upload all model files
files_to_upload = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "README.md",
]

for fname in files_to_upload:
    fpath = os.path.join(model_dir, fname)
    if os.path.exists(fpath):
        print(f"Uploading {fname}...")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=repo_id,
        )
        print(f"  ✓ {fname}")

# Optionally upload metrics
metrics_path = os.path.join(model_dir, "metrics.jsonl")
if os.path.exists(metrics_path):
    api.upload_file(
        path_or_fileobj=metrics_path,
        path_in_repo="metrics.jsonl",
        repo_id=repo_id,
    )
    print("  ✓ metrics.jsonl")

print(f"\nModel pushed to: https://huggingface.co/{repo_id}")
```

### 6. Confirm success

After upload completes, print the model URL and verify with:
```bash
python -c "from huggingface_hub import HfApi; files = HfApi().list_repo_files('{REPO_NAME}'); print('Files in repo:', files)"
```

Report the URL to the user: `https://huggingface.co/{REPO_NAME}`
