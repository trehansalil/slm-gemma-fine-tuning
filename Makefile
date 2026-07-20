# ──────────────────────────────────────────────────────────────────────────────
# Gemma 2 2B — fine-tuning, conversion & inference pipeline
# ──────────────────────────────────────────────────────────────────────────────
#
# Model layout (models/):
#   base/                 google/gemma-2-2b base weights (PyTorch)
#   sft_modal/adapter/    QLoRA adapter from Modal training
#   sft_modal/merged/     Adapter merged into base (PyTorch)
#   sft_local/adapter/    QLoRA adapter from local training
#   sft_local/merged/     Adapter merged into base (PyTorch)
#   coreml_modal/         CoreML conversion of sft_modal/merged
#   coreml_local/         CoreML conversion of sft_local/merged
#
# Five runnable model variants:
#   1. Base (PyTorch)          — make chat-base
#   2. Modal SFT (PyTorch)     — make chat-modal
#   3. Local SFT (PyTorch)     — make chat-local
#   4. Modal SFT (CoreML)      — make chat-coreml-modal
#   5. Local SFT (CoreML)      — make chat-coreml-local
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: help \
        download-base download-adapter \
        upload tokenize \
        finetune-modal finetune-local \
        merge-modal merge-local \
        convert-coreml-modal convert-coreml-local \
        chat-base chat-modal chat-local chat-coreml-modal chat-coreml-local chat-remote \
        pipeline-modal pipeline-local \
        ARGS

# Directories
MODELS          := models
BASE_DIR        := $(MODELS)/base
MODAL_ADAPTER   := $(MODELS)/sft_modal/adapter
MODAL_MERGED    := $(MODELS)/sft_modal/merged
LOCAL_ADAPTER   := $(MODELS)/sft_local/adapter
LOCAL_MERGED    := $(MODELS)/sft_local/merged
COREML_MODAL    := $(MODELS)/coreml_modal
COREML_LOCAL    := $(MODELS)/coreml_local

help: ## Show all targets
	@echo "Usage: make <target> [ARGS=\"--flag value ...\"]"
	@echo ""
	@awk '\
		/^# Args: none/ { next } \
		/^# Args: same/ { sub(/^# Args: /, ""); args = "            " $$0 "\n"; next } \
		/^# Args:$$/ { next } \
		/^# --/ { sub(/^# /, ""); args = args "            " $$0 "\n"; next } \
		/^[a-zA-Z_-]+:.*## / { split($$0, a, ":.*## "); printf "  \033[36m%-24s\033[0m %s\n", a[1], a[2]; if (args != "") printf "%s", args; args = "" } \
	' $(MAKEFILE_LIST)

# ── Setup ─────────────────────────────────────────────────────────────────────

# Usage: make download-base
# Args: none
download-base: ## Download google/gemma-2-2b base model → models/base/
	python gemma2b/download_base.py --output $(BASE_DIR) $(ARGS)

# Usage: make download-adapter
# Args: none
download-adapter: ## Download Modal adapter → models/sft_modal/adapter/
	@mkdir -p $(MODAL_ADAPTER)
	modal volume get gemma2-2b-sft final/ $(MODAL_ADAPTER)/

# ── Data & Tokenization ──────────────────────────────────────────────────────

# Usage: make upload
# Args: none
upload: ## Upload SFT data to Modal volume
	modal run gemma2b/upload_data.py

# Usage: make tokenize
# Args: none
tokenize: ## Tokenize SFT dataset on Modal
	modal run gemma2b/finetune_modal.py::tokenize

# ── Fine-tuning ───────────────────────────────────────────────────────────────

# Usage: make finetune-modal [ARGS="--skip-tokenize"]
# Args:
# --skip-tokenize         Skip tokenization (already done)
finetune-modal: ## QLoRA fine-tune on Modal (8× A100)
	modal run --detach gemma2b/finetune_modal.py $(ARGS)

# Usage: make finetune-local [ARGS="--n-epochs 3 --lr 2e-4 ..."]
# Args:
# --base          (default=models/base) Path to pre-downloaded base model
# --n-epochs      (default=3)    Total training epochs
# --batch-size    (default=1)    Batch size
# --grad-accum    (default=8)    Gradient accumulation steps
# --lr            (default=2e-4) Learning rate
# --min-lr        (default=2e-5) Minimum learning rate
# --device        (default=auto) Force device (mps/cuda/cpu)
# --fresh                        Train from scratch (ignore checkpoints)
# --retokenize                   Re-tokenize the dataset
finetune-local: ## QLoRA fine-tune locally (MPS/CUDA)
	python gemma2b/finetune_local.py --base $(BASE_DIR) $(ARGS)

# ── Adapter Merging ──────────────────────────────────────────────────────────

# Usage: make merge-modal [ARGS="..."]
# Args: none
merge-modal: ## Merge Modal adapter into base → models/sft_modal/merged/
	python gemma2b/merge_adapter.py --adapter $(MODAL_ADAPTER) --output $(MODAL_MERGED) --base $(BASE_DIR) $(ARGS)

# Usage: make merge-local [ARGS="..."]
# Args: none
merge-local: ## Merge local adapter into base → models/sft_local/merged/
	python gemma2b/merge_adapter.py --adapter $(LOCAL_ADAPTER) --output $(LOCAL_MERGED) --base $(BASE_DIR) $(ARGS)

# ── CoreML Conversion ────────────────────────────────────────────────────────

# Usage: make convert-coreml-modal [ARGS="--seq-len 1024"]
# Args:
# --seq-len       (default=1024) Max sequence length
convert-coreml-modal: ## Convert Modal merged model → CoreML (models/coreml_modal/)
	python gemma2b/convert_coreml.py --model $(MODAL_MERGED) --output $(COREML_MODAL) $(ARGS)

# Usage: make convert-coreml-local [ARGS="--seq-len 1024"]
# Args:
# --seq-len       (default=1024) Max sequence length
convert-coreml-local: ## Convert local merged model → CoreML (models/coreml_local/)
	python gemma2b/convert_coreml.py --model $(LOCAL_MERGED) --output $(COREML_LOCAL) $(ARGS)

# ── Chat / Inference (5 variants) ────────────────────────────────────────────

# Usage: make chat-base [ARGS="--prompt 'Hello' --max-tokens 256"]
# Args:
# --prompt        (optional)     Single prompt (omit for REPL)
# --system        (optional)     System prompt
# --max-tokens    (default=256)  Max generation tokens
# --temperature   (default=0.7)  Sampling temperature
chat-base: ## Chat with base google/gemma-2-2b (PyTorch)
	python gemma2b/chat_pytorch.py --model $(BASE_DIR) $(ARGS)

# Usage: make chat-modal [ARGS="--prompt 'Hello'"]
# Args: same as chat-base
chat-modal: ## Chat with Modal fine-tuned model (PyTorch)
	python gemma2b/chat_pytorch.py --model $(MODAL_MERGED) $(ARGS)

# Usage: make chat-local [ARGS="--prompt 'Hello'"]
# Args: same as chat-base
chat-local: ## Chat with local fine-tuned model (PyTorch)
	python gemma2b/chat_pytorch.py --model $(LOCAL_MERGED) $(ARGS)

# Usage: make chat-coreml-modal [ARGS="--prompt 'Hello'"]
# Args:
# --prompt        (optional)     Single prompt (omit for REPL)
# --system        (optional)     System prompt
# --max-tokens    (default=256)  Max generation tokens
# --temperature   (default=0.7)  Sampling temperature
chat-coreml-modal: ## Chat with Modal CoreML model (Apple Silicon)
	python gemma2b/chat_coreml.py --model-dir $(COREML_MODAL) $(ARGS)

# Usage: make chat-coreml-local [ARGS="--prompt 'Hello'"]
# Args: same as chat-coreml-modal
chat-coreml-local: ## Chat with local CoreML model (Apple Silicon)
	python gemma2b/chat_coreml.py --model-dir $(COREML_LOCAL) $(ARGS)

# Usage: make chat-remote [ARGS="--prompt 'Hello'"]
# Args:
# --prompt        (optional)     Single prompt
chat-remote: ## Chat via Modal (remote GPU inference)
	modal run gemma2b/inference.py $(ARGS)

# ── Pipeline Shortcuts ───────────────────────────────────────────────────────

# Usage: make pipeline-modal
# Args: none
pipeline-modal: download-adapter merge-modal convert-coreml-modal ## Full Modal pipeline: download → merge → CoreML

# Usage: make pipeline-local
# Args: none
pipeline-local: merge-local convert-coreml-local ## Full local pipeline: merge → CoreML
