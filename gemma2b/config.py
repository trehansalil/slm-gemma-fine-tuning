"""Configuration for Gemma 2 2B QLoRA fine-tuning."""

PROJECT = "gemma2-2b-sft"
VOLUME_NAME = "gemma2-2b-sft"
DATA_ROOT = "/data"

SFT_DATA_PATH = f"{DATA_ROOT}/sft_train.jsonl"
TOKENIZED_DIR = f"{DATA_ROOT}/tokenized"
CKPT_DIR = f"{DATA_ROOT}/checkpoints"
FINAL_MODEL_DIR = f"{DATA_ROOT}/final"

BASE_MODEL = "google/gemma-2-2b"
HF_SECRET_NAME = "huggingface-secret"

MAX_SEQ_LEN = 1024
IGNORE_INDEX = -100

NUM_GPUS = 8
GPU_TYPE = "A100"

# Local paths (relative to project root)
LOCAL_MODEL_DIR = "models/sft_local"
LOCAL_TOKENS_DIR = f"{LOCAL_MODEL_DIR}/tokens"
LOCAL_CKPT_DIR = f"{LOCAL_MODEL_DIR}/checkpoints"
LOCAL_OUTPUT_DIR = f"{LOCAL_MODEL_DIR}/adapter"
LOCAL_METRICS_PATH = f"{LOCAL_MODEL_DIR}/metrics.jsonl"

SFT_DATA_LOCAL = "data/sft_train.jsonl"
