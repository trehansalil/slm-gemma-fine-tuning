"""Training backend dispatcher — routes to MLX or PyTorch automatically.

On Apple Silicon with MLX installed, uses the MLX backend for ~2-4x faster
training. Falls back to PyTorch MPS/CUDA/CPU otherwise.
"""
from __future__ import annotations

import sys


def is_mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
        return sys.platform == "darwin"
    except ImportError:
        return False


def train_auto(model_path: str, tokenizer, features: list[dict],
               output_dir: str, device, **kwargs):
    """Dispatch to MLX or PyTorch based on availability.

    Accepts the same kwargs as train_utils.train() and train_mlx.train().
    """
    if is_mlx_available() and device.type in ("mps", "cpu"):
        print("Using MLX backend (Apple Silicon)")
        from shared.train_mlx import train as mlx_train
        return mlx_train(model_path, features, output_dir, **kwargs)
    else:
        print(f"Using PyTorch backend ({device})")
        from transformers import AutoModelForCausalLM
        from shared.train_utils import get_attn_impl, get_dtype, maybe_compile, train

        dtype = get_dtype(device)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype,
            attn_implementation=get_attn_impl(device))
        model = maybe_compile(model, device)
        model.to(device)
        return train(model, tokenizer, features, output_dir, device, **kwargs)
