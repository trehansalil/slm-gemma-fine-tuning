"""Chat with fine-tuned Gemma 2 2B via PyTorch (MPS/CUDA/CPU)."""

import argparse
import os
import sys
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_MODEL_DIR = "models/sft_local/merged"
SYSTEM_MSG = "You are a helpful legal and financial assistant. Answer based only on provided context."


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_dir: str, tokenizer_dir: str | None = None):
    """Load model — merges LoRA adapter if adapter_config.json is present."""
    print(f"Loading model from {model_dir}...")
    device = get_device()
    tok_dir = tokenizer_dir or model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    adapter_config = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_config):
        from peft import PeftModel
        import json
        with open(adapter_config) as f:
            cfg = json.load(f)
        base_name = cfg.get("base_model_name_or_path", "google/gemma-2-2b")
        print(f"  Loading base model: {base_name}")
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(base, model_dir)
        print("  Merging adapter...")
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)

    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params/1e6:.1f}M params on {device}")
    return model, tokenizer


def build_prompt(tokenizer, system: str, user: str) -> list[int]:
    """Build ChatML-formatted prompt for Gemma 2."""
    # Gemma uses <start_of_turn>role\ncontent<end_of_turn>
    # System is prepended as first user turn since Gemma has no native system role
    prompt = (
        f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return tokenizer.encode(prompt, add_special_tokens=True)


def generate(model, tokenizer, prompt_ids: list[int],
             max_new_tokens: int = 256, temperature: float = 0.7,
             top_k: int = 50, top_p: float = 0.9,
             stream: bool = True) -> str:
    device = next(model.parameters()).device
    max_ctx = getattr(model.config, "max_position_embeddings", 8192)
    if len(prompt_ids) >= max_ctx:
        print(f"Warning: prompt ({len(prompt_ids)} tokens) fills context ({max_ctx}), truncating")
        prompt_ids = prompt_ids[-(max_ctx - 2):]
    if len(prompt_ids) + max_new_tokens > max_ctx:
        max_new_tokens = max(1, max_ctx - len(prompt_ids))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.eos_token_id
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    do_sample = temperature > 0

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        top_k=top_k if do_sample else 0,
        top_p=top_p if do_sample else 1.0,
        do_sample=do_sample,
        eos_token_id=[eos_id, eot_id],
        pad_token_id=tokenizer.pad_token_id or eos_id,
    )
    if streamer:
        gen_kwargs["streamer"] = streamer

    if streamer:
        thread = threading.Thread(target=model.generate, args=(input_ids,), kwargs=gen_kwargs)
        thread.start()
        chunks = []
        for chunk in streamer:
            chunks.append(chunk)
            sys.stdout.write(chunk)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        thread.join()
        return "".join(chunks).strip()
    else:
        output = model.generate(input_ids, **gen_kwargs)
        new_tokens = output[0][len(prompt_ids):]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


def interactive(model, tokenizer, system: str, max_tokens: int, temperature: float):
    print("\n--- Gemma 2 2B SFT Chat (PyTorch) ---")
    print("Type your question (or 'quit' to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        prompt_ids = build_prompt(tokenizer, system, user_input)
        print(f"[{len(prompt_ids)} prompt tokens]")
        sys.stdout.write("Gemma: ")
        sys.stdout.flush()
        generate(model, tokenizer, prompt_ids,
                 max_new_tokens=max_tokens, temperature=temperature, stream=True)
        print()


def main():
    parser = argparse.ArgumentParser(description="Chat with Gemma 2 2B SFT model")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR,
                        help="Path to model directory")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer directory (defaults to --model)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--system", type=str, default=SYSTEM_MSG,
                        help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.tokenizer)

    if args.prompt:
        prompt_ids = build_prompt(tokenizer, args.system, args.prompt)
        response = generate(model, tokenizer, prompt_ids,
                            max_new_tokens=args.max_tokens,
                            temperature=args.temperature)
        print(response)
    else:
        interactive(model, tokenizer, args.system, args.max_tokens,
                    args.temperature)


if __name__ == "__main__":
    main()
