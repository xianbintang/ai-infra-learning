"""Day 1 baseline inference script."""
from __future__ import annotations

import torch

from config import MODEL_NAME, DEVICE, describe_environment
from metrics import log_stats, timed
from model import load_model
from tokenizer import load_tokenizer

PROMPT = "Explain why batching improves GPU utilization."
MAX_NEW_TOKENS = 50


def main() -> None:
    print("Environment:", describe_environment())
    print("Prompt:", PROMPT)
    stats = {}

    with timed("load_model_and_tokenizer", stats):
        model = load_model(MODEL_NAME)
        tokenizer = load_tokenizer(MODEL_NAME)

    with timed("tokenize", stats):
        inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

    with timed("forward_generate", stats):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

    with timed("decode", stats):
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n=== OUTPUT ===")
    print(text)
    print()
    print(log_stats(stats))


if __name__ == "__main__":
    main()
