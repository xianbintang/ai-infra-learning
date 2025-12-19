"""Day 1 baseline inference script."""
from __future__ import annotations

import torch

from config import MODEL_NAME, DEVICE, describe_environment
from metrics import log_stats, timed
from model import load_model
from tokenizer import load_tokenizer

PROMPT = "Q: Explain why batching improves GPU utilization.\nA:"
MAX_NEW_TOKENS = 50


def _make_whitespace_visible(text: str) -> str:
    return text.replace("\r", "\\r").replace("\t", "\\t").replace("\n", "\\n")


def main() -> None:
    print("Environment:", describe_environment())
    print("Prompt:", PROMPT)
    stats = {}

    with timed("load_model_and_tokenizer", stats):
        model = load_model(MODEL_NAME)
        tokenizer = load_tokenizer(MODEL_NAME)

    with timed("tokenize", stats):
        inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

    with timed("prefill", stats):
        with torch.no_grad():
            prefill_outputs = model(
                input_ids=inputs["input_ids"],
                use_cache=True,
            )
    logits = prefill_outputs.logits
    past_key_values = prefill_outputs.past_key_values
    next_token_logits = logits[:, -1, :]
    print("logits shape:", logits.shape)
    next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    print("next_token_ids:", next_token_ids)
    print("next_token_logits:", next_token_logits)
    print("past_key_values:", past_key_values)
    print("logits:", logits)
    

    # with timed("forward_generate", stats):
    #     with torch.no_grad():
    #         outputs = model.generate(
    #             **inputs,
    #             max_new_tokens=MAX_NEW_TOKENS,
    #             do_sample=False,
    #         )

    with timed("decode", stats):
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n=== OUTPUT ===")
    if generated_text.strip() == "":
        print("(模型主要生成了空白字符；下面用可见形式展示生成结果)")
        print(_make_whitespace_visible(generated_text))
    else:
        print(generated_text)

    print("\n=== FULL TEXT (prompt + generated) ===")
    print(full_text)
    print()
    print(log_stats(stats))


if __name__ == "__main__":
    main()
