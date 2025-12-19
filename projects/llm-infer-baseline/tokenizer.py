"""Tokenizer loading helper."""
from __future__ import annotations

from transformers import AutoTokenizer


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only 模型需要左填充，确保生成时 padding 不影响输出
    tokenizer.padding_side = "left"
    return tokenizer
