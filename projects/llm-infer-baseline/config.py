"""Global configuration for the LLM inference baseline project."""
from __future__ import annotations

import platform

import torch

MODEL_NAME = "distilgpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def describe_environment() -> str:
    """Return a string summarizing hardware assumptions for logging."""
    chip = platform.processor() or platform.machine()
    if DEVICE == "cuda":
        gpu = torch.cuda.get_device_name(0)
        return f"GPU={gpu}, dtype={DTYPE}, device={DEVICE}"
    return f"CPU={chip}, dtype={DTYPE}, device={DEVICE}"
