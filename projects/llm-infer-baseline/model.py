"""Model loading helper."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from config import DEVICE, DTYPE


def load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE)
    model.to(DEVICE)
    model.eval()
    return model
