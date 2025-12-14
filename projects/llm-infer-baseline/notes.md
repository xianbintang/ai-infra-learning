# 实验记录模板

复制该模板记录每次实验，确保能够复现。

```
## Experiment: <name>
Date: <YYYY-MM-DD>
Hardware: <GPU / CPU 型号>
Model: distilgpt2 @ <dtype>
Prompt: "..."
Batch policy: <size / 等待窗口>
KV Cache: on/off

Metrics
-------
load_model        :
load_tokenizer    :
prefill           :
decode_loop       :
token/s           :
GPU Utilization   :
Notes             :
```

---

## Experiment: day1-baseline-setup
Date: 2024-XX-XX
Hardware: Apple Silicon M4 (CPU inference, no dedicated GPU)
Model: distilgpt2 @ fp32
Prompt: "Explain why batching improves GPU utilization."
Batch policy: single request (no batching)
KV Cache: default (on)

Metrics
-------
Status: Pending first successful run (Torch/Transformers installation required).
Notes : Repository scaffolded (`config.py`, `metrics.py`, `tokenizer.py`, `model.py`, `infer.py`), ready to execute once dependencies are installed via `pip install -r requirements.txt`.
