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
Date: 2024-12-19
Hardware: Apple Silicon (CPU inference, no dedicated GPU, arm64)
Model: distilgpt2 @ torch.float32
Prompt: "Q: Explain why batching improves GPU utilization.\nA:"
Batch policy: single request (no batching)
KV Cache: on (use_cache=True)

Metrics
-------
load_model+tok    : 98.30 s (首次加载，含网络重试)
tokenize          : 2.51 ms (13 tokens)
prefill           : 434.26 ms (13 tokens, 33.40 ms/token)
decode_loop       : 40.70 ms (9 steps, 4.52 ms/step avg)
total_generate    : 474.96 ms
token/s           : 21.05 tokens/s
GPU Utilization   : N/A (CPU only)

Notes:
- 环境: macOS arm64, torch.float32, CPU
- Prompt 长度: 13 tokens
- 生成长度: 10 tokens (含首个 token)
- 生成文本: "The GPU is a very powerful GPU, and it"
- KV Cache 最终 shape: (1, 12, 22, 64) = [batch, heads, seq_len=13+9, head_dim]

关键发现:
- Prefill 每 token (33.40 ms) 远慢于 Decode 每 step (4.52 ms)
  - 原因: Prefill 需要对所有输入 token 做完整的 attention 计算
  - Decode 有 KV Cache 加速，只需处理 1 个新 token
- Decode 每步耗时稳定在 4-5 ms，说明 KV Cache 有效复用
- 模型加载耗时长是因为 HuggingFace 网络重试 (ConnectionResetError)

---

## Experiment: day1-prefill-decode-deep-dive
Date: 2024-12-19
Hardware: Apple Silicon (CPU inference, arm64)
Model: distilgpt2 @ torch.float32
Script: inspect_prefill_decode.py

目的: 手动拆解 Prefill 和 Decode 阶段，观察 KV Cache 变化

Prefill 阶段:
- 输入: 13 tokens
- 输出: logits (1, 13, 50257), KV Cache 6 层 × (1, 12, 13, 64)
- 首个 token 选择: argmax(logits[0, -1]) -> ID 383 " The"

Decode 阶段 (每步):
- 输入: 1 token + past_key_values
- 输出: logits (1, 1, 50257), 更新后的 KV Cache
- KV Cache seq_len: 13 -> 14 -> 15 -> ... -> 22

Decode 各步耗时 (ms):
| Step | 耗时 | KV seq_len | 选中 token |
|------|------|------------|------------|
| 1    | 5.25 | 14         | " GPU"     |
| 2    | 4.85 | 15         | " is"      |
| 3    | 4.14 | 16         | " a"       |
| 4    | 4.68 | 17         | " very"    |
| 5    | 4.34 | 18         | " powerful"|
| 6    | 4.60 | 19         | " GPU"     |
| 7    | 4.45 | 20         | ","        |
| 8    | 4.36 | 21         | " and"     |
| 9    | 4.03 | 22         | " it"      |

结论:
- 手动实现 generate() 验证了 Prefill + Decode 的核心逻辑
- KV Cache 是加速 Decode 的关键，seq_len 线性增长
- 每步 Decode 耗时稳定，不随 seq_len 显著增长 (在小规模下)
- 理解这些机制是优化 LLM 推理的基础
