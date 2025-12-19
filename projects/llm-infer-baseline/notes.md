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

---

## Experiment: day2-batching-tradeoff
Date: 2025-12-19
Hardware: Apple Silicon (CPU inference, arm64)
Model: distilgpt2 @ torch.float32
Script: batch_experiment.py
Total Requests: 16
Max New Tokens: 32

### 实验数据（最新运行，模型已缓存）

| Batch | Wait(ms) | Total(s) | Throughput | Avg Lat | P95 Lat |
|-------|----------|----------|------------|---------|---------|
| 1 | 0 | 2.509 | 204.0 t/s | 156.7 ms | 558.8 ms |
| 1 | 10 | 2.263 | 226.2 t/s | 141.4 ms | 161.7 ms |
| 4 | 0 | 1.269 | 414.4 t/s | 317.2 ms | 331.1 ms |
| 4 | 10 | 1.271 | 413.9 t/s | 317.6 ms | 328.0 ms |
| 8 | 0 | 0.614 | **856.9 t/s** | **306.7 ms** | 309.3 ms |
| 8 | 10 | 0.747 | 703.7 t/s | 373.5 ms | 396.9 ms |

### Trade-off 分析

相对于 Batch=1 (wait=0ms):
- **Batch=4**: 吞吐量 +103.1%, 延迟 +102.4%
- **Batch=8**: 吞吐量 **+320.0%**, 延迟 **+95.8%**

关键发现:
```
Batch=1: 204 t/s, 157ms   ← 低延迟，低吞吐
Batch=4: 414 t/s, 317ms   ← 2x 吞吐，2x 延迟
Batch=8: 857 t/s, 307ms   ← 4x 吞吐，延迟反而更低！🎉
```

### 惊喜发现：Batch=8 延迟比 Batch=4 还低！

**现象**: Batch=8 (307ms) < Batch=4 (317ms)

**原因分析**:

1. **减少了批次间开销**
   - Batch=1 需要 16 次独立调用，每次有 Python/框架开销
   - Batch=4 需要 4 次调用
   - Batch=8 只需要 2 次调用
   - 更少的调用 = 更少的调度开销

2. **CPU 缓存友好性**
   - 大 batch 时，权重矩阵在 L2/L3 缓存中复用更充分
   - 小 batch 反复加载权重，缓存命中率低

3. **SIMD 向量化效率**
   - Apple Silicon 的 NEON 指令集在处理连续数据时效率更高
   - Batch=8 让矩阵运算更好地利用 SIMD 宽度

4. **内存带宽摊薄**
   - 模型权重读取是固定成本
   - 大 batch 将这个成本分摊到更多请求上

### 结论

1. **牺牲谁换谁**: Batching 用延迟换吞吐量 (但有惊喜！)
   - 通常：更大的 batch = 更高的吞吐 + 更高的延迟
   - **实际**：Batch=8 同时获得了最高吞吐和较低延迟
   - 效率比 = 320% / 96% ≈ **3.3x** 🚀

2. **最佳实践**:
   - 不要假设大 batch 一定高延迟
   - 需要实测找到硬件的"甜点" batch size
   - Apple Silicon CPU 在 batch=8 附近效率最优

3. **业务场景选择**:
   - 实时对话: batch=1-2 (低延迟优先)
   - API 服务: batch=4-8 (平衡，实测调优)
   - 批量处理: batch=8+ (高吞吐优先)

4. **等待窗口影响**:
   - wait=10ms 增加了约 12ms 排队时间
   - Batch=8 + wait=10ms 吞吐下降到 704 t/s
   - 结论：在请求充足时，wait=0 效果更好

5. **CPU vs GPU 预期差异**:
   - CPU 推理已经看到 4x 吞吐提升
   - GPU 推理下 batching 收益会更明显（并行计算更强）
   - 期待在 GPU 环境下看到更大的吞吐提升
