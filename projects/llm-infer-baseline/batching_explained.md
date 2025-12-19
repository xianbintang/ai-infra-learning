# Day 2: Batching 深度解析

## 1. 为什么需要 Batching？

### 1.1 单请求推理的问题

回顾 Day 1 的实验数据：

```
单请求推理 (distilgpt2, CPU):
- Prefill: 434 ms (13 tokens)
- Decode:  40 ms (9 steps)
- 生成速度: 21 tokens/s
```

问题：**GPU/CPU 利用率低**

```
时间线（单请求）:
请求1: [====Prefill====][==Decode==Decode==Decode==]
GPU:   [####利用中####][##部分利用##..................]
              ↑ 计算密集           ↑ 内存带宽密集，计算单元空闲
```

### 1.2 Batching 的核心思想

**把多个请求"打包"一起处理**，让计算单元保持忙碌：

```
时间线（Batch=4）:
请求1: [Prefill][Decode][Decode][Decode]
请求2: [Prefill][Decode][Decode][Decode]    → 合并成一次矩阵运算
请求3: [Prefill][Decode][Decode][Decode]
请求4: [Prefill][Decode][Decode][Decode]

GPU:   [############ 高利用率 ############]
```

### 1.3 数学直觉

单请求 Attention 计算：
```
Q: [1, seq_len, hidden]
K: [1, seq_len, hidden]
V: [1, seq_len, hidden]
计算量 = O(seq_len² × hidden)
```

Batch=4 的 Attention 计算：
```
Q: [4, seq_len, hidden]    # 4个请求堆叠
K: [4, seq_len, hidden]
V: [4, seq_len, hidden]
计算量 = O(4 × seq_len² × hidden)
```

**关键洞察**：计算量增加 4 倍，但 GPU 并行能力可以"消化"这些额外计算，总时间增加远小于 4 倍！

---

## 2. Batching 的 Trade-off

### 2.1 Throughput vs Latency

| 指标 | 定义 | Batching 影响 |
|------|------|---------------|
| **Throughput** | 单位时间处理的 token 数 | ↑ 提高（更高效利用硬件） |
| **Latency** | 单个请求从提交到完成的时间 | ↑ 增加（需要等待凑 batch） |

```
Batch=1 (低延迟，低吞吐):
请求到达 → 立即处理 → 返回
延迟: 100ms
吞吐: 10 req/s

Batch=4 (高延迟，高吞吐):
请求到达 → 等待凑齐4个 → 批量处理 → 返回
延迟: 150ms (等待 + 处理)
吞吐: 30 req/s
```

### 2.2 牺牲谁换谁？

```
┌─────────────────────────────────────────────────────┐
│                  Trade-off 曲线                      │
│                                                     │
│  Latency                                            │
│     ↑                                               │
│     │     ×                                         │
│     │       ×    ← Batch=8 (高吞吐，高延迟)          │
│     │         ×                                     │
│     │           ×  ← Batch=4                        │
│     │              ×                                │
│     │                 × ← Batch=1 (低吞吐，低延迟)   │
│     └────────────────────────────────→ Throughput   │
└─────────────────────────────────────────────────────┘
```

### 2.3 实际业务场景

| 场景 | 优先级 | 推荐 Batch |
|------|--------|-----------|
| 实时对话（ChatGPT） | 低延迟 | 小 batch (1-4) |
| 批量文档处理 | 高吞吐 | 大 batch (8-32) |
| API 服务 | 平衡 | 动态 batch |

---

## 3. 动态 Batching 机制

### 3.1 静态 vs 动态 Batching

**静态 Batching**:
- 固定 batch size
- 必须等够 N 个请求才处理
- 问题：请求少时延迟爆炸

**动态 Batching**:
- 设置等待窗口（如 10ms）
- 窗口内收集请求，窗口结束立即处理
- 兼顾延迟和吞吐

### 3.2 等待窗口策略

```python
# 动态 batcher 伪代码
class DynamicBatcher:
    def __init__(self, max_batch_size=8, wait_window_ms=10):
        self.queue = []
        self.max_batch_size = max_batch_size
        self.wait_window_ms = wait_window_ms
    
    def collect_batch(self):
        # 等待窗口时间
        time.sleep(self.wait_window_ms / 1000)
        
        # 取出队列中的请求（最多 max_batch_size 个）
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        
        return batch
```

### 3.3 Padding 问题

不同长度的请求需要 padding 到相同长度：

```
请求1: "Hello"           → [15496, PAD, PAD, PAD, PAD]
请求2: "How are you?"    → [2437, 389, 345, 30, PAD]
请求3: "What is AI?"     → [2061, 318, 9552, 30, PAD]
请求4: "Hi"              → [17250, PAD, PAD, PAD, PAD]
```

**问题**：Padding 带来无效计算
**解决**：Attention Mask 忽略 padding 位置

---

## 4. Batching 对 KV Cache 的影响

### 4.1 KV Cache 形状变化

单请求 KV Cache:
```
K: [1, num_heads, seq_len, head_dim]
V: [1, num_heads, seq_len, head_dim]
```

Batch=4 KV Cache:
```
K: [4, num_heads, seq_len, head_dim]  # batch 维度 ×4
V: [4, num_heads, seq_len, head_dim]
显存占用 ×4!
```

### 4.2 显存瓶颈

```
显存占用公式:
KV Cache 大小 = 2 × batch × layers × heads × seq_len × head_dim × dtype_bytes

distilgpt2 示例 (batch=4, seq_len=100):
= 2 × 4 × 6 × 12 × 100 × 64 × 4 bytes
= 14.7 MB

大模型示例 (Llama-7B, batch=32, seq_len=2048):
= 2 × 32 × 32 × 32 × 2048 × 128 × 2 bytes
= 17.2 GB  ← 显存爆炸！
```

### 4.3 Continuous Batching (预告)

传统 Batching 的问题：
```
请求1: [Prefill][Decode][Decode][DONE]
请求2: [Prefill][Decode][Decode][Decode][Decode][DONE]
请求3: [Prefill][Decode][Decode][Decode][DONE]

GPU:   [########][######][######][##闲置##][DONE]
                                    ↑ 等待最慢的请求
```

Continuous Batching（vLLM/TGI 使用）：
```
请求1: [Prefill][Decode][DONE]
请求2: [Prefill][Decode][Decode][Decode][DONE]
请求3:          [Prefill插入][Decode][Decode][DONE]
请求4:                        [Prefill插入][Decode][DONE]

GPU:   [####始终满载####]
```

---

## 5. 实验设计

### 5.1 实验变量

| 变量 | 值 |
|------|-----|
| Batch Size | 1, 4, 8 |
| Wait Window | 0ms, 10ms, 50ms |
| Prompt | 固定相同 prompt |
| Max New Tokens | 32 |

### 5.2 测量指标

| 指标 | 计算方式 |
|------|----------|
| **Avg Latency** | 从请求提交到返回的平均时间 |
| **P95 Latency** | 95% 请求的延迟上限 |
| **Throughput** | 总 token 数 / 总时间 |
| **GPU Utilization** | nvidia-smi 观测 |

### 5.3 预期结果

| Batch | Latency | Throughput | GPU Util |
|-------|---------|------------|----------|
| 1     | 最低 ✓  | 最低       | 低       |
| 4     | 中等    | 中等       | 中等     |
| 8     | 最高    | 最高 ✓    | 高 ✓    |

---

## 6. 核心概念总结

```
┌──────────────────────────────────────────────────────────────┐
│                    Batching 核心要点                          │
├──────────────────────────────────────────────────────────────┤
│ 1. Batching 提高 GPU 利用率，是推理优化的基础手段              │
│ 2. Latency 和 Throughput 是 trade-off 关系                   │
│ 3. 动态 Batching 通过等待窗口平衡延迟和吞吐                   │
│ 4. Padding 带来额外计算开销，需要 attention mask             │
│ 5. KV Cache 显存随 batch size 线性增长                       │
│ 6. Continuous Batching 是更先进的调度策略 (Day 4+)           │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. 接下来

1. 实现 `batcher.py` - 动态 batching 逻辑
2. 实现 `batch_experiment.py` - 批量实验脚本
3. 运行实验，收集数据
4. 分析 trade-off，写结论

让我们开始写代码！

