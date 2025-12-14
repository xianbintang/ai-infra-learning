# AI Infra · 推理优化学习大纲 & 计划

> **一句话画像**：拥有扎实模型训练背景、熟悉 Transformer 结构，目标是转向 AI 推理系统/Infra，聚焦“入门即核心”的推理优化。

## 1. 学习目标

- 建立推理系统视角：理解一次推理请求在系统中的端到端链路。
- 掌握 latency vs throughput 权衡方法，能用数据验证批处理对性能的影响。
- 深入理解 KV Cache：知道它存什么、如何提速、显存代价何在。
- 形成推理服务架构直觉：请求队列、调度、GPU Worker、CPU↔GPU 数据流。
- 打造可复用的实验与观测框架，为后续 TensorRT / vLLM / Triton 等优化铺路。

## 2. 阶段化路线（Phase D1 → D5）

| Phase | 重点主题 | 关键问题 | 产出 |
| --- | --- | --- | --- |
| **D1 推理请求基础** | 请求链路、算子执行顺序 | “一次推理到底算了什么？” | 逐步剖析 tokenizer → embedding → attention → decode 的耗时 | 
| **D2 性能瓶颈认知** | 系统瓶颈、IO、显存访问 | “推理慢一定是算力不够吗？” | 明确数据流/调度/显存访问的重要性 | 
| **D3 核心优化武器** | Batching、KV Cache、并行 | “牺牲谁换谁？” | TTFB、Token/s 与 Batch 的定量观察；KV Cache 表格对比 | 
| **D4 推理系统架构** | 调度、队列、多租户 | “一个真实推理服务长什么样？” | 绘制 end-to-end 流程图 + 模块职责 | 
| **D5 延伸方向** | 性能/工程/系统三条支线 | “下一步怎么深挖？” | 形成个人发展选项（Kernel、Serving、稳定性） |

## 3. 1–3 天实战冲刺计划（对应 `projects/llm-infer-baseline`）

| Day | 目标 | 关键动作 | 衡量指标 |
| --- | --- | --- | --- |
| **Day 1** | 最小可观测推理基线 | 完成模型/Tokenizer/metrics 脚手架，记录 `load/tokenize/generate/decode` 分段耗时 | 单请求 latency、token/s、硬件环境记录 |
| **Day 2** | Batching 权衡 | 实现 batcher（1/4/8 + 等待窗口），对比 latency/throughput，分析 GPU 利用率 | 三组 batch 数据、结论性文字 | 
| **Day 3** | KV Cache 认知跃迁 | 实验 `prompt_len = 10/50/200` × `KVCache on/off`，计算每 token latency | 表格 + 结论：“何时救命/何时吃显存” |

## 4. 学习 Checklist（可复制为进度墙）

| 类别 | 检查点 | 状态 | 备注 |
| --- | --- | --- | --- |
| 基础 | ✅ 明确设备/精度假设（CPU vs GPU / FP32 vs FP16） | ☐ | `config.py` 中记下硬件信息 |
| 基础 | ✅ 完成最小推理脚本并打印分段 STATS | ☐ | 记录第一份基准数据 |
| Batch | ✅ 实现动态 batcher（固定等待窗口） | ☐ | 记录不同 batch 的 latency & throughput |
| Batch | ✅ 给出“牺牲谁换谁”的文字结论 | ☐ | 用真实数字说明 | 
| KV Cache | ✅ 复现 `past_key_values` on/off 实验 | ☐ | 生成表格（prompt 长度 × 时间）|
| 系统 | ✅ 绘制简化的推理服务流程图 | ☐ | CPU/Queue/GPU Worker/Decode | 
| 系统 | ✅ 列出下一步扩展方向（Serving、Kernel、系统工程） | ☐ | 结合 Phase D5 |

> 建议使用表格中的“状态”列打 `☐ / ☑`，并附测量数据链接或日志路径，确保能回溯每次实验。

## 5. 观测与记录模板

```
[Experiment ID]
Hardware   : (e.g., 1×A100 40G / MacBook M2)
Model      : distilgpt2 @ FP16
Prompt     : "..."
Max Tokens : 50
Batch size : 1 / 4 / 8
Notes      : e.g., waiting window = 10 ms

Metrics
-------
load_model        : 0.00 s
load_tokenizer    : 0.00 s
prefill           : 0.00 s
per_token_latency : [list]
throughput        : xx tokens/s
Observations      : ...
```

## 6. 后续延伸建议

1. **工程线**：引入 FastAPI + 队列，观察请求排队对延迟的影响。
2. **性能线**：尝试 Torch.compile / TensorRT / FlashAttention，对比 kernel 时间线。
3. **系统线**：研究 vLLM、Triton、TensorRT-LLM 的调度策略，结合本项目的实验结论理解设计动机。

---

将此文档视作“作战图”；每次实验后回来更新 Checklist 与观测记录，让路线动态而可追踪。
