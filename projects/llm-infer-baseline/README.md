# 从 0 构建可观测的 LLM 推理服务（Day1-Day3）

> 项目目标：亲手构建一个可观测的 LLM 推理 baseline，用实测数据理解 batching、KV Cache 与调度对 latency/throughput 的影响。完成后可作为后续所有推理优化实验（TensorRT、vLLM、Flash Attention 等）的基准台。

## 0. 骨架目录（建议）

```
llm_infer_baseline/
├── config.py         # 设备/精度/模型名称等全局配置
├── metrics.py        # `timed()` 等观测工具
├── tokenizer.py      # tokenizer 封装
├── model.py          # 模型加载、设备搬运、dtype 控制
├── infer.py          # Day1 最小推理脚本
├── batcher.py        # Day2 动态 batching 逻辑
├── kv_experiment.py  # Day3 KV Cache 对比实验
└── notes.md          # 记录实验数据与结论
```

> 可以根据个人喜好调整命名，核心是保证“指标采集 + 结论”可追踪。

---

## Day 1 · Baseline 推理服务

**目的**：清楚一次推理请求的完整耗时构成，建立首份基准数据。

> 依赖：`pip install -r requirements.txt`（首次运行前），推荐 `python3` 运行并在 `projects/llm-infer-baseline/` 目录下执行 `python3 infer.py`。

1. **配置环境**
   - `config.py` 中定义 `DEVICE`, `DTYPE`, `MODEL_NAME = "distilgpt2"`。
   - 运行脚本前打印硬件信息（CPU/GPU/显存）。

2. **实现 metrics**
   - `timed(name: str, stats: dict)` 上下文管理器，分段记录 `load_model/tokenize/prefill/decode`。

3. **模型 & Tokenizer**
   - `AutoModelForCausalLM` + `AutoTokenizer`；确保 `pad_token` 补全。
   - `model.eval()`，`torch.no_grad()`。

4. **最小推理脚本** (`infer.py`)
   - Prompt: `"Explain why batching improves GPU utilization."`
   - `max_new_tokens = 50`, `do_sample = False`。
   - 打印生成文本 + STATS 表。

5. **交付物**
   - `notes.md` 更新：硬件、Prompt、各阶段耗时、Token/s。
   - 完成 Checklist 中的 Day1 项目。

> ✅ Day1 完成后，你应该能回答：“一次请求耗时在哪里？prefill vs decode 谁更慢？”。

---

## Day 2 · Batching & Latency/Throughput Tradeoff

**目的**：量化 batch size 调整对 latency/throughput 的影响，形成“牺牲谁换谁”的工程直觉。

1. **实现 `batcher.py`**
   - 简单队列：请求进入后等待固定窗口（例如 10ms），再合并为 batch。
   - 实验 batch size = 1 / 4 / 8，保持 `max_new_tokens = 32`（便于比较）。
   - 记录每个请求的排队时间、模型执行时间、总 latency。

2. **实验用例**
   - 构造多个并发请求（可脚本模拟）：同 prompt 或不同 prompt 皆可。
   - 同时记录 GPU 利用率（如 `nvidia-smi --loop=1`），观察是否吃满。

3. **数据记录表**

| Batch Size | Wait Window (ms) | Avg Latency (s) | P95 Latency (s) | Throughput (req/s or token/s) | 备注 |
| --- | --- | --- | --- | --- | --- |

4. **交付物**
   - `notes.md` 中写出数据 + 结论：“为了 throughput 牺牲了多少 latency？”
   - 更新 Checklist，标记完成 `batcher` 与结论项。

---

## Day 3 · KV Cache 深度实验

**目的**：用实测数据理解 KV Cache 如何改变长上下文的性能轨迹。

1. **脚本结构** (`kv_experiment.py`)
   - 函数 `run_experiment(prompt_len: int, use_kv_cache: bool) -> dict`。
   - 构造不同长度的 prompt（随机文本或复制句子），生成 50 个 tokens。
   - 记录：总耗时、prefill 耗时、decode per-token latency（可存列表/均值）。

2. **实验矩阵**

| Prompt Length | KV Cache | Total Time (s) | Avg Token Latency (ms) | Prefill Time (s) | 观察 |
| --- | --- | --- | --- | --- | --- |

3. **观察重点**
   - 长 prompt 下无 KV Cache 的 decode 将接近“指数级”变慢；
   - 有 KV Cache 时 per-token latency 应接近常数，但显存占用会升高；
   - 结合 vLLM Paged KV Cache、Flash Attention 的设计动机写一段感想。

4. **交付物**
   - 更新 `notes.md`：贴表格 + 结论；
   - Checklist 中标记 KV Cache 相关条目。

---

## Checklist（项目内版本）

| Day | 项目 | 状态 | 证据 / 链接 |
| --- | --- | --- | --- |
| Day1 | 记录硬件 + dtype 假设 | ☐ |  | 
| Day1 | 完成 `infer.py` 并输出分段 STATS | ☐ |  | 
| Day1 | 拆解 generate：prefill vs decode loop | ☐ |  | 
| Day2 | 实现 batcher & 日志（队列等待 + 批大小） | ☐ |  |
| Day2 | 表格化 latency/throughput，对比 3 组 batch | ☐ |  |
| Day2 | 写出 trade-off 结论（牺牲谁换谁） | ☐ |  |
| Day3 | 复现 KV Cache on/off 实验 | ☐ |  |
| Day3 | 绘制 prompt 长度 × latency 表格 | ☐ |  |
| Day3 | 写下“KV Cache 什么时候救命/什么时候吃显存” | ☐ |  |

> 使用 `☐` → `☑` 表示完成，必要时附上日志文件路径或截图位置，方便未来回顾。

---

## 参考提示

- **观测优先**：任何优化前，先保证 metrics 可信。`time.perf_counter()`、GPU 利用率采样、日志结构化输出。
- **一次只改一件事**：尤其在 Day2/Day3，修改 batcher 或 KV Cache 时，控制变量，避免同时改变 prompt 或 max tokens。
- **记录失败**：若实验效果不如预期，照样写进 `notes.md`，这些记录对后续优化极具价值。

完成此项目后，你会：

1. 能用数据解释为什么推理慢；
2. 知道如何调度请求提高吞吐；
3. 对 KV Cache 的价值与代价有定量认识；
4. 拥有一个可迭代的推理性能基准台，随时可叠加更复杂的优化。
