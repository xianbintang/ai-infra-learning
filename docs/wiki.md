# LLM Inference Baseline Wiki

此 Wiki 用来记录各个模块的作用与原理，方便你在学习每一行代码时快速回顾设计动机。

## config.py
1. `"""Global configuration for the LLM inference baseline project."""` — 说明此模块提供全局配置，是整个服务的入口点。
2. `from __future__ import annotations` — 启用延后注解解析，保持 Python 3.7+ 的类型提示更简洁。
3. `import platform` — 用来获取本机 CPU/芯片信息，便于记录运行环境。
4. `import torch` — PyTorch 是推理的核心依赖，在此模块用于判断设备与数据类型。
5. `MODEL_NAME = "distilgpt2"` — 固定使用一个可以在 Mac 上快速跑通的 decoder-only 模型。
6. `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` — 自动选择可用 GPU，否则回退到 CPU，确保跨机型可执行。
7. `DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32` — 在 GPU 时使用 FP16 以提高吞吐，CPU 上保持稳定的 FP32。
8. `def describe_environment() -> str:` — 封装一个可打印的环境描述，方便每次运行都能明确硬件假设。
9. `chip = platform.processor() or platform.machine()` — 尝试提取实际的芯片/处理器名称。
10. `if DEVICE == "cuda": ...` — 如果有 GPU，则补充 CUDA 设备名称，否则返回 CPU 信息，结果在 `infer.py` 中输出。

## metrics.py
1. `"""Utility helpers for timing inference steps."""` — 提醒使用者本文件负责分段计时。
2. `from __future__ import annotations` — 同样开启延迟评估类型注解。
3. `import time` 与 `from contextlib import contextmanager` — 用来实现基于 `time.perf_counter()` 的精确计时。
4. `from typing import Dict` — 用于类型注释 `stats` 字典。
5. `@contextmanager` 与 `def timed(name: str, stats: Dict[str, float]):` — 提供一个上下文管理器，进入/退出时分别记录开始与结束时间。
6. `start = time.perf_counter()` — 记录高精度起点。
7. `yield` — 上下文体在此执行，确保 `with timed(...):` 内的代码被包裹。
8. `stats[name] = time.perf_counter() - start` — 将耗时写入共享字典，便于后面统一展示。
9. `def log_stats(stats: Dict[str, float]) -> str:` — 将 `stats` 转成可打印的格式。
10. `lines.append(f"{key:20s}: {value:.4f}")` — 固定宽度对齐，便于日志阅读。

## tokenizer.py
1. `"""Tokenizer loading helper."""` — 简述模块用途。
2. `from __future__ import annotations` — 按规范启用。
3. `from transformers import AutoTokenizer` — 依赖 Hugging Face tokenizer。
4. `def load_tokenizer(model_name: str):` — 封装模型名称作为参数。
5. `tokenizer = AutoTokenizer.from_pretrained(model_name)` — 下载/加载 tokenizer。
6. `if tokenizer.pad_token is None:` — 某些 GPT 模型没有 pad token；为了 batcher 方便我们赋值。
7. `tokenizer.pad_token = tokenizer.eos_token` — 统一使用 eos 作为填充符。
8. `return tokenizer` — 返回给 `infer.py` 使用。

## model.py
1. `"""Model loading helper."""` — 标注模块作用。
2. `from __future__ import annotations` — 继续保持注解兼容性。
3. `import torch` — 用于 `torch_dtype` 与 `.to(DEVICE)`。
4. `from transformers import AutoModelForCausalLM` — 选择 decoder-only Causal LM API。
5. `from config import DEVICE, DTYPE` — 从配置模块获取设备与数据类型。
6. `def load_model(model_name: str):` — 读取模型名称，封装加载逻辑。
7. `AutoModelForCausalLM.from_pretrained(...)` — 指定 dtype，以控制精度和显存占用。
8. `model.to(DEVICE)` — 把模型搬到 CPU/GPU，以免后续再迁移。
9. `model.eval()` — 关闭训练分支，如 dropout，确保行为一致。
10. `return model` — 返回用于推理的模型。

## infer.py
1. `"""Day 1 baseline inference script."""` — 文件头说明当前脚本目的。
2. `from __future__ import annotations` — 统一启用。
3. `import torch` — 推理时的核心框架。
4. `from config import MODEL_NAME, DEVICE, describe_environment` — 从配置模块导入常量与环境描述函数。
5. `from metrics import log_stats, timed` — 引入计时工具。
6. `from model import load_model` — 模型加载辅助。
7. `from tokenizer import load_tokenizer` — tokenizer 加载辅助。
8. `PROMPT = ...` 与 `MAX_NEW_TOKENS = 50` — 固定 prompt 与要生成的 token 数。
9. `def main() -> None:` — 推理脚本的入口。
10. `print("Environment:", describe_environment())` — 输出运行时硬件信息，便于记录。
11. `print("Prompt:", PROMPT)` — 方便快速定位生成输入，尤其在多个实验时辨别 prompt。
12. `stats = {}` — 初始化空字典用于记录耗时。
13. `with timed("load_model_and_tokenizer", stats): ...` — 一次性加载模型和 tokenizer 并计时。
14. `with timed("tokenize", stats): inputs = tokenizer(...).to(DEVICE)` — 记录 tokenizer 部分耗时，并搬运张量到目标设备。
15. `with timed("forward_generate", stats):` — 包裹 `.generate()` 调用，衡量前向时间。
16. `with torch.no_grad(): outputs = model.generate(...)` — 生成时关闭梯度以节省内存。
17. `with timed("decode", stats): text = tokenizer.decode(...)` — 记录 decode 的耗时，帮助对比 prefill 与 decode 消耗。
18. `print("\n=== OUTPUT ===")` — 美化输出区。
19. `print(text)` — 显示生成文本，解答你提出的“没有输出”的疑问；该 print 已确保模型回答会展示在终端。
20. `print(log_stats(stats))` — 打印各阶段耗时，便于后续分析。
21. `if __name__ == "__main__": main()` — 允许脚本直接执行。

## requirements.txt
1. `torch` — 推理必需的框架，包含张量计算与 `AutoModelForCausalLM`。
2. `transformers` — Hugging Face 库，提供 tokenizer 与模型接口。

## .gitignore
1. `__pycache__`、`*.py[cod]` 等 — 屏蔽 Python 编译缓存。
2. `.venv` / `venv` / `env` — 避免虚拟环境文件入库。
3. `build/`, `dist/`, `*.egg-info/` — 打包产物。
4. `.DS_Store`、`.vscode/`、`.idea/` — IDE 与 macOS 特有文件。
5. `*.log`, `.env`, `*.cache` — 日志、环境变量文件与缓存。

> 将此 Wiki 作为“代码剖析总览”，在学习过程中随时补充新的观察与链接（例如 Day2 的 batcher 思路、Day3 的 KV Cache 表格），保持“代码 → 观测 → 理解”闭环。
