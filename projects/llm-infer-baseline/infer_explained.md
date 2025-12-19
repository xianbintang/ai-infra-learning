# infer.py 逐行代码解释

## 文件概述
这是 Day1 的基线推理脚本，用于执行一次完整的 LLM 推理请求，并记录各个阶段的耗时。

---

## 逐行详细解释

### 第 1-2 行：文件文档字符串和导入
```python
"""Day 1 baseline inference script."""
from __future__ import annotations
```
- **第 1 行**：文件级别的文档字符串，说明这是 Day1 的基线推理脚本
- **第 2 行**：启用延迟类型注解（PEP 563），允许使用字符串形式的类型注解，提高前向引用的灵活性

### 第 4-9 行：导入依赖
```python
import torch

from config import MODEL_NAME, DEVICE, describe_environment
from metrics import log_stats, timed
from model import load_model
from tokenizer import load_tokenizer
```
- **第 4 行**：导入 PyTorch，用于张量操作和模型推理
- **第 6 行**：从 `config.py` 导入：
  - `MODEL_NAME`：模型名称（默认 "distilgpt2"）
  - `DEVICE`：计算设备（"cuda" 或 "cpu"）
  - `describe_environment()`：环境描述函数
- **第 7 行**：从 `metrics.py` 导入：
  - `log_stats`：格式化并打印统计信息的函数
  - `timed`：上下文管理器，用于测量代码块执行时间
- **第 8 行**：从 `model.py` 导入 `load_model` 函数，用于加载模型
- **第 9 行**：从 `tokenizer.py` 导入 `load_tokenizer` 函数，用于加载分词器

### 第 11-12 行：全局常量定义
```python
PROMPT = "Q: Explain why batching improves GPU utilization.\nA:"
MAX_NEW_TOKENS = 50
```
- **第 11 行**：定义输入提示词（prompt），这是一个问答格式的提示
- **第 12 行**：定义最大生成 token 数量为 50，限制生成文本的长度

### 第 15-16 行：辅助函数
```python
def _make_whitespace_visible(text: str) -> str:
    return text.replace("\r", "\\r").replace("\t", "\\t").replace("\n", "\\n")
```
- **第 15 行**：定义私有辅助函数（以下划线开头），将不可见的空白字符转换为可见的转义序列
- **第 16 行**：将回车符 `\r`、制表符 `\t`、换行符 `\n` 分别替换为字符串 `\r`、`\t`、`\n`，便于在输出中查看这些字符

### 第 19 行：主函数定义
```python
def main() -> None:
```
- 定义主函数，返回类型为 `None`，这是程序的入口点

### 第 20-22 行：初始化和打印环境信息
```python
    print("Environment:", describe_environment())
    print("Prompt:", PROMPT)
    stats = {}
```
- **第 20 行**：打印环境信息（硬件、设备类型、数据类型等）
- **第 21 行**：打印输入的提示词
- **第 22 行**：初始化空字典 `stats`，用于存储各个阶段的耗时统计

### 第 24-26 行：加载模型和分词器（计时）
```python
    with timed("load_model_and_tokenizer", stats):
        model = load_model(MODEL_NAME)
        tokenizer = load_tokenizer(MODEL_NAME)
```
- **第 24 行**：使用 `timed` 上下文管理器，测量模型和分词器加载的总耗时
  - `"load_model_and_tokenizer"` 是统计项的名称
  - `stats` 是存储统计信息的字典
- **第 25 行**：调用 `load_model` 加载预训练模型到指定设备
- **第 26 行**：调用 `load_tokenizer` 加载对应的分词器

### 第 28-29 行：文本分词（计时）
```python
    with timed("tokenize", stats):
        inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
```
- **第 28 行**：使用 `timed` 测量分词阶段的耗时
- **第 29 行**：
  - `tokenizer(PROMPT, ...)` 将文本转换为 token IDs
  - `return_tensors="pt"` 指定返回 PyTorch 张量格式
  - `.to(DEVICE)` 将张量移动到指定设备（CPU 或 GPU）

### 第 31-37 行：模型生成（计时）
```python
    with timed("forward_generate", stats):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
```
- **第 31 行**：使用 `timed` 测量模型生成阶段的耗时（包括 prefill 和 decode）
- **第 32 行**：`torch.no_grad()` 上下文管理器，禁用梯度计算以节省内存和加速推理
- **第 33-36 行**：调用模型的 `generate` 方法：
  - `**inputs` 解包输入字典（通常包含 `input_ids` 和 `attention_mask`）
  - `max_new_tokens=MAX_NEW_TOKENS` 限制生成的最大新 token 数
  - `do_sample=False` 使用贪婪解码（选择概率最高的 token），而非采样

### 第 39-43 行：解码生成结果（计时）
```python
    with timed("decode", stats):
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```
- **第 39 行**：使用 `timed` 测量解码阶段的耗时
- **第 40 行**：获取输入 prompt 的长度（最后一个维度的长度，即 token 数量）
- **第 41 行**：从完整输出中提取新生成的部分（去掉 prompt 部分）
  - `outputs[0]` 是第一个样本的输出序列
  - `[prompt_len:]` 切片操作，只取 prompt 之后的部分
- **第 42 行**：将生成的 token IDs 解码为文本（仅新生成的部分）
  - `skip_special_tokens=True` 跳过特殊 token（如 `<pad>`, `<eos>` 等）
- **第 43 行**：将完整的输出序列（prompt + 生成）解码为文本

### 第 45-50 行：打印生成结果
```python
    print("\n=== OUTPUT ===")
    if generated_text.strip() == "":
        print("(模型主要生成了空白字符；下面用可见形式展示生成结果)")
        print(_make_whitespace_visible(generated_text))
    else:
        print(generated_text)
```
- **第 45 行**：打印分隔符和标题
- **第 46 行**：检查生成的文本是否为空（去除首尾空白后）
- **第 47-48 行**：如果为空，打印提示信息并使用辅助函数显示空白字符
- **第 49-50 行**：否则直接打印生成的文本

### 第 52-53 行：打印完整文本
```python
    print("\n=== FULL TEXT (prompt + generated) ===")
    print(full_text)
```
- **第 52 行**：打印分隔符和标题
- **第 53 行**：打印完整的文本（包含原始 prompt 和生成的内容）

### 第 54-55 行：打印统计信息
```python
    print()
    print(log_stats(stats))
```
- **第 54 行**：打印空行
- **第 55 行**：调用 `log_stats` 函数格式化并打印所有阶段的耗时统计

### 第 58-59 行：程序入口
```python
if __name__ == "__main__":
    main()
```
- **第 58 行**：检查是否作为主程序运行（而非被导入）
- **第 59 行**：如果是主程序，调用 `main()` 函数开始执行

---

## 执行流程总结

1. **环境检查**：打印硬件和配置信息
2. **模型加载**：加载模型和分词器（首次运行会从 Hugging Face 下载）
3. **文本分词**：将 prompt 转换为 token IDs
4. **模型推理**：执行前向传播生成新 tokens
5. **结果解码**：将 token IDs 转换回文本
6. **结果展示**：打印生成文本和性能统计

## 性能指标

脚本会记录以下阶段的耗时：
- `load_model_and_tokenizer`：模型和分词器加载时间
- `tokenize`：文本分词时间
- `forward_generate`：模型生成时间（包括 prefill 和 decode）
- `decode`：文本解码时间

这些指标有助于理解推理过程的性能瓶颈。

