"""
Day 2: Batching 实验脚本

实验目标:
1. 对比 batch_size = 1, 4, 8 的性能差异
2. 测量 Latency vs Throughput 的 trade-off
3. 形成"牺牲谁换谁"的工程直觉
"""

import time
from typing import List, Dict, Any
import torch

from config import MODEL_NAME, DEVICE, describe_environment
from model import load_model
from tokenizer import load_tokenizer
from batcher import DynamicBatcher, run_batch_experiment


# ============================================================
# 实验配置
# ============================================================

# 测试 prompt 池
TEST_PROMPTS = [
    "Q: What is artificial intelligence?\nA:",
    "Q: Explain machine learning in simple terms.\nA:",
    "Q: What is deep learning?\nA:",
    "Q: How does natural language processing work?\nA:",
    "Q: What is a neural network?\nA:",
    "Q: Explain the transformer architecture.\nA:",
    "Q: What is GPU computing?\nA:",
    "Q: How does batch processing improve performance?\nA:",
    "Q: What is model inference?\nA:",
    "Q: Explain the concept of latency.\nA:",
    "Q: What is throughput in computing?\nA:",
    "Q: How does caching work?\nA:",
    "Q: What is parallel computing?\nA:",
    "Q: Explain CPU vs GPU.\nA:",
    "Q: What is memory bandwidth?\nA:",
    "Q: How does attention mechanism work?\nA:",
]

# 实验参数
BATCH_SIZES = [1, 4, 8]
WAIT_WINDOWS = [0, 10]  # ms
MAX_NEW_TOKENS = 32
NUM_REQUESTS = 16  # 总请求数


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def run_all_experiments() -> List[Dict[str, Any]]:
    """运行所有实验配置"""
    
    print_header("Day 2: Batching 实验")
    print(f"\n📋 实验配置:")
    print(f"  - 模型: {MODEL_NAME}")
    print(f"  - 设备: {DEVICE}")
    print(f"  - Batch sizes: {BATCH_SIZES}")
    print(f"  - Wait windows: {WAIT_WINDOWS} ms")
    print(f"  - Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  - 总请求数: {NUM_REQUESTS}")
    print(f"  - 环境: {describe_environment()}")

    # 加载模型
    print_header("加载模型")
    load_start = time.perf_counter()
    model = load_model(MODEL_NAME)
    tokenizer = load_tokenizer(MODEL_NAME)
    load_time = time.perf_counter() - load_start
    print(f"模型加载耗时: {load_time:.2f} s")

    # 准备测试数据
    prompts = (TEST_PROMPTS * ((NUM_REQUESTS // len(TEST_PROMPTS)) + 1))[:NUM_REQUESTS]
    print(f"\n准备了 {len(prompts)} 个测试请求")

    # 运行实验
    all_results = []

    print_header("开始实验")

    for batch_size in BATCH_SIZES:
        for wait_window in WAIT_WINDOWS:
            exp_name = f"batch={batch_size}, wait={wait_window}ms"
            print(f"\n▶ 运行实验: {exp_name}")

            result = run_batch_experiment(
                model=model,
                tokenizer=tokenizer,
                device=DEVICE,
                prompts=prompts,
                batch_size=batch_size,
                max_new_tokens=MAX_NEW_TOKENS,
                wait_window_ms=wait_window,
            )

            all_results.append(result)

            # 打印实验结果
            print(f"  ├─ 总耗时: {result['total_time_s']:.3f} s")
            print(f"  ├─ 总 tokens: {result['total_tokens']}")
            print(f"  ├─ 吞吐量: {result['throughput_tokens_s']:.2f} tokens/s")
            print(f"  ├─ 平均延迟: {result['latency_avg_ms']:.2f} ms")
            print(f"  ├─ P95 延迟: {result['latency_p95_ms']:.2f} ms")
            print(f"  └─ 平均排队: {result['queue_time_avg_ms']:.2f} ms")

    return all_results


def print_comparison_table(results: List[Dict[str, Any]]):
    """打印对比表格"""
    
    print_header("实验结果对比表")

    # 表头
    headers = ["Batch", "Wait(ms)", "Total(s)", "Tokens", "Throughput", "Avg Lat", "P95 Lat", "Queue"]
    widths = [6, 8, 8, 7, 12, 10, 10, 8]
    
    # 打印表头
    header_line = " | ".join(f"{h:^{w}}" for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    # 打印数据
    for r in results:
        row = [
            f"{r['batch_size']}",
            f"{r['wait_window_ms']:.0f}",
            f"{r['total_time_s']:.3f}",
            f"{r['total_tokens']}",
            f"{r['throughput_tokens_s']:.1f} t/s",
            f"{r['latency_avg_ms']:.1f} ms",
            f"{r['latency_p95_ms']:.1f} ms",
            f"{r['queue_time_avg_ms']:.1f} ms",
        ]
        print(" | ".join(f"{v:^{w}}" for v, w in zip(row, widths)))


def analyze_tradeoff(results: List[Dict[str, Any]]):
    """分析 Trade-off"""
    
    print_header("Trade-off 分析")

    # 找出关键数据点
    batch_1 = [r for r in results if r['batch_size'] == 1][0]
    batch_4 = [r for r in results if r['batch_size'] == 4][0]
    batch_8 = [r for r in results if r['batch_size'] == 8][0]

    # 计算变化
    throughput_gain_4 = (batch_4['throughput_tokens_s'] / batch_1['throughput_tokens_s'] - 1) * 100
    throughput_gain_8 = (batch_8['throughput_tokens_s'] / batch_1['throughput_tokens_s'] - 1) * 100

    latency_increase_4 = (batch_4['latency_avg_ms'] / batch_1['latency_avg_ms'] - 1) * 100
    latency_increase_8 = (batch_8['latency_avg_ms'] / batch_1['latency_avg_ms'] - 1) * 100

    print("\n📊 相对于 Batch=1 的变化:\n")

    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    Batch Size 对比                          │")
    print("├──────────┬──────────────────┬──────────────────────────────┤")
    print("│  指标    │     Batch=4      │         Batch=8              │")
    print("├──────────┼──────────────────┼──────────────────────────────┤")
    print(f"│ Throughput │ {throughput_gain_4:+.1f}% │ {throughput_gain_8:+.1f}% │")
    print(f"│ Latency    │ {latency_increase_4:+.1f}% │ {latency_increase_8:+.1f}% │")
    print("└──────────┴──────────────────┴──────────────────────────────┘")

    # 效率比
    efficiency_4 = throughput_gain_4 / max(latency_increase_4, 0.1) if latency_increase_4 > 0 else float('inf')
    efficiency_8 = throughput_gain_8 / max(latency_increase_8, 0.1) if latency_increase_8 > 0 else float('inf')

    print("\n💡 关键发现:")
    print(f"""
1. Throughput 变化:
   - Batch=4: 吞吐量 {"提升" if throughput_gain_4 > 0 else "下降"} {abs(throughput_gain_4):.1f}%
   - Batch=8: 吞吐量 {"提升" if throughput_gain_8 > 0 else "下降"} {abs(throughput_gain_8):.1f}%

2. Latency 变化:
   - Batch=4: 延迟 {"增加" if latency_increase_4 > 0 else "减少"} {abs(latency_increase_4):.1f}%
   - Batch=8: 延迟 {"增加" if latency_increase_8 > 0 else "减少"} {abs(latency_increase_8):.1f}%

3. 牺牲谁换谁?
   - Batch=4: 用 {abs(latency_increase_4):.1f}% 的延迟换 {abs(throughput_gain_4):.1f}% 的吞吐
   - Batch=8: 用 {abs(latency_increase_8):.1f}% 的延迟换 {abs(throughput_gain_8):.1f}% 的吞吐
""")

    # 业务建议
    print("\n📝 业务场景建议:")
    print("""
┌────────────────────────────────────────────────────────────────┐
│ 场景              │ 推荐 Batch │ 原因                           │
├───────────────────┼────────────┼────────────────────────────────┤
│ 实时对话 (Chat)   │ 1-2        │ 用户等待敏感，低延迟优先        │
│ API 服务          │ 4          │ 平衡延迟和吞吐                  │
│ 批量处理          │ 8+         │ 高吞吐优先，延迟不敏感          │
│ 离线分析          │ 最大可能    │ 最大化硬件利用率                │
└────────────────────────────────────────────────────────────────┘
""")


def generate_notes_update(results: List[Dict[str, Any]]) -> str:
    """生成 notes.md 更新内容"""
    
    # 找出关键数据
    batch_1 = [r for r in results if r['batch_size'] == 1][0]
    batch_4 = [r for r in results if r['batch_size'] == 4][0]
    batch_8 = [r for r in results if r['batch_size'] == 8][0]

    content = f"""
---

## Experiment: day2-batching-tradeoff
Date: {time.strftime('%Y-%m-%d')}
Hardware: {describe_environment()}
Model: {MODEL_NAME}
Total Requests: {NUM_REQUESTS}
Max New Tokens: {MAX_NEW_TOKENS}

### 实验数据

| Batch | Wait(ms) | Total(s) | Throughput | Avg Lat | P95 Lat |
|-------|----------|----------|------------|---------|---------|
"""
    for r in results:
        content += f"| {r['batch_size']} | {r['wait_window_ms']:.0f} | {r['total_time_s']:.3f} | {r['throughput_tokens_s']:.1f} t/s | {r['latency_avg_ms']:.1f} ms | {r['latency_p95_ms']:.1f} ms |\n"

    # 计算变化
    throughput_gain_4 = (batch_4['throughput_tokens_s'] / batch_1['throughput_tokens_s'] - 1) * 100
    throughput_gain_8 = (batch_8['throughput_tokens_s'] / batch_1['throughput_tokens_s'] - 1) * 100
    latency_increase_4 = (batch_4['latency_avg_ms'] / batch_1['latency_avg_ms'] - 1) * 100
    latency_increase_8 = (batch_8['latency_avg_ms'] / batch_1['latency_avg_ms'] - 1) * 100

    content += f"""
### Trade-off 分析

相对于 Batch=1:
- Batch=4: 吞吐量 {throughput_gain_4:+.1f}%, 延迟 {latency_increase_4:+.1f}%
- Batch=8: 吞吐量 {throughput_gain_8:+.1f}%, 延迟 {latency_increase_8:+.1f}%

### 结论

1. **牺牲谁换谁**: Batching 用延迟换吞吐量
   - 更大的 batch = 更高的吞吐 + 更高的延迟
   - 这是 LLM 推理服务设计的核心 trade-off

2. **业务场景选择**:
   - 实时对话: batch=1-2 (低延迟优先)
   - API 服务: batch=4 (平衡)
   - 批量处理: batch=8+ (高吞吐优先)

3. **观察**:
   - CPU 推理下 batching 效果受限于计算能力
   - GPU 推理下 batching 收益更明显（并行计算）
   - 等待窗口增加会进一步增加延迟
"""
    return content


def main():
    """主函数"""
    # 运行实验
    results = run_all_experiments()

    # 打印对比表
    print_comparison_table(results)

    # 分析 Trade-off
    analyze_tradeoff(results)

    # 生成 notes 更新
    print_header("Notes 更新建议")
    notes_content = generate_notes_update(results)
    print("以下内容可追加到 notes.md:\n")
    print(notes_content)

    print_header("实验完成")
    print("\n✅ Day 2 Batching 实验完成!")
    print("📝 请将上述内容更新到 notes.md")


if __name__ == "__main__":
    main()

