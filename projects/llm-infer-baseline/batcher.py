"""
Day 2: 动态 Batching 实现

核心功能:
1. 请求队列管理
2. 等待窗口机制
3. Padding 处理
4. 批量推理执行
"""

import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class InferenceRequest:
    """单个推理请求"""
    request_id: int
    prompt: str
    max_new_tokens: int = 32
    # 时间戳记录
    submit_time: float = field(default_factory=time.perf_counter)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    # 结果
    generated_text: Optional[str] = None
    generated_ids: Optional[List[int]] = None

    @property
    def queue_time(self) -> float:
        """排队等待时间"""
        if self.start_time is None:
            return 0.0
        return self.start_time - self.submit_time

    @property
    def inference_time(self) -> float:
        """推理执行时间"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def total_latency(self) -> float:
        """总延迟 = 排队 + 推理"""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.submit_time


@dataclass
class BatchResult:
    """批量推理结果"""
    batch_size: int
    requests: List[InferenceRequest]
    total_time: float
    tokens_generated: int

    @property
    def throughput_tokens_per_sec(self) -> float:
        """吞吐量 (tokens/s)"""
        if self.total_time == 0:
            return 0.0
        return self.tokens_generated / self.total_time

    @property
    def throughput_requests_per_sec(self) -> float:
        """吞吐量 (requests/s)"""
        if self.total_time == 0:
            return 0.0
        return self.batch_size / self.total_time

    @property
    def avg_latency(self) -> float:
        """平均延迟"""
        if not self.requests:
            return 0.0
        return sum(r.total_latency for r in self.requests) / len(self.requests)

    @property
    def avg_queue_time(self) -> float:
        """平均排队时间"""
        if not self.requests:
            return 0.0
        return sum(r.queue_time for r in self.requests) / len(self.requests)


class DynamicBatcher:
    """
    动态 Batcher
    
    工作原理:
    1. 请求进入队列
    2. 等待 wait_window_ms 时间
    3. 收集队列中的请求（最多 max_batch_size 个）
    4. 批量执行推理
    5. 返回结果
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu",
        max_batch_size: int = 8,
        wait_window_ms: float = 10.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.wait_window_ms = wait_window_ms

        # 确保 tokenizer 有 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 请求计数器
        self._request_counter = 0
        self._lock = threading.Lock()

    def _get_request_id(self) -> int:
        """生成唯一请求 ID"""
        with self._lock:
            self._request_counter += 1
            return self._request_counter

    def create_request(self, prompt: str, max_new_tokens: int = 32) -> InferenceRequest:
        """创建推理请求"""
        return InferenceRequest(
            request_id=self._get_request_id(),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    def _pad_and_batch(self, requests: List[InferenceRequest]) -> Dict[str, torch.Tensor]:
        """
        将多个请求 padding 成 batch
        
        返回:
            input_ids: [batch_size, max_seq_len]
            attention_mask: [batch_size, max_seq_len]
        """
        prompts = [r.prompt for r in requests]

        # 使用 tokenizer 的 batch 编码功能
        # padding=True 会自动 pad 到最长序列
        # return_tensors="pt" 返回 PyTorch tensor
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # 移动到目标设备
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _batch_generate(
        self,
        requests: List[InferenceRequest],
        max_new_tokens: int = 32,
    ) -> List[InferenceRequest]:
        """
        批量生成
        
        Args:
            requests: 请求列表
            max_new_tokens: 生成的最大 token 数
            
        Returns:
            更新后的请求列表（包含生成结果）
        """
        if not requests:
            return requests

        # 记录开始时间
        start_time = time.perf_counter()
        for r in requests:
            r.start_time = start_time

        # Padding 和批量编码
        batch_inputs = self._pad_and_batch(requests)
        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs["attention_mask"]

        # 批量生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy decoding
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 记录结束时间
        end_time = time.perf_counter()

        # 解码并更新请求
        for i, request in enumerate(requests):
            request.end_time = end_time

            # 获取生成的部分（去掉 prompt）
            prompt_len = (attention_mask[i] == 1).sum().item()
            generated_ids = outputs[i][prompt_len:].tolist()

            request.generated_ids = generated_ids
            request.generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

        return requests

    def process_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 32,
        wait_for_batch: bool = False,
    ) -> BatchResult:
        """
        处理一批请求
        
        Args:
            prompts: prompt 列表
            max_new_tokens: 生成的最大 token 数
            wait_for_batch: 是否等待凑齐 batch（模拟动态 batching）
            
        Returns:
            BatchResult 包含所有请求的结果和统计
        """
        # 创建请求
        requests = [self.create_request(p, max_new_tokens) for p in prompts]

        # 如果模拟动态 batching，添加等待时间
        if wait_for_batch and self.wait_window_ms > 0:
            time.sleep(self.wait_window_ms / 1000)

        # 批量处理
        batch_start = time.perf_counter()
        requests = self._batch_generate(requests, max_new_tokens)
        batch_time = time.perf_counter() - batch_start

        # 计算总 token 数
        total_tokens = sum(
            len(r.generated_ids) for r in requests if r.generated_ids
        )

        return BatchResult(
            batch_size=len(requests),
            requests=requests,
            total_time=batch_time,
            tokens_generated=total_tokens,
        )

    def process_single(self, prompt: str, max_new_tokens: int = 32) -> InferenceRequest:
        """处理单个请求（batch_size=1）"""
        result = self.process_batch([prompt], max_new_tokens)
        return result.requests[0]


def run_batch_experiment(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int = 32,
    wait_window_ms: float = 0.0,
) -> Dict[str, Any]:
    """
    运行单次 batch 实验
    
    Args:
        model: 模型
        tokenizer: tokenizer
        device: 设备
        prompts: prompt 列表
        batch_size: batch 大小
        max_new_tokens: 生成的最大 token 数
        wait_window_ms: 等待窗口（毫秒）
        
    Returns:
        实验结果字典
    """
    batcher = DynamicBatcher(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_batch_size=batch_size,
        wait_window_ms=wait_window_ms,
    )

    # 分批处理
    all_results = []
    total_start = time.perf_counter()

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        result = batcher.process_batch(
            batch_prompts,
            max_new_tokens=max_new_tokens,
            wait_for_batch=True,
        )
        all_results.append(result)

    total_time = time.perf_counter() - total_start

    # 汇总统计
    all_requests = []
    total_tokens = 0
    for r in all_results:
        all_requests.extend(r.requests)
        total_tokens += r.tokens_generated

    latencies = [r.total_latency for r in all_requests]
    queue_times = [r.queue_time for r in all_requests]
    inference_times = [r.inference_time for r in all_requests]

    return {
        "batch_size": batch_size,
        "wait_window_ms": wait_window_ms,
        "num_requests": len(prompts),
        "num_batches": len(all_results),
        "total_time_s": total_time,
        "total_tokens": total_tokens,
        "throughput_tokens_s": total_tokens / total_time if total_time > 0 else 0,
        "throughput_requests_s": len(prompts) / total_time if total_time > 0 else 0,
        "latency_avg_ms": sum(latencies) / len(latencies) * 1000 if latencies else 0,
        "latency_min_ms": min(latencies) * 1000 if latencies else 0,
        "latency_max_ms": max(latencies) * 1000 if latencies else 0,
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000 if latencies else 0,
        "queue_time_avg_ms": sum(queue_times) / len(queue_times) * 1000 if queue_times else 0,
        "inference_time_avg_ms": sum(inference_times) / len(inference_times) * 1000 if inference_times else 0,
        "requests": all_requests,
    }


# ============================================================
# 演示和测试
# ============================================================

if __name__ == "__main__":
    from config import MODEL_NAME, DEVICE, describe_environment
    from model import load_model
    from tokenizer import load_tokenizer

    print("=" * 60)
    print("Batcher 功能测试")
    print("=" * 60)

    print("\n[1] 加载模型...")
    model = load_model(MODEL_NAME)
    tokenizer = load_tokenizer(MODEL_NAME)
    print(f"环境: {describe_environment()}")

    print("\n[2] 创建 Batcher...")
    batcher = DynamicBatcher(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        max_batch_size=4,
        wait_window_ms=10,
    )

    print("\n[3] 测试单请求...")
    single_result = batcher.process_single(
        "Q: What is machine learning?\nA:",
        max_new_tokens=20,
    )
    print(f"  Request ID: {single_result.request_id}")
    print(f"  Prompt: {single_result.prompt!r}")
    print(f"  Generated: {single_result.generated_text!r}")
    print(f"  Latency: {single_result.total_latency*1000:.2f} ms")

    print("\n[4] 测试批量请求 (batch=4)...")
    test_prompts = [
        "Q: What is AI?\nA:",
        "Q: What is ML?\nA:",
        "Q: What is DL?\nA:",
        "Q: What is NLP?\nA:",
    ]
    batch_result = batcher.process_batch(test_prompts, max_new_tokens=20)

    print(f"  Batch size: {batch_result.batch_size}")
    print(f"  Total time: {batch_result.total_time*1000:.2f} ms")
    print(f"  Tokens generated: {batch_result.tokens_generated}")
    print(f"  Throughput: {batch_result.throughput_tokens_per_sec:.2f} tokens/s")
    print(f"  Avg latency: {batch_result.avg_latency*1000:.2f} ms")

    print("\n  各请求结果:")
    for r in batch_result.requests:
        print(f"    [{r.request_id}] {r.generated_text!r} ({r.inference_time*1000:.2f} ms)")

    print("\n✅ Batcher 测试完成!")

