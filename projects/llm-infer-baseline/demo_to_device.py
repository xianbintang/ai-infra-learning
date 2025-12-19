"""演示 model.to(DEVICE) 的作用和影响。"""
from __future__ import annotations

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME, DEVICE, DTYPE


def demo_device_mismatch_error() -> None:
    """演示设备不匹配时的错误。"""
    print("\n" + "=" * 60)
    print("❌ 演示 1: 设备不匹配错误")
    print("=" * 60)
    
    print("\n场景：模型在 CPU，输入在 GPU")
    
    # 加载模型但不迁移到 GPU
    print(f"\n1. 加载模型到 CPU...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    print(f"   模型设备: {next(model.parameters()).device}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 输入在 GPU
    print(f"\n2. 将输入放到 GPU...")
    inputs = tokenizer("Hello", return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        print(f"   输入设备: {inputs['input_ids'].device}")
    
    # 尝试推理
    print(f"\n3. 尝试推理（会报错）...")
    try:
        model.eval()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5)
        print("   ✅ 成功（不应该发生）")
    except RuntimeError as e:
        print(f"   ❌ 错误: {str(e)[:100]}...")
        print("\n💡 原因：模型参数在 CPU，输入在 GPU，无法计算")


def demo_correct_usage() -> None:
    """演示正确的用法。"""
    print("\n" + "=" * 60)
    print("✅ 演示 2: 正确的用法")
    print("=" * 60)
    
    print(f"\n场景：模型和输入都在 {DEVICE}")
    
    # 加载模型并迁移
    print(f"\n1. 加载模型并迁移到 {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    model = model.to(DEVICE)  # ← 关键步骤
    model.eval()
    print(f"   模型设备: {next(model.parameters()).device}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 输入也在同一设备
    print(f"\n2. 将输入放到 {DEVICE}...")
    inputs = tokenizer("Hello", return_tensors="pt").to(DEVICE)
    print(f"   输入设备: {inputs['input_ids'].device}")
    
    # 推理
    print(f"\n3. 执行推理...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   ✅ 成功！生成文本: {generated_text}")


def demo_performance_comparison() -> None:
    """演示 CPU vs GPU 性能对比。"""
    print("\n" + "=" * 60)
    print("⚡ 演示 3: CPU vs GPU 性能对比")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt")
    max_new_tokens = 20
    
    # CPU 推理
    print("\n📊 CPU 推理测试...")
    model_cpu = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model_cpu.eval()
    
    start = time.perf_counter()
    with torch.no_grad():
        outputs_cpu = model_cpu.generate(**inputs, max_new_tokens=max_new_tokens)
    cpu_time = time.perf_counter() - start
    
    cpu_text = tokenizer.decode(outputs_cpu[0], skip_special_tokens=True)
    print(f"   耗时: {cpu_time:.3f} 秒")
    print(f"   生成: {cpu_text[:50]}...")
    
    # GPU 推理（如果可用）
    if torch.cuda.is_available():
        print("\n📊 GPU 推理测试...")
        model_gpu = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
        model_gpu = model_gpu.to("cuda")  # ← 关键步骤
        model_gpu.eval()
        
        inputs_gpu = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 预热
        with torch.no_grad():
            _ = model_gpu.generate(**inputs_gpu, max_new_tokens=max_new_tokens)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs_gpu = model_gpu.generate(**inputs_gpu, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start
        
        gpu_text = tokenizer.decode(outputs_gpu[0], skip_special_tokens=True)
        print(f"   耗时: {gpu_time:.3f} 秒")
        print(f"   生成: {gpu_text[:50]}...")
        
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"\n🚀 GPU 加速比: {speedup:.1f}x")
    else:
        print("\n⚠️  未检测到 GPU，跳过 GPU 测试")


def demo_memory_usage() -> None:
    """演示设备迁移对内存/显存的影响。"""
    print("\n" + "=" * 60)
    print("💾 演示 4: 内存/显存使用")
    print("=" * 60)
    
    # 计算模型大小
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n模型参数数量: {total_params:,}")
    
    if DTYPE == torch.float32:
        size_mb = total_params * 4 / (1024 ** 2)
        print(f"float32 大小: {size_mb:.2f} MB")
    elif DTYPE == torch.float16:
        size_mb = total_params * 2 / (1024 ** 2)
        print(f"float16 大小: {size_mb:.2f} MB")
    
    # CPU 内存
    print(f"\n📌 CPU 内存占用:")
    print(f"   模型加载后: ~{size_mb:.2f} MB (系统内存)")
    
    # GPU 显存（如果使用 GPU）
    if DEVICE == "cuda" and torch.cuda.is_available():
        print(f"\n📌 GPU 显存占用:")
        print(f"   迁移前: 0 MB")
        
        model = model.to("cuda")
        
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print(f"   迁移后:")
        print(f"     - 已分配: {allocated:.2f} MB")
        print(f"     - 已保留: {reserved:.2f} MB")
        print(f"   💡 显存占用略大于模型大小（包含 PyTorch 开销）")


def demo_what_happens_inside() -> None:
    """演示 to(DEVICE) 内部发生了什么。"""
    print("\n" + "=" * 60)
    print("🔍 演示 5: to(DEVICE) 内部过程")
    print("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    
    print("\n1. 迁移前的设备状态:")
    devices_before = set()
    for name, param in model.named_parameters():
        devices_before.add(str(param.device))
        if len(devices_before) == 1:
            print(f"   参数 '{name}' 在设备: {param.device}")
            break
    
    print(f"   所有参数都在: {list(devices_before)[0]}")
    
    if DEVICE == "cuda" and torch.cuda.is_available():
        print(f"\n2. 执行 model.to('{DEVICE}')...")
        start = time.perf_counter()
        model = model.to(DEVICE)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"   耗时: {elapsed:.3f} 秒")
        
        print(f"\n3. 迁移后的设备状态:")
        devices_after = set()
        for name, param in model.named_parameters():
            devices_after.add(str(param.device))
            if len(devices_after) == 1:
                print(f"   参数 '{name}' 在设备: {param.device}")
                break
        
        print(f"   所有参数都在: {list(devices_after)[0]}")
        
        print(f"\n💡 发生了什么:")
        print(f"   - 遍历了所有 {sum(1 for _ in model.parameters())} 个参数")
        print(f"   - 通过 PCIe 总线复制了数据")
        print(f"   - 在 GPU 显存中分配了新空间")
        print(f"   - 更新了所有参数的设备指针")


def main() -> None:
    """主函数。"""
    print("=" * 60)
    print("🔬 model.to(DEVICE) 详细演示")
    print("=" * 60)
    print(f"\n环境: {DEVICE}, dtype: {DTYPE}")
    
    # 演示 1: 设备不匹配错误
    if torch.cuda.is_available():
        demo_device_mismatch_error()
    
    # 演示 2: 正确用法
    demo_correct_usage()
    
    # 演示 3: 性能对比
    demo_performance_comparison()
    
    # 演示 4: 内存使用
    demo_memory_usage()
    
    # 演示 5: 内部过程
    if torch.cuda.is_available():
        demo_what_happens_inside()
    
    print("\n" + "=" * 60)
    print("✅ 演示完成")
    print("=" * 60)
    print("\n💡 关键要点:")
    print("   1. model.to(DEVICE) 将模型参数迁移到目标设备")
    print("   2. 模型和输入必须在同一设备，否则会报错")
    print("   3. GPU 推理比 CPU 快 10-100 倍")
    print("   4. 迁移过程会复制数据，需要时间（0.5-2 秒）")


if __name__ == "__main__":
    main()

