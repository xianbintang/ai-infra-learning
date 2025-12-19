"""详细观察 tokenizer 和 model 加载过程的脚本。"""
from __future__ import annotations

import time
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_NAME, DEVICE, DTYPE, describe_environment


def get_cache_dir() -> str:
    """获取 Hugging Face 缓存目录。"""
    cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(cache_home, "hub")


def format_size(size_bytes: int) -> str:
    """格式化文件大小。"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_model_cache_size(model_name: str) -> int:
    """计算模型缓存目录的总大小。"""
    cache_dir = get_cache_dir()
    model_dir = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    
    if not os.path.exists(model_dir):
        return 0
    
    total_size = 0
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                pass
    return total_size


def observe_tokenizer_loading(model_name: str) -> None:
    """观察 tokenizer 加载过程。"""
    print("\n" + "=" * 60)
    print("🔤 TOKENIZER 加载过程观察")
    print("=" * 60)
    
    # 检查缓存
    cache_dir = get_cache_dir()
    print(f"\n📁 缓存目录: {cache_dir}")
    
    cache_size_before = get_model_cache_size(model_name)
    if cache_size_before > 0:
        print(f"📦 缓存大小（加载前）: {format_size(cache_size_before)}")
        print("✅ Tokenizer 文件已缓存，将直接从缓存加载")
    else:
        print("❌ Tokenizer 文件未缓存，需要从网络下载")
    
    # 加载 tokenizer
    print(f"\n⏳ 开始加载 tokenizer: {model_name}")
    start_time = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    load_time = time.perf_counter() - start_time
    
    print(f"✅ Tokenizer 加载完成，耗时: {load_time:.3f} 秒")
    
    # 检查缓存变化
    cache_size_after = get_model_cache_size(model_name)
    if cache_size_after > cache_size_before:
        downloaded = cache_size_after - cache_size_before
        print(f"📥 下载大小: {format_size(downloaded)}")
    
    # 显示 tokenizer 信息
    print(f"\n📊 Tokenizer 信息:")
    print(f"  - 类型: {type(tokenizer).__name__}")
    print(f"  - 词汇表大小: {tokenizer.vocab_size:,}")
    print(f"  - 最大长度: {tokenizer.model_max_length}")
    print(f"  - BOS token: {tokenizer.bos_token}")
    print(f"  - EOS token: {tokenizer.eos_token}")
    print(f"  - PAD token: {tokenizer.pad_token}")
    
    # 测试编码/解码
    test_text = "Hello, world!"
    print(f"\n🧪 测试编码/解码:")
    print(f"  原文: {test_text}")
    encoded = tokenizer.encode(test_text)
    print(f"  编码: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"  解码: {decoded}")
    
    return tokenizer


def observe_model_loading(model_name: str, dtype: torch.dtype) -> None:
    """观察 model 加载过程。"""
    print("\n" + "=" * 60)
    print("🤖 MODEL 加载过程观察")
    print("=" * 60)
    
    # 检查缓存
    cache_size_before = get_model_cache_size(model_name)
    if cache_size_before > 0:
        print(f"📦 缓存大小（加载前）: {format_size(cache_size_before)}")
    
    # 加载模型
    print(f"\n⏳ 开始加载 model: {model_name}")
    print(f"  - 设备: {DEVICE}")
    print(f"  - 数据类型: {dtype}")
    
    # 阶段 1: 下载和读取配置
    print("\n📋 阶段 1: 读取配置和构建架构...")
    start_time = time.perf_counter()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        torch_dtype=dtype,
    )
    
    config_time = time.perf_counter() - start_time
    print(f"✅ 配置读取和架构构建完成，耗时: {config_time:.3f} 秒")
    
    # 显示模型信息
    print(f"\n📊 Model 信息:")
    print(f"  - 类型: {type(model).__name__}")
    print(f"  - 设备: {next(model.parameters()).device}")
    print(f"  - 数据类型: {next(model.parameters()).dtype}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数数: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    
    # 估算显存占用
    if DEVICE == "cuda":
        param_size_mb = total_params * 2 / (1024 ** 2) if dtype == torch.float16 else total_params * 4 / (1024 ** 2)
        print(f"  - 估算显存占用: {param_size_mb:.2f} MB")
    
    # 阶段 2: 设备迁移
    print(f"\n🚚 阶段 2: 迁移到设备 {DEVICE}...")
    start_time = time.perf_counter()
    
    model = model.to(DEVICE)
    
    migrate_time = time.perf_counter() - start_time
    print(f"✅ 设备迁移完成，耗时: {migrate_time:.3f} 秒")
    print(f"  - 当前设备: {next(model.parameters()).device}")
    
    # 阶段 3: 切换到评估模式
    print(f"\n🎯 阶段 3: 切换到评估模式...")
    start_time = time.perf_counter()
    
    model.eval()
    
    eval_time = time.perf_counter() - start_time
    print(f"✅ 评估模式设置完成，耗时: {eval_time:.6f} 秒")
    
    # 检查缓存变化
    cache_size_after = get_model_cache_size(model_name)
    if cache_size_after > cache_size_before:
        downloaded = cache_size_after - cache_size_before
        print(f"\n📥 下载大小: {format_size(downloaded)}")
    
    return model


def main() -> None:
    """主函数。"""
    print("=" * 60)
    print("🔍 Tokenizer 和 Model 加载过程详细观察")
    print("=" * 60)
    print(f"\n环境信息: {describe_environment()}")
    print(f"模型名称: {MODEL_NAME}")
    
    # 观察 tokenizer 加载
    tokenizer = observe_tokenizer_loading(MODEL_NAME)
    
    # 观察 model 加载
    model = observe_model_loading(MODEL_NAME, DTYPE)
    
    # 总结
    print("\n" + "=" * 60)
    print("📝 总结")
    print("=" * 60)
    print("\n✅ Tokenizer 和 Model 都已成功加载！")
    print("\n💡 提示:")
    print("  - 首次运行需要从网络下载文件，耗时较长")
    print("  - 后续运行会直接从缓存加载，速度更快")
    print("  - 缓存位置:", get_cache_dir())
    print("  - 可以通过设置环境变量 HF_HOME 更改缓存位置")


if __name__ == "__main__":
    main()

