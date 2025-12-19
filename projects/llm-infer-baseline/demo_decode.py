"""演示 tokenizer.decode() 的过程和逻辑。"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from config import MODEL_NAME


def demo_basic_decode() -> None:
    """演示基本的 decode 过程。"""
    print("\n" + "=" * 60)
    print("📝 演示 1: 基本 decode 过程")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 编码
    text = "Hello world!"
    print(f"\n1. 原始文本: {text}")
    
    token_ids = tokenizer.encode(text)
    print(f"2. 编码后的 Token IDs: {token_ids}")
    
    # 显示每个 token 的详细信息
    print(f"\n3. Token 详细信息:")
    for i, token_id in enumerate(token_ids):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        print(f"   [{i}] ID: {token_id:5d} → Token: '{token}'")
    
    # 解码
    decoded = tokenizer.decode(token_ids)
    print(f"\n4. 解码后的文本: '{decoded}'")
    print(f"5. 是否完全匹配: {text == decoded}")


def demo_decode_with_special_tokens() -> None:
    """演示特殊 token 的处理。"""
    print("\n" + "=" * 60)
    print("🔖 演示 2: 特殊 Token 处理")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 包含特殊 token 的序列
    text = "Hello world!"
    token_ids = tokenizer.encode(text)
    
    # 添加 EOS token
    eos_id = tokenizer.eos_token_id
    token_ids_with_eos = token_ids + [eos_id]
    
    print(f"\n1. 原始 Token IDs: {token_ids}")
    print(f"2. 添加 EOS token: {token_ids_with_eos}")
    print(f"   EOS token ID: {eos_id}")
    print(f"   EOS token 字符串: '{tokenizer.eos_token}'")
    
    # 不跳过特殊 token
    decoded_with_special = tokenizer.decode(
        token_ids_with_eos,
        skip_special_tokens=False
    )
    print(f"\n3. 解码（保留特殊 token）: '{decoded_with_special}'")
    
    # 跳过特殊 token
    decoded_without_special = tokenizer.decode(
        token_ids_with_eos,
        skip_special_tokens=True
    )
    print(f"4. 解码（跳过特殊 token）: '{decoded_without_special}'")
    
    print(f"\n💡 区别: 是否包含 '{tokenizer.eos_token}' 标记")


def demo_decode_process() -> None:
    """演示 decode 的详细过程。"""
    print("\n" + "=" * 60)
    print("🔍 演示 3: decode 详细过程")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 示例 token IDs
    token_ids = [15496, 995, 33]  # "Hello world!"
    
    print(f"\n输入 Token IDs: {token_ids}")
    
    # 步骤 1: 查找词汇表
    print(f"\n步骤 1: 查找词汇表（ID → Token 字符串）")
    tokens = []
    for token_id in token_ids:
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        tokens.append(token)
        print(f"   ID {token_id:5d} → '{token}'")
    
    # 步骤 2: 显示 BPE 处理
    print(f"\n步骤 2: BPE Token 分析")
    for i, token in enumerate(tokens):
        if token.startswith("Ġ"):
            print(f"   [{i}] '{token}' → 前面有空格（Ġ 表示空格）")
            print(f"       处理: 添加空格 + '{token[1:]}'")
        else:
            print(f"   [{i}] '{token}' → 直接添加")
    
    # 步骤 3: 合并结果
    print(f"\n步骤 3: 合并 BPE tokens")
    text = tokenizer.decode(token_ids)
    print(f"   结果: '{text}'")
    
    # 步骤 4: 验证
    print(f"\n步骤 4: 验证")
    print(f"   原始 Token IDs: {token_ids}")
    print(f"   解码后文本: '{text}'")
    print(f"   可以重新编码: {tokenizer.encode(text) == token_ids}")


def demo_decode_from_model_output() -> None:
    """演示如何处理模型输出。"""
    print("\n" + "=" * 60)
    print("🤖 演示 4: 处理模型输出")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 模拟模型输出（包含 prompt + 生成部分）
    prompt = "Hello"
    prompt_ids = tokenizer.encode(prompt)
    
    # 模拟生成的部分
    generated_ids = [995, 33]  # " world!"
    
    # 完整输出
    full_output = prompt_ids + generated_ids
    
    print(f"\n1. Prompt: '{prompt}'")
    print(f"   Prompt Token IDs: {prompt_ids}")
    print(f"\n2. 生成部分 Token IDs: {generated_ids}")
    print(f"\n3. 完整输出 Token IDs: {full_output}")
    
    # 解码完整输出
    full_text = tokenizer.decode(full_output, skip_special_tokens=True)
    print(f"\n4. 完整文本: '{full_text}'")
    
    # 只解码生成部分
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"5. 仅生成部分: '{generated_text}'")
    
    # 对比
    print(f"\n💡 区别:")
    print(f"   完整文本包含 prompt: '{full_text}'")
    print(f"   仅生成部分: '{generated_text}'")


def demo_decode_performance() -> None:
    """演示 decode 的性能。"""
    print("\n" + "=" * 60)
    print("⚡ 演示 5: decode 性能")
    print("=" * 60)
    
    import time
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 生成不同长度的 token 序列
    test_cases = [
        ("短序列", [15496, 995, 33]),
        ("中等序列", list(range(100))),
        ("长序列", list(range(1000))),
    ]
    
    print(f"\n性能测试:")
    for name, token_ids in test_cases:
        # 单个 decode
        start = time.perf_counter()
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        single_time = time.perf_counter() - start
        
        # 批量 decode（模拟）
        start = time.perf_counter()
        for _ in range(100):
            _ = tokenizer.decode(token_ids, skip_special_tokens=True)
        batch_time = time.perf_counter() - start
        
        print(f"\n{name} ({len(token_ids)} tokens):")
        print(f"  单次 decode: {single_time*1000:.3f} ms")
        print(f"  100次 decode: {batch_time*1000:.3f} ms")
        print(f"  平均每次: {batch_time/100*1000:.3f} ms")


def demo_decode_edge_cases() -> None:
    """演示 decode 的边界情况。"""
    print("\n" + "=" * 60)
    print("⚠️  演示 6: 边界情况")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 情况 1: 空序列
    print("\n1. 空序列:")
    empty_ids = []
    empty_text = tokenizer.decode(empty_ids)
    print(f"   Token IDs: {empty_ids}")
    print(f"   解码结果: '{empty_text}'")
    
    # 情况 2: 只有特殊 token
    print("\n2. 只有特殊 token:")
    eos_only = [tokenizer.eos_token_id]
    eos_text_with = tokenizer.decode(eos_only, skip_special_tokens=False)
    eos_text_without = tokenizer.decode(eos_only, skip_special_tokens=True)
    print(f"   Token IDs: {eos_only}")
    print(f"   保留特殊 token: '{eos_text_with}'")
    print(f"   跳过特殊 token: '{eos_text_without}'")
    
    # 情况 3: Tensor 输入
    print("\n3. Tensor 输入:")
    tensor_ids = torch.tensor([15496, 995, 33])
    tensor_text = tokenizer.decode(tensor_ids)
    print(f"   输入类型: {type(tensor_ids)}")
    print(f"   解码结果: '{tensor_text}'")
    
    # 情况 4: 无效 token ID
    print("\n4. 无效 token ID:")
    invalid_ids = [999999]  # 不存在的 ID
    try:
        invalid_text = tokenizer.decode(invalid_ids)
        print(f"   Token IDs: {invalid_ids}")
        print(f"   解码结果: '{invalid_text}'")
        print(f"   💡 tokenizer 会使用 unk_token 或跳过")
    except Exception as e:
        print(f"   错误: {e}")


def demo_encode_decode_roundtrip() -> None:
    """演示编码-解码的往返过程。"""
    print("\n" + "=" * 60)
    print("🔄 演示 7: 编码-解码往返")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a programming language.",
        "   Multiple   spaces   here   ",
    ]
    
    print(f"\n测试文本的编码-解码往返:")
    for text in test_texts:
        # 编码
        token_ids = tokenizer.encode(text)
        
        # 解码
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # 比较
        match = text.strip() == decoded.strip()
        
        print(f"\n原文: '{text}'")
        print(f"  Token IDs: {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
        print(f"  解码: '{decoded}'")
        print(f"  匹配: {'✅' if match else '❌'}")
        if not match:
            print(f"  差异: 原始长度={len(text)}, 解码长度={len(decoded)}")


def main() -> None:
    """主函数。"""
    print("=" * 60)
    print("🔬 tokenizer.decode() 详细演示")
    print("=" * 60)
    print(f"\n模型: {MODEL_NAME}")
    
    # 演示 1: 基本 decode
    demo_basic_decode()
    
    # 演示 2: 特殊 token 处理
    demo_decode_with_special_tokens()
    
    # 演示 3: 详细过程
    demo_decode_process()
    
    # 演示 4: 模型输出处理
    demo_decode_from_model_output()
    
    # 演示 5: 性能
    demo_decode_performance()
    
    # 演示 6: 边界情况
    demo_decode_edge_cases()
    
    # 演示 7: 往返测试
    demo_encode_decode_roundtrip()
    
    print("\n" + "=" * 60)
    print("✅ 演示完成")
    print("=" * 60)
    print("\n💡 关键要点:")
    print("   1. decode 将 Token IDs 转换为文本")
    print("   2. 需要查找词汇表和合并 BPE tokens")
    print("   3. 可以跳过特殊 token 以得到干净输出")
    print("   4. 性能很好，通常 < 1ms")
    print("   5. 编码-解码不完全可逆（可能丢失空格等）")


if __name__ == "__main__":
    main()


