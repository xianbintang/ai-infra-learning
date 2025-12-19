"""演示模型的重要方法（除了 generate 之外）。"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME, DEVICE, DTYPE


def demo_forward_method() -> None:
    """演示 forward() / __call__() 方法。"""
    print("\n" + "=" * 60)
    print("🔍 演示 1: forward() / __call__() 方法")
    print("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    model = model.to(DEVICE)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备输入
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    print(f"\n1. 输入文本: {text}")
    print(f"   输入 Token IDs: {inputs['input_ids']}")
    
    # 方式 1: 直接调用（推荐）
    print(f"\n2. 使用 model() 直接调用（推荐）:")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            use_cache=True,
        )
    
    logits = outputs.logits
    past_key_values = outputs.past_key_values
    
    print(f"   logits shape: {logits.shape}")
    print(f"   logits 含义: [batch_size={logits.shape[0]}, seq_len={logits.shape[1]}, vocab_size={logits.shape[2]}]")
    print(f"   past_key_values: {type(past_key_values)} (KV cache)")
    
    # 方式 2: 显式调用 forward
    print(f"\n3. 使用 model.forward() 显式调用:")
    with torch.no_grad():
        outputs2 = model.forward(
            input_ids=inputs["input_ids"],
            use_cache=True,
        )
    
    print(f"   两种方式结果相同: {torch.equal(logits, outputs2.logits)}")
    
    # 获取下一个 token 的概率
    print(f"\n4. 获取下一个 token 的概率分布:")
    next_token_logits = logits[:, -1, :]  # 最后一个位置的 logits
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(next_token_probs, k=5, dim=-1)
    
    print(f"   下一个 token 的 top-5 预测:")
    for i, (prob, idx) in enumerate(zip(top_k_probs[0], top_k_indices[0])):
        token = tokenizer.decode([idx.item()])
        print(f"     {i+1}. Token ID {idx.item():5d} ({prob.item():.4f}): '{token}'")


def demo_eval_train_methods() -> None:
    """演示 eval() 和 train() 方法。"""
    print("\n" + "=" * 60)
    print("🎯 演示 2: eval() 和 train() 方法")
    print("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    model = model.to(DEVICE)
    
    print("\n1. 默认模式:")
    print(f"   训练模式: {model.training}")
    
    # 切换到评估模式
    print("\n2. 调用 model.eval():")
    model.eval()
    print(f"   训练模式: {model.training}")
    print(f"   💡 已禁用 Dropout 和 BatchNorm 更新")
    
    # 检查 Dropout 层
    dropout_modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    if dropout_modules:
        print(f"   Dropout 模块数量: {len(dropout_modules)}")
        print(f"   第一个 Dropout 的训练模式: {dropout_modules[0].training}")
    
    # 切换到训练模式
    print("\n3. 调用 model.train():")
    model.train()
    print(f"   训练模式: {model.training}")
    print(f"   💡 已启用 Dropout 和 BatchNorm 更新")
    
    if dropout_modules:
        print(f"   第一个 Dropout 的训练模式: {dropout_modules[0].training}")
    
    # 切换回评估模式（推理需要）
    model.eval()
    print("\n4. 推理前必须切换回 eval() 模式")


def demo_parameters_methods() -> None:
    """演示 parameters() 和 named_parameters() 方法。"""
    print("\n" + "=" * 60)
    print("📊 演示 3: parameters() 和 named_parameters() 方法")
    print("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    model = model.to(DEVICE)
    model.eval()
    
    # 使用 parameters()
    print("\n1. 使用 parameters() 统计参数:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   总参数数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    # 使用 named_parameters()
    print("\n2. 使用 named_parameters() 查看参数详情（前 5 个）:")
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= 5:
            break
        print(f"   {name}:")
        print(f"     - Shape: {param.shape}")
        print(f"     - 参数数量: {param.numel():,}")
        print(f"     - 需要梯度: {param.requires_grad}")
        print(f"     - 设备: {param.device}")
        print(f"     - 数据类型: {param.dtype}")
    
    # 统计各层参数
    print("\n3. 按层统计参数:")
    layer_params = {}
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0] if '.' in name else name
        if layer_type not in layer_params:
            layer_params[layer_type] = 0
        layer_params[layer_type] += param.numel()
    
    for layer, count in sorted(layer_params.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {layer}: {count:,} 参数")


def demo_state_dict_methods() -> None:
    """演示 state_dict() 和 load_state_dict() 方法。"""
    print("\n" + "=" * 60)
    print("💾 演示 4: state_dict() 和 load_state_dict() 方法")
    print("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    model = model.to(DEVICE)
    model.eval()
    
    # 获取状态字典
    print("\n1. 获取 state_dict():")
    state_dict = model.state_dict()
    print(f"   状态字典键的数量: {len(state_dict)}")
    print(f"   前 5 个键:")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"     {i+1}. {key}: {state_dict[key].shape}")
    
    # 保存状态字典（示例，不实际保存）
    print("\n2. 保存状态字典（示例）:")
    print("   torch.save(state_dict, 'model.pth')")
    print("   💡 可以保存模型的所有参数和缓冲区")
    
    # 加载状态字典（示例）
    print("\n3. 加载状态字典（示例）:")
    print("   state_dict = torch.load('model.pth')")
    print("   model.load_state_dict(state_dict)")
    print("   💡 可以恢复模型的参数值")


def demo_to_method() -> None:
    """演示 to() 方法。"""
    print("\n" + "=" * 60)
    print("🚚 演示 5: to() 方法")
    print("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    
    print("\n1. 初始设备:")
    first_param = next(model.parameters())
    print(f"   第一个参数的设备: {first_param.device}")
    
    # 迁移到设备
    print(f"\n2. 迁移到 {DEVICE}:")
    model = model.to(DEVICE)
    first_param = next(model.parameters())
    print(f"   第一个参数的设备: {first_param.device}")
    print(f"   数据类型: {first_param.dtype}")
    
    # 验证所有参数都在同一设备
    devices = set(str(p.device) for p in model.parameters())
    print(f"\n3. 所有参数的设备: {devices}")
    print(f"   💡 所有参数都在同一设备: {len(devices) == 1}")


def demo_modules_methods() -> None:
    """演示 modules() 和 named_modules() 方法。"""
    print("\n" + "=" * 60)
    print("🧩 演示 6: modules() 和 named_modules() 方法")
    print("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    model = model.to(DEVICE)
    model.eval()
    
    # 使用 modules()
    print("\n1. 使用 modules() 统计模块类型:")
    module_types = {}
    for module in model.modules():
        module_type = type(module).__name__
        module_types[module_type] = module_types.get(module_type, 0) + 1
    
    print(f"   模块类型统计（前 5 个）:")
    for module_type, count in sorted(module_types.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     {module_type}: {count} 个")
    
    # 使用 named_modules()
    print("\n2. 使用 named_modules() 查看模块结构（前 5 个）:")
    for i, (name, module) in enumerate(model.named_modules()):
        if i >= 5:
            break
        print(f"   {name}: {type(module).__name__}")


def demo_forward_vs_generate() -> None:
    """对比 forward() 和 generate() 的区别。"""
    print("\n" + "=" * 60)
    print("⚖️  演示 7: forward() vs generate() 对比")
    print("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    model = model.to(DEVICE)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text = "Hello"
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    print(f"\n输入: {text}")
    
    # 使用 forward()
    print("\n1. 使用 forward() (单次前向传播):")
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], use_cache=True)
    
    logits = outputs.logits
    next_token_logits = logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)
    
    print(f"   返回: logits shape {logits.shape}")
    print(f"   下一个 token ID: {next_token_id.item()}")
    print(f"   下一个 token: '{tokenizer.decode([next_token_id.item()])}'")
    print(f"   💡 只执行一次前向传播，需要手动实现生成循环")
    
    # 使用 generate()
    print("\n2. 使用 generate() (自动生成):")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"   返回: 完整 token 序列 shape {generated.shape}")
    print(f"   生成文本: '{generated_text}'")
    print(f"   💡 自动完成整个生成过程（prefill + decode loop）")
    
    print("\n3. 对比:")
    print("   forward():")
    print("     - 单次前向传播")
    print("     - 返回 logits")
    print("     - 需要手动实现生成循环")
    print("     - 更精细的控制")
    print("   generate():")
    print("     - 自动完成生成")
    print("     - 返回完整序列")
    print("     - 内部已优化")
    print("     - 更快速开发")


def demo_manual_generation_loop() -> None:
    """演示使用 forward() 实现手动生成循环。"""
    print("\n" + "=" * 60)
    print("🔄 演示 8: 使用 forward() 实现手动生成循环")
    print("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
    model = model.to(DEVICE)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text = "Hello"
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    print(f"\n输入: {text}")
    print(f"开始手动生成循环...\n")
    
    # Prefill 阶段
    print("1. Prefill 阶段（处理 prompt）:")
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], use_cache=True)
    
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    print(f"   获得 KV cache 和下一个 token 的 logits")
    
    # Decode 循环
    print("\n2. Decode 循环（生成新 tokens）:")
    generated_ids = []
    max_new_tokens = 5
    
    for step in range(max_new_tokens):
        # 选择下一个 token
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids.append(next_token_id)
        
        token = tokenizer.decode([next_token_id.item()])
        print(f"   步骤 {step+1}: 选择 token ID {next_token_id.item()} ('{token}')")
        
        # 如果遇到 EOS，停止
        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"   遇到 EOS token，停止生成")
            break
        
        # 下一步前向传播（使用 KV cache）
        with torch.no_grad():
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
    
    # 解码结果
    if generated_ids:
        all_generated = torch.cat(generated_ids, dim=1)
        full_sequence = torch.cat([inputs["input_ids"], all_generated], dim=1)
        generated_text = tokenizer.decode(full_sequence[0], skip_special_tokens=True)
        
        print(f"\n3. 生成结果:")
        print(f"   生成的 token IDs: {[id.item() for id in generated_ids]}")
        print(f"   完整文本: '{generated_text}'")
        print(f"   💡 使用 forward() + KV cache 实现高效生成")


def main() -> None:
    """主函数。"""
    print("=" * 60)
    print("🔬 模型重要方法演示（除了 generate 之外）")
    print("=" * 60)
    print(f"\n模型: {MODEL_NAME}")
    print(f"设备: {DEVICE}")
    
    # 演示 1: forward 方法
    demo_forward_method()
    
    # 演示 2: eval/train 方法
    demo_eval_train_methods()
    
    # 演示 3: parameters 方法
    demo_parameters_methods()
    
    # 演示 4: state_dict 方法
    demo_state_dict_methods()
    
    # 演示 5: to 方法
    demo_to_method()
    
    # 演示 6: modules 方法
    demo_modules_methods()
    
    # 演示 7: forward vs generate
    demo_forward_vs_generate()
    
    # 演示 8: 手动生成循环
    demo_manual_generation_loop()
    
    print("\n" + "=" * 60)
    print("✅ 演示完成")
    print("=" * 60)
    print("\n💡 关键要点:")
    print("   1. forward() 用于单次前向传播，需要手动实现生成循环")
    print("   2. generate() 自动完成整个生成过程")
    print("   3. eval() 推理前必须调用，禁用训练时的随机性")
    print("   4. to(device) 用于设备迁移")
    print("   5. parameters() 和 named_parameters() 用于参数访问")
    print("   6. state_dict() 用于模型保存和加载")
    print("   7. 理解这些方法有助于精细控制模型行为")


if __name__ == "__main__":
    main()


