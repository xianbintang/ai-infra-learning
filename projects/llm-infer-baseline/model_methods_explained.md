# 模型重要方法详解：除了 generate 之外

## 目录
1. [快速概览](#快速概览)
2. [核心推理方法](#核心推理方法)
3. [模式管理方法](#模式管理方法)
4. [设备管理方法](#设备管理方法)
5. [参数访问方法](#参数访问方法)
6. [状态保存和加载](#状态保存和加载)
7. [其他重要方法](#其他重要方法)
8. [方法对比和使用场景](#方法对比和使用场景)

---

## 快速概览

除了 `generate()` 方法，PyTorch/Hugging Face 模型还提供了许多重要方法：

| 类别 | 方法 | 用途 | 重要性 |
|------|------|------|--------|
| **推理** | `forward()` / `__call__()` | 单次前向传播 | ⭐⭐⭐⭐⭐ |
| **推理** | `generate()` | 自回归生成 | ⭐⭐⭐⭐⭐ |
| **模式** | `eval()` | 切换到评估模式 | ⭐⭐⭐⭐⭐ |
| **模式** | `train()` | 切换到训练模式 | ⭐⭐⭐⭐ |
| **设备** | `to(device)` | 设备迁移 | ⭐⭐⭐⭐⭐ |
| **参数** | `parameters()` | 访问模型参数 | ⭐⭐⭐⭐ |
| **状态** | `state_dict()` | 保存模型状态 | ⭐⭐⭐⭐ |
| **状态** | `load_state_dict()` | 加载模型状态 | ⭐⭐⭐⭐ |

---

## 核心推理方法

### 1. `forward()` / `__call__()` - 单次前向传播

**作用**：执行一次前向传播，返回 logits 和可选的 KV cache。

**代码示例**：
```python
# 方式 1: 直接调用（推荐）
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    use_cache=True,  # 返回 KV cache
)

# 方式 2: 显式调用 forward
outputs = model.forward(
    input_ids=input_ids,
    attention_mask=attention_mask,
    use_cache=True,
)

# 访问输出
logits = outputs.logits  # [batch_size, seq_len, vocab_size]
past_key_values = outputs.past_key_values  # KV cache（如果 use_cache=True）
```

**在你的代码中**：
```33:36:projects/llm-infer-baseline/infer.py
            prefill_outputs = model(
                input_ids=inputs["input_ids"],
                use_cache=True,
            )
```

**关键特性**：
- **单次前向传播**：只处理一次输入，不进行自回归生成
- **返回 logits**：每个位置的词汇表概率分布
- **KV Cache 支持**：可以返回 past_key_values 用于加速后续生成
- **灵活控制**：可以精确控制每一步的计算

**使用场景**：
- 手动实现生成循环（更精细的控制）
- Prefill 阶段（处理 prompt）
- 单步解码（decode 阶段）
- 获取中间层输出

**与 generate() 的区别**：
- `forward()`：单次前向传播，需要手动实现生成循环
- `generate()`：自动完成整个生成过程（prefill + decode loop）

---

### 2. `generate()` - 自回归生成

**作用**：自动完成从 prompt 到完整生成的整个过程。

**代码示例**：
```python
outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    do_sample=False,  # 贪婪解码
    temperature=1.0,  # 采样温度
    top_p=0.9,        # nucleus sampling
    top_k=50,         # top-k sampling
    num_beams=4,      # beam search
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

**关键特性**：
- **自动生成循环**：内部实现 prefill + decode loop
- **多种解码策略**：贪婪、采样、beam search
- **停止条件**：自动处理 EOS token
- **批处理支持**：可以同时生成多个序列

**使用场景**：
- 快速原型开发
- 标准文本生成任务
- 不需要精细控制的场景

---

## 模式管理方法

### 3. `eval()` - 切换到评估模式

**作用**：将模型切换到评估模式，禁用训练时的随机性。

**代码示例**：
```python
model.eval()  # 切换到评估模式
```

**在你的代码中**：
```13:13:projects/llm-infer-baseline/model.py
    model.eval()
```

**关键特性**：
- **禁用 Dropout**：所有 dropout 层输出保持不变
- **禁用 BatchNorm 更新**：使用训练时的统计量
- **确保确定性**：推理结果可重复
- **节省内存**：不需要存储梯度信息

**内部变化**：
```python
# eval() 内部会：
for module in model.modules():
    if isinstance(module, torch.nn.Dropout):
        module.train(False)  # 禁用 dropout
    if isinstance(module, torch.nn.BatchNorm1d):
        module.eval()  # 使用固定统计量
```

**使用场景**：
- 推理前必须调用
- 评估模型性能
- 生成文本

**注意**：
- 推理前**必须**调用 `model.eval()`
- 与 `torch.no_grad()` 配合使用效果更好

---

### 4. `train()` - 切换到训练模式

**作用**：将模型切换到训练模式，启用训练时的随机性。

**代码示例**：
```python
model.train()  # 切换到训练模式
```

**关键特性**：
- **启用 Dropout**：随机丢弃部分神经元
- **启用 BatchNorm 更新**：更新 running statistics
- **启用梯度计算**：为反向传播做准备

**使用场景**：
- 模型训练
- Fine-tuning
- 继续训练

**注意**：
- 推理时**不要**调用 `train()`
- 训练时必须调用 `train()`

---

## 设备管理方法

### 5. `to(device)` - 设备迁移

**作用**：将模型迁移到指定设备（CPU 或 GPU）。

**代码示例**：
```python
model = model.to("cuda")      # 迁移到 GPU
model = model.to("cpu")        # 迁移到 CPU
model = model.to(torch.device("cuda:0"))  # 指定 GPU 设备
```

**在你的代码中**：
```12:12:projects/llm-infer-baseline/model.py
    model.to(DEVICE)
```

**关键特性**：
- **递归迁移**：所有参数和缓冲区都迁移
- **设备统一**：确保模型组件在同一设备
- **内存管理**：在目标设备上分配内存/显存

**使用场景**：
- 模型加载后迁移到 GPU
- 多 GPU 训练时分配设备
- 动态设备切换

**详细说明**：参考 `to_device_explained.md`

---

## 参数访问方法

### 6. `parameters()` - 访问模型参数

**作用**：返回模型的所有可训练参数（权重和偏置）。

**代码示例**：
```python
# 获取所有参数
for param in model.parameters():
    print(param.shape)
    print(param.requires_grad)  # 是否计算梯度

# 获取参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数数: {total_params:,}")

# 获取可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数: {trainable_params:,}")
```

**关键特性**：
- **迭代器**：返回参数的迭代器
- **包含所有层**：递归访问所有子模块
- **可训练参数**：默认所有参数都 `requires_grad=True`

**使用场景**：
- 统计模型大小
- 参数初始化
- 梯度裁剪
- 参数冻结/解冻

---

### 7. `named_parameters()` - 命名参数访问

**作用**：返回带名称的参数，便于识别和操作特定层。

**代码示例**：
```python
# 获取所有命名参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 输出示例：
# transformer.wte.weight: torch.Size([50257, 768])
# transformer.wpe.weight: torch.Size([1024, 768])
# transformer.h.0.attn.c_attn.weight: torch.Size([768, 2304])
# ...

# 冻结特定层
for name, param in model.named_parameters():
    if "embedding" in name:
        param.requires_grad = False  # 冻结嵌入层
```

**关键特性**：
- **带名称**：参数名称包含层级信息
- **便于调试**：可以精确定位参数
- **选择性操作**：可以针对特定层进行操作

**使用场景**：
- 参数可视化
- 选择性冻结/解冻
- 参数分析
- 调试模型结构

---

### 8. `buffers()` / `named_buffers()` - 访问缓冲区

**作用**：返回模型的缓冲区（如 BatchNorm 的 running_mean）。

**代码示例**：
```python
# 获取所有缓冲区
for buffer in model.buffers():
    print(buffer.shape)

# 获取命名缓冲区
for name, buffer in model.named_buffers():
    print(f"{name}: {buffer.shape}")
```

**关键特性**：
- **非可训练**：缓冲区不参与梯度计算
- **状态信息**：存储模型的状态信息
- **BatchNorm**：running_mean, running_var 等

**使用场景**：
- 模型状态分析
- 状态迁移
- 调试

---

## 状态保存和加载

### 9. `state_dict()` - 保存模型状态

**作用**：返回模型的状态字典（所有参数和缓冲区）。

**代码示例**：
```python
# 获取状态字典
state_dict = model.state_dict()

# 保存到文件
torch.save(state_dict, "model.pth")

# 查看状态字典内容
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")
```

**关键特性**：
- **完整状态**：包含所有参数和缓冲区
- **可序列化**：可以保存到文件
- **轻量级**：只包含数据，不包含模型结构

**使用场景**：
- 模型检查点保存
- 模型分享
- 模型版本管理

---

### 10. `load_state_dict()` - 加载模型状态

**作用**：从状态字典加载模型参数。

**代码示例**：
```python
# 加载状态字典
state_dict = torch.load("model.pth")
model.load_state_dict(state_dict)

# 严格模式（默认）
model.load_state_dict(state_dict, strict=True)  # 必须完全匹配

# 非严格模式
model.load_state_dict(state_dict, strict=False)  # 允许部分匹配
```

**关键特性**：
- **参数恢复**：恢复模型的参数值
- **严格模式**：确保完全匹配
- **非严格模式**：允许部分加载

**使用场景**：
- 加载检查点
- 模型迁移
- 参数初始化

---

## 其他重要方法

### 11. `modules()` / `named_modules()` - 访问所有模块

**作用**：返回模型的所有子模块。

**代码示例**：
```python
# 获取所有模块
for module in model.modules():
    print(type(module))

# 获取命名模块
for name, module in model.named_modules():
    print(f"{name}: {type(module)}")
```

**使用场景**：
- 模型结构分析
- 模块级操作
- 模型可视化

---

### 12. `children()` - 访问直接子模块

**作用**：返回模型的直接子模块（不递归）。

**代码示例**：
```python
# 获取直接子模块
for child in model.children():
    print(type(child))
```

**使用场景**：
- 访问顶层模块
- 模块替换

---

### 13. `zero_grad()` - 清零梯度

**作用**：将所有参数的梯度清零。

**代码示例**：
```python
# 训练循环中
optimizer.zero_grad()  # 通常使用 optimizer
# 或
model.zero_grad()  # 直接调用模型方法
```

**使用场景**：
- 训练循环
- 梯度累积前清零

---

### 14. `half()` / `float()` - 数据类型转换

**作用**：转换模型的数据类型。

**代码示例**：
```python
model = model.half()   # 转换为 float16
model = model.float()  # 转换为 float32
model = model.double() # 转换为 float64
```

**使用场景**：
- 显存优化（float16）
- 精度要求（float32）
- 模型量化

---

## 方法对比和使用场景

### forward() vs generate()

| 特性 | forward() | generate() |
|------|-----------|------------|
| **用途** | 单次前向传播 | 完整生成过程 |
| **返回** | logits + KV cache | 完整 token 序列 |
| **控制** | 精细控制每一步 | 自动完成所有步骤 |
| **性能** | 需要手动优化 | 内部已优化 |
| **使用场景** | 手动生成循环 | 快速原型 |

**选择建议**：
- **需要精细控制**：使用 `forward()`
- **快速开发**：使用 `generate()`
- **性能优化**：使用 `forward()` + 手动循环

---

### eval() vs train()

| 特性 | eval() | train() |
|------|--------|---------|
| **Dropout** | 禁用 | 启用 |
| **BatchNorm** | 固定统计量 | 更新统计量 |
| **梯度** | 通常不需要 | 需要 |
| **使用场景** | 推理、评估 | 训练 |

**选择建议**：
- **推理前**：必须调用 `eval()`
- **训练时**：必须调用 `train()`

---

### parameters() vs named_parameters()

| 特性 | parameters() | named_parameters() |
|------|---------------|-------------------|
| **返回** | 参数迭代器 | (名称, 参数) 元组 |
| **信息** | 只有参数值 | 包含名称 |
| **使用场景** | 统计、初始化 | 选择性操作、调试 |

**选择建议**：
- **需要名称**：使用 `named_parameters()`
- **只需要值**：使用 `parameters()`

---

## 实际使用示例

### 示例 1：手动生成循环（使用 forward）

```python
# 使用 forward() 实现手动生成
model.eval()
input_ids = tokenizer.encode("Hello", return_tensors="pt").to(device)

# Prefill 阶段
outputs = model(input_ids=input_ids, use_cache=True)
past_key_values = outputs.past_key_values
next_token_logits = outputs.logits[:, -1, :]

# Decode 循环
generated_ids = []
for _ in range(50):
    # 选择下一个 token
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_ids.append(next_token_id)
    
    # 如果遇到 EOS，停止
    if next_token_id == tokenizer.eos_token_id:
        break
    
    # 下一步前向传播（使用 KV cache）
    outputs = model(
        input_ids=next_token_id,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

# 解码结果
generated_text = tokenizer.decode(torch.cat(generated_ids, dim=1)[0])
```

### 示例 2：参数统计和分析

```python
# 统计模型参数
total_params = 0
trainable_params = 0

for name, param in model.named_parameters():
    num_params = param.numel()
    total_params += num_params
    if param.requires_grad:
        trainable_params += num_params
    print(f"{name}: {num_params:,} params")

print(f"总参数: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
```

### 示例 3：选择性冻结参数

```python
# 冻结嵌入层，只训练顶层
for name, param in model.named_parameters():
    if "embedding" in name or "wte" in name or "wpe" in name:
        param.requires_grad = False
        print(f"冻结: {name}")
    else:
        print(f"训练: {name}")
```

---

## 总结

### 必须掌握的方法（⭐⭐⭐⭐⭐）

1. **`forward()` / `__call__()`**：单次前向传播，手动生成的基础
2. **`generate()`**：自动生成，快速开发
3. **`eval()`**：推理前必须调用
4. **`to(device)`**：设备迁移，GPU 加速

### 重要方法（⭐⭐⭐⭐）

5. **`train()`**：训练模式切换
6. **`parameters()` / `named_parameters()`**：参数访问
7. **`state_dict()` / `load_state_dict()`**：模型保存和加载

### 有用方法（⭐⭐⭐）

8. **`buffers()`**：缓冲区访问
9. **`modules()`**：模块访问
10. **`zero_grad()`**：梯度清零
11. **`half()` / `float()`**：数据类型转换

### 学习建议

1. **先掌握核心方法**：`forward()`, `generate()`, `eval()`, `to()`
2. **理解方法区别**：`forward()` vs `generate()`, `eval()` vs `train()`
3. **实践应用**：尝试手动实现生成循环
4. **深入理解**：了解每个方法的内部机制

---

## 相关资源

- **PyTorch 文档**：https://pytorch.org/docs/stable/nn.html
- **Hugging Face 文档**：https://huggingface.co/docs/transformers/
- **模型方法参考**：查看具体模型类的文档


