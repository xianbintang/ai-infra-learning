# inspect_prefill_decode.py 详解

## 目录
1. [这个文件想教我们什么？](#这个文件想教我们什么)
2. [为什么要这么做？](#为什么要这么做)
3. [代码逐段解析](#代码逐段解析)
4. [输出详细解释](#输出详细解释)
5. [如何理解 4 维张量](#如何理解-4-维张量)
6. [distilgpt2 的 6 层 Transformer 结构](#distilgpt2-的-6-层-transformer-结构)
7. [pkv (past_key_values) 详解](#pkv-past_key_values-详解)
8. [关键概念图解](#关键概念图解)
9. [为什么这些知识重要？](#为什么这些知识重要)
10. [实验建议](#实验建议)
11. [总结](#总结)

---

## 这个文件想教我们什么？

**核心目标**：让你亲眼看到 LLM 推理的两个关键阶段——**Prefill** 和 **Decode**——是如何工作的。

这是理解 LLM 推理性能优化（如 KV Cache、vLLM、Flash Attention）的**基础中的基础**。

---

## 为什么要这么做？

### 问题：`model.generate()` 是一个"黑盒"

当你使用 `model.generate()` 时：
```python
outputs = model.generate(**inputs, max_new_tokens=50)
```

你看不到内部发生了什么。它隐藏了：
1. Prefill 和 Decode 的区别
2. KV Cache 如何工作
3. 每一步是如何选择下一个 token 的

### 解决方案：手动实现生成循环

这个脚本通过**手动实现生成循环**，让你看到：
1. Prefill 阶段处理整个 prompt
2. Decode 阶段逐 token 生成
3. KV Cache 如何累积
4. 每一步的 token 选择过程

---

## 代码逐段解析

### 第一部分：辅助函数

```python
def topk_tokens(tokenizer, logits_1d, k=5):
    # logits_1d: [vocab_size]
    vals, idx = torch.topk(logits_1d, k)
    items = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        tok = tokenizer.decode([i])
        items.append((i, tok.replace("\n", "\\n"), v))
    return items
```

**作用**：显示模型认为最可能的 top-5 个候选 token

**为什么需要**：
- 帮助你理解模型如何"思考"
- 看到除了被选中的 token 之外，还有哪些候选
- 理解 logits 和概率的关系

---

### 第二部分：输入处理

```python
inputs = tok(PROMPT, return_tensors="pt").to(DEVICE)
input_ids = inputs["input_ids"]
print("\n[INPUT]")
print("prompt:", repr(PROMPT))
print("input_ids.shape:", tuple(input_ids.shape))
print("input_ids (first 20):", input_ids[0, :20].tolist())
```

**作用**：展示 tokenizer 如何将文本转换为 token IDs

**输出示例**：
```
[INPUT]
prompt: 'Q: Explain why batching improves GPU utilization.\nA:'
input_ids.shape: (1, 13)
input_ids (first 20): [48, 25, 48696, 1521, 15458, 278, ...]
```

**关于 `input_ids[0, :20]` 的说明**：

为什么 `input_ids` 只有 13 列，但可以取前 20 个？

因为 **PyTorch（和 NumPy）的切片是安全的**：超出范围不会报错，只返回实际存在的元素。

```python
# 假设 input_ids 只有 13 个 token
input_ids[0, :20]  # 不会报错，返回全部 13 个

# 等价于
input_ids[0, :13]  # 实际只有 13 个
```

这是**防御性编程**：
- 不知道确切长度时，写一个足够大的数
- 如果 prompt 很长（超过 20 个 token），就只显示前 20 个
- 如果 prompt 很短（不到 20 个），就显示全部

**对比**：
```python
# ❌ 索引会报错
input_ids[0, 20]  # IndexError: 超出范围

# ✅ 切片不会报错
input_ids[0, :20]  # 安全，返回实际存在的元素
```

**教你什么**：
- prompt 被转换为 13 个 token
- 每个 token 是一个整数 ID
- 这些 ID 会输入到模型中

---

### 第三部分：Prefill 阶段（核心！）

```python
# ---------- Prefill ----------
with torch.no_grad():
    out = model(input_ids=input_ids, use_cache=True)

logits = out.logits
pkv = out.past_key_values

print("\n[PREFILL OUTPUT]")
print("logits.shape:", tuple(logits.shape), " = [batch, seq_len, vocab]")
print("num_layers in past_key_values:", len(pkv))
k0, v0 = pkv[0]
print("layer0 K.shape:", tuple(k0.shape))
print("layer0 V.shape:", tuple(v0.shape))
```

**作用**：展示 Prefill 阶段的输入和输出

**输出示例**：
```
[PREFILL OUTPUT]
logits.shape: (1, 13, 50257)  = [batch, seq_len, vocab]
num_layers in past_key_values: 6
layer0 K.shape: (1, 12, 13, 64)
layer0 V.shape: (1, 12, 13, 64)
```

---

### 第四部分：选择第一个生成的 token

```python
# 取最后一个位置的 logits（用来选"下一个 token"）
next_logits = logits[0, -1]  # [vocab]
print("\n[PREFILL -> NEXT TOKEN]")
print("next_logits.shape:", tuple(next_logits.shape))
print("top-5 candidates (id, token, logit):")
for i, t, v in topk_tokens(tok, next_logits, k=5):
    print(f"  {i:6d}  {t!r:12s}  logit={v:.4f}")

next_id = torch.argmax(next_logits).view(1, 1)  # [1,1]
print("chosen next_id:", next_id.item(), "token:", repr(tok.decode(next_id[0].tolist())))
```

**输出示例**：
```
[PREFILL -> NEXT TOKEN]
next_logits.shape: (50257,)
top-5 candidates (id, token, logit):
     383  ' The'        logit=-66.5244
     314  ' I'          logit=-66.5982
     775  ' We'         logit=-66.8043
     632  ' It'         logit=-66.8370
     921  ' You'        logit=-66.8639
chosen next_id: 383 token: ' The'
```

---

### 第五部分：Decode 循环（核心！）

```python
# ---------- Decode loop ----------
generated = [next_id]
past = pkv

for step in range(1, MAX_NEW_TOKENS):
    with torch.no_grad():
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)

    logits = out.logits            # [1, 1, vocab]  (通常 seq_len=1)
    past = out.past_key_values     # K/V 的 seq_len 增长
    next_logits = logits[0, -1]

    k0, v0 = past[0]
    print(f"\n[DECODE step {step}]")
    print("input token id:", next_id.item(), "token:", repr(tok.decode([next_id.item()])))
    print("logits.shape:", tuple(logits.shape))
    print("layer0 K.shape:", tuple(k0.shape), "(注意 seq_len 在增长)")
    print("top-5 candidates (id, token, logit):")
    for i, t, v in topk_tokens(tok, next_logits, k=5):
        print(f"  {i:6d}  {t!r:12s}  logit={v:.4f}")

    next_id = torch.argmax(next_logits).view(1, 1)
    generated.append(next_id)
```

**输出示例**：
```
[DECODE step 1]
input token id: 383 token: ' The'
logits.shape: (1, 1, 50257)
layer0 K.shape: (1, 12, 14, 64) (注意 seq_len 在增长)
top-5 candidates:
    464  ' It'        logit=9.8765
    ...

[DECODE step 2]
input token id: 464 token: ' It'
logits.shape: (1, 1, 50257)
layer0 K.shape: (1, 12, 15, 64) (注意 seq_len 在增长)
...
```

---

## 输出详细解释

### `logits.shape: (1, 13, 50257)` 每个数字的含义

| 维度位置 | 值 | 含义 |
|---------|-----|------|
| 第 1 维 | `1` | batch_size：1 个样本（1 条 prompt） |
| 第 2 维 | `13` | seq_len：prompt 有 13 个 token |
| 第 3 维 | `50257` | vocab_size：词汇表大小（每个位置预测 50257 个候选词的分数） |

### `num_layers: 6` 的含义

- distilgpt2 有 **6 个 Transformer 层**
- 每层都有自己的 KV Cache
- 所以 `past_key_values` 是一个包含 6 个元素的元组

### `layer0 K.shape: (1, 12, 13, 64)` 每个数字的含义

| 维度位置 | 值 | 含义 |
|---------|-----|------|
| 第 1 维 | `1` | batch_size：1 个样本 |
| 第 2 维 | `12` | num_heads：12 个注意力头 |
| 第 3 维 | `13` | seq_len：缓存了 13 个位置的 Key |
| 第 4 维 | `64` | head_dim：每个头的维度（768 ÷ 12 = 64） |

### `layer0 V.shape: (1, 12, 13, 64)` 

- 与 K 相同，Value 也是 `(1, 12, 13, 64)`

**关键点**：
- Decode 时，`seq_len` 会从 13 → 14 → 15 → ... 逐步增长
- 隐藏维度 768 = 12 头 × 64 维/头

### `[PREFILL -> NEXT TOKEN]` 输出详解

```
[PREFILL -> NEXT TOKEN]
next_logits.shape: (50257,)
top-5 candidates (id, token, logit):
     383  ' The'        logit=-66.5244
     314  ' I'          logit=-66.5982
     775  ' We'         logit=-66.8043
     632  ' It'         logit=-66.8370
     921  ' You'        logit=-66.8639
chosen next_id: 383 token: ' The'
```

**每个数字的含义**：

| 输出 | 含义 |
|------|------|
| `next_logits.shape: (50257,)` | 50257 个词的"分数"（词汇表大小） |
| `383` | Token ID |
| `' The'` | Token ID 383 对应的文本（注意前面有空格） |
| `logit=-66.5244` | 模型给这个词的原始分数（越大越好） |
| `chosen next_id: 383` | 选择了分数最高的 token |

**这一步之前做了什么？**

```
1. 输入 prompt: "Q: Explain why batching improves GPU utilization.\nA:"
   ↓
2. Prefill: 模型一次性处理 13 个 token
   ↓
3. 输出 logits: (1, 13, 50257)
   ↓
4. 取最后一个位置: logits[0, -1] → (50257,)  ← 就是这里！
   （最后一个位置预测"下一个词"）
```

**这一步在做什么？**

```python
next_logits = logits[0, -1]  # 取最后位置的 50257 个分数
# top-5: 找出分数最高的 5 个词
# argmax: 选分数最大的 → ' The' (383)
```

**后面要做什么？**

```
1. 把选中的 token (383 = ' The') 作为下一个输入
   ↓
2. 进入 Decode 循环:
   - 输入: [383]（1 个 token）
   - 使用 KV Cache（不重新算 prompt）
   - 输出: 下一个 token 的 logits
   ↓
3. 重复直到生成 MAX_NEW_TOKENS 个
```

**为什么 logits 是负数？**

- logits 是原始分数，可正可负
- 不是概率！要经过 softmax 才变成概率
- 只关心相对大小：`-66.52 > -66.60`，所以 `' The'` 胜出

---

## 如何理解 4 维张量

### 张量形状：`(1, 12, 13, 64)`

如何想象这个 4D 张量？

### 方式 1：嵌套的表格/Excel

```
batch=0 的数据（只有 1 个样本，所以就 1 个）
│
└── 12 个注意力头（想象成 12 张表格）
    │
    ├── Head 0 的表格:
    │     token0: [64 个数字]
    │     token1: [64 个数字]
    │     ...
    │     token12: [64 个数字]
    │     → 共 13 行，每行 64 列
    │
    ├── Head 1 的表格:
    │     （同样 13 行 × 64 列）
    │
    └── ... Head 11
```

### 方式 2：立体想象

```
                    64 (head_dim)
                   ←─────────→
              ┌─────────────────┐
             /│                /│
            / │   Head 0      / │
     13    /  │              /  │
   (seq)  ┌───┼─────────────┐   │
          │   │             │   │
          │   │  [数字矩阵]  │   │  ↑
          │   │  13×64      │   │  │ 12 个头
          │   └─────────────┼───┘  │ (堆叠)
          │  /              │  /   ↓
          │ /    Head 1     │ /
          └─────────────────┘
              ... × 12 层
```

### 方式 3：代码访问

```python
K[0]           # 形状 (12, 13, 64) → 第 1 个样本的所有头
K[0, 0]        # 形状 (13, 64)     → 第 1 个样本，Head 0 的 Key 矩阵
K[0, 0, 0]     # 形状 (64,)        → Head 0，token 0 的 Key 向量
K[0, 0, 0, 0]  # 标量              → 具体某个数字
```

### 方式 4：实际含义

```
K[batch, head, token_pos, :] = 64 维向量

这个向量代表：
- 第 batch 个样本
- 第 head 个注意力头
- 第 token_pos 个 token
- 的 Key 表示（64 个数字描述这个 token 的"特征"）
```

**简单记忆**：12 个头，每个头都记住了 13 个 token，每个 token 用 64 个数字表示。

---

## distilgpt2 的 6 层 Transformer 结构

### 这 6 层是什么？

**这 6 层是相同结构的 Transformer Decoder Block，只是权重不同**。

### 每一层（Block）内部结构

```
Transformer Block (×6 层，结构相同)
├── LayerNorm 1
├── Self-Attention（自注意力）
│   ├── Q, K, V 投影（c_attn）
│   └── 输出投影（c_proj）
├── 残差连接
├── LayerNorm 2
├── MLP/FFN（前馈网络）
│   ├── 扩展层（c_fc）: 768 → 3072
│   └── 收缩层（c_proj）: 3072 → 768
└── 残差连接
```

### 模型整体结构

```
Input
  ↓
词嵌入 (wte) + 位置嵌入 (wpe)
  ↓
Transformer Block 0  ← 第 1 层
  ↓
Transformer Block 1  ← 第 2 层
  ↓
Transformer Block 2  ← 第 3 层
  ↓
Transformer Block 3  ← 第 4 层
  ↓
Transformer Block 4  ← 第 5 层
  ↓
Transformer Block 5  ← 第 6 层
  ↓
LayerNorm (最后)
  ↓
LM Head (输出词汇表概率)
```

**所以**：`past_key_values` 有 6 个元素，每个元素是该层的 (K, V) 缓存。

### 如何验证真的用到了 6 层？

代码已经打印了层数：
```python
print("num_layers in past_key_values:", len(pkv))  # 输出: 6
```

这一行证明 `pkv` 里有 **6 个元素**（6 层）。

你也可以遍历所有层来验证：
```python
# 当前代码只取了第 0 层
k0, v0 = pkv[0]

# 你可以遍历所有层来验证：
for i, (k, v) in enumerate(pkv):
    print(f"layer{i} K.shape: {tuple(k.shape)}")
    print(f"layer{i} V.shape: {tuple(v.shape)}")
```

**预期输出**：
```
layer0 K.shape: (1, 12, 13, 64)
layer1 K.shape: (1, 12, 13, 64)
layer2 K.shape: (1, 12, 13, 64)
layer3 K.shape: (1, 12, 13, 64)
layer4 K.shape: (1, 12, 13, 64)
layer5 K.shape: (1, 12, 13, 64)
```

**为什么只打印 layer0**：因为所有层形状相同，打印一个就够了，避免输出太多。

---

## pkv (past_key_values) 详解

### 1. pkv 是贯穿全过程的变量吗？

**是的！** 它就是 **KV Cache**，在整个生成过程中不断累积。

```python
# Prefill 后
pkv = out.past_key_values  # 初始化，包含 prompt 的 K/V

# Decode 循环中
past = pkv  # 传入
out = model(input_ids=next_id, past_key_values=past, use_cache=True)
past = out.past_key_values  # 更新（增长了 1 个位置）
```

### 2. 为什么需要它？

**避免重复计算！**

```
没有 KV Cache:
  每次都要重新计算所有 token 的 K/V → 慢

有 KV Cache:
  只计算新 token 的 K/V，拼接到历史 → 快
```

### 3. 数据如何变化？

```
Prefill 后:   pkv[0][0].shape = (1, 12, 13, 64)  ← 13 个 token
Decode 步 1:  past[0][0].shape = (1, 12, 14, 64)  ← 增长到 14
Decode 步 2:  past[0][0].shape = (1, 12, 15, 64)  ← 增长到 15
Decode 步 3:  past[0][0].shape = (1, 12, 16, 64)  ← 增长到 16
...
```

**只有 seq_len 维度在增长**，其他维度不变。

### 4. 完整数据结构

```python
pkv = (
    (K_layer0, V_layer0),   # pkv[0]
    (K_layer1, V_layer1),   # pkv[1]
    (K_layer2, V_layer2),   # pkv[2]
    (K_layer3, V_layer3),   # pkv[3]
    (K_layer4, V_layer4),   # pkv[4]
    (K_layer5, V_layer5),   # pkv[5]
)
len(pkv) == 6  # ✅ 6 层

每个 K 或 V: (batch=1, heads=12, seq_len=变化, dim=64)
```

### 5. 如何想象和定位数据？

```python
pkv[layer][0]  # K  或  pkv[layer][1]  # V

# 定位一个具体数字：
pkv[2][0][0, 5, 10, 30]
#   │  │  │  │  │   └── head_dim 第 30 维
#   │  │  │  │  └────── seq_len 第 10 个 token
#   │  │  │  └───────── num_heads 第 5 个注意力头
#   │  │  └──────────── batch 第 0 个样本
#   │  └─────────────── 0=K, 1=V
#   └────────────────── 第 2 层 Transformer
```

### 6. 具体数字代表什么？

```
pkv[2][0][0, 5, 10, 30] = 0.1234
         │  │  │   │
         │  │  │   └── 第 30 个特征维度的值
         │  │  └────── "token 10 在 head 5 眼中的 Key 特征"
         │  └───────── head 5 有自己独特的"视角"
         └──────────── 样本 0

含义：第 2 层 Transformer 中，注意力头 5 对 token 10 
      计算出的 Key 向量的第 30 个分量
```

### 7. 总数据量计算

```
总数据量 = 6层 × 2(K+V) × 1(batch) × 12(heads) × seq_len × 64(dim)
         = 1536 × seq_len 个浮点数

对于 seq_len=13:
  = 1536 × 13 = 19,968 个浮点数
  = 19,968 × 4 bytes (float32) ≈ 80 KB

对于 seq_len=1000:
  = 1536 × 1000 = 1,536,000 个浮点数
  ≈ 6 MB
```

**简单理解**：pkv 记住了"每一层、每个头、每个历史 token 的特征"，用于后续计算注意力。

---

## 关键概念图解

### Prefill vs Decode

```
Prefill 阶段（处理 prompt）:
┌─────────────────────────────────────┐
│ 输入: [token1, token2, ..., token13] │  ← 整个 prompt
│ 输出: logits [1, 13, vocab]          │  ← 每个位置的预测
│ KV Cache: [1, 12, 13, 64]            │  ← 13 个位置的 K/V
└─────────────────────────────────────┘

Decode 阶段（逐 token 生成）:
┌─────────────────────────────────────┐
│ Step 1:                              │
│   输入: [token14]                    │  ← 只有 1 个 token
│   KV Cache: [1, 12, 14, 64]          │  ← 增长到 14
│                                      │
│ Step 2:                              │
│   输入: [token15]                    │  ← 只有 1 个 token
│   KV Cache: [1, 12, 15, 64]          │  ← 增长到 15
│   ...                                │
└─────────────────────────────────────┘
```

### KV Cache 的作用

```
没有 KV Cache（每次都重新计算）:
Step 1: 计算 [token1, ..., token13, token14] 的注意力
Step 2: 计算 [token1, ..., token13, token14, token15] 的注意力
Step 3: 计算 [token1, ..., token13, token14, token15, token16] 的注意力
→ 时间复杂度: O(n²)，n 是总长度

有 KV Cache（复用历史计算）:
Prefill: 计算 [token1, ..., token13] 的 K/V，保存起来
Step 1: 只计算 [token14] 的 K/V，与历史拼接
Step 2: 只计算 [token15] 的 K/V，与历史拼接
→ 时间复杂度: O(n)，每步只计算 1 个 token
```

---

## 为什么这些知识重要？

### 1. 理解推理性能瓶颈

| 阶段 | 特点 | 瓶颈 |
|------|------|------|
| Prefill | 一次处理多 token，计算量大 | **计算密集型** |
| Decode | 每次只处理 1 个 token，但要重复很多次 | **内存带宽密集型** |

- Prefill 慢是因为计算量大
- Decode 慢是因为内存访问（每次都要读取 KV Cache）

### 2. 理解优化技术

| 优化技术 | 优化什么 | 为什么有效 |
|---------|---------|-----------|
| **KV Cache** | Decode 阶段 | 避免重复计算历史 token |
| **Flash Attention** | Prefill 阶段 | 减少显存访问 |
| **vLLM PagedAttention** | KV Cache 管理 | 减少显存碎片 |
| **Continuous Batching** | Decode 阶段 | 提高 GPU 利用率 |

### 3. 理解为什么长上下文慢

```
KV Cache 大小 = num_layers × 2 × batch × num_heads × seq_len × head_dim

对于 distilgpt2 (6 层, 12 头, 64 维):
  seq_len=100:  KV Cache ≈ 6×2×1×12×100×64 ≈ 920 KB
  seq_len=1000: KV Cache ≈ 6×2×1×12×1000×64 ≈ 9.2 MB
  seq_len=10000: KV Cache ≈ 92 MB

对于大模型 (32 层, 32 头, 128 维):
  seq_len=10000: KV Cache ≈ 32×2×1×32×10000×128 ≈ 2.6 GB
```

---

## 实验建议

### 实验 1：修改 MAX_NEW_TOKENS

```python
MAX_NEW_TOKENS = 20  # 观察更长的生成过程
```

观察 KV Cache 如何持续增长。

### 实验 2：观察 top-5 候选的变化

每一步的 top-5 候选告诉你：
- 模型认为哪些词最可能
- 贪婪解码是否选择了最佳路径

### 实验 3：比较 Prefill 和 Decode 的耗时

```python
import time

# Prefill
start = time.time()
out = model(input_ids=input_ids, use_cache=True)
prefill_time = time.time() - start

# Decode (单步)
start = time.time()
out = model(input_ids=next_id, past_key_values=past, use_cache=True)
decode_time = time.time() - start

print(f"Prefill: {prefill_time:.4f}s (处理 {prompt_len} tokens)")
print(f"Decode: {decode_time:.4f}s (处理 1 token)")
```

### 实验 4：打印所有层的 KV Cache

```python
# 验证所有 6 层
for i, (k, v) in enumerate(pkv):
    print(f"layer{i} K.shape: {tuple(k.shape)}, V.shape: {tuple(v.shape)}")
```

### 实验 5：查看具体数值

```python
# 查看某个具体的 Key 向量
k0, v0 = pkv[0]
print("layer0, head0, token0 的 Key 向量 (前 10 维):")
print(k0[0, 0, 0, :10])
```

---

## 总结

### 这个文件教你的核心知识

1. **Prefill 阶段**：
   - 一次性处理整个 prompt
   - 生成所有位置的 KV Cache
   - 计算密集型

2. **Decode 阶段**：
   - 每次只处理 1 个新 token
   - 使用 KV Cache 避免重复计算
   - 内存带宽密集型

3. **KV Cache (past_key_values)**：
   - 缓存历史 token 的 Key 和 Value
   - 每步增长 1 个位置
   - 贯穿整个生成过程
   - 是长上下文优化的关键

4. **Token 选择**：
   - logits → argmax/softmax → next token
   - 可以看到模型的"思考过程"

5. **张量形状理解**：
   - `logits: (batch, seq_len, vocab_size)`
   - `K/V: (batch, num_heads, seq_len, head_dim)`

6. **distilgpt2 结构**：
   - 6 个相同结构的 Transformer Block
   - 每层包含 Self-Attention + MLP

### 为什么这很重要

- 理解这些概念后，你才能理解：
  - 为什么 KV Cache 能加速推理
  - 为什么 vLLM 要用 PagedAttention
  - 为什么 Continuous Batching 能提高吞吐量
  - 为什么长上下文需要特殊优化

这是 LLM 推理优化的**基础知识**，必须掌握！
