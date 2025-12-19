# torch 和 PyTorch 的关系与区别

## 快速回答

**简单来说**：`torch` 和 `PyTorch` 是**同一个东西**，只是命名不同：

- **PyTorch**：框架的正式名称（品牌名、项目名）
- **torch**：Python 包的名称（安装和导入时使用）

**关系**：
```
PyTorch（框架名称）
    ↓
torch（Python 包名）
    ↓
import torch（代码中使用）
```

---

## 详细解释

### 1. 命名历史

#### PyTorch（框架名称）
- **正式名称**：PyTorch
- **来源**：Facebook（现 Meta）开发的深度学习框架
- **用途**：在文档、讨论、官网中使用
- **网站**：https://pytorch.org/

#### torch（Python 包名）
- **包名**：torch
- **来源**：继承自 Lua 版本的 Torch 框架
- **用途**：在代码、安装、导入时使用
- **安装**：`pip install torch`

### 2. 为什么叫 torch？

**历史原因**：
1. **Lua Torch**：PyTorch 的前身是 Lua 语言的 Torch 框架
2. **命名延续**：为了保持一致性，Python 版本也使用 `torch` 作为包名
3. **简洁性**：`torch` 比 `pytorch` 更短，代码中更易用

### 3. 实际使用

#### 安装时
```bash
# ✅ 正确：使用 torch 作为包名
pip install torch

# ❌ 错误：不存在 pytorch 包
pip install pytorch  # 这会失败或安装错误的包
```

#### 代码中
```python
# ✅ 正确：导入时使用 torch
import torch

# ❌ 错误：不能导入 pytorch
import pytorch  # 这会失败
```

#### 文档中
- 官方文档：通常说 "PyTorch 框架"
- 代码示例：使用 `import torch`
- 讨论：两种说法都可以，但代码中必须用 `torch`

---

## 代码中的使用

### 项目中的使用

在项目中，我们使用 `torch`：

```4:4:projects/llm-infer-baseline/infer.py
import torch
```

```6:10:projects/llm-infer-baseline/config.py
import torch

MODEL_NAME = "distilgpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
```

```4:4:projects/llm-infer-baseline/model.py
import torch
```

### requirements.txt

```1:2:projects/llm-infer-baseline/requirements.txt
torch
transformers
```

**注意**：`requirements.txt` 中使用的是 `torch`，不是 `pytorch`。

---

## 常见混淆

### 混淆 1：包名 vs 框架名

**问题**：为什么安装时用 `torch`，但框架叫 `PyTorch`？

**答案**：
- 框架名称：PyTorch（用于品牌和文档）
- 包名称：torch（用于安装和代码）
- 这是历史原因和命名约定

### 混淆 2：torch vs pytorch 包

**问题**：`pip install pytorch` 和 `pip install torch` 有什么区别？

**答案**：
- `pip install torch`：✅ 正确，安装 PyTorch 框架
- `pip install pytorch`：❌ 可能不存在或指向错误的包
- **始终使用 `torch`**

### 混淆 3：导入语句

**问题**：为什么不能 `import pytorch`？

**答案**：
- Python 包名是 `torch`，不是 `pytorch`
- 必须使用 `import torch`
- 这是 Python 包系统的要求

---

## 版本和变体

### 标准版本
```bash
pip install torch
```
- 包含 CPU 和 CUDA 支持
- 适用于大多数情况

### CPU 版本
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
- 仅 CPU 版本
- 更小，不包含 CUDA

### CUDA 版本
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
- 特定 CUDA 版本
- 需要指定 CUDA 版本号

### 所有版本都使用 `torch` 包名

无论安装哪个版本，包名都是 `torch`：
```python
import torch  # 所有版本都这样导入
```

---

## 与其他框架的对比

### TensorFlow
```python
# TensorFlow
import tensorflow as tf  # 包名是 tensorflow
```

### JAX
```python
# JAX
import jax  # 包名是 jax
```

### PyTorch
```python
# PyTorch
import torch  # 包名是 torch（不是 pytorch）
```

**对比**：
- TensorFlow：包名和框架名一致（tensorflow）
- JAX：包名和框架名一致（jax）
- PyTorch：包名（torch）和框架名（PyTorch）不一致

---

## 官方文档中的使用

### 官网
- **网站名称**：PyTorch.org
- **标题**：PyTorch
- **代码示例**：`import torch`

### GitHub
- **仓库名**：pytorch/pytorch
- **包名**：torch
- **导入**：`import torch`

### 文档
- **文档标题**：PyTorch Documentation
- **代码示例**：`import torch`
- **安装说明**：`pip install torch`

---

## 实际示例

### 示例 1：检查版本
```python
import torch

# 查看 PyTorch 版本
print(torch.__version__)
# 输出: 2.0.1

# 注意：虽然框架叫 PyTorch，但属性是 torch.__version__
```

### 示例 2：检查 CUDA
```python
import torch

# 检查 CUDA 是否可用
print(torch.cuda.is_available())
# 输出: True 或 False

# 注意：使用 torch.cuda，不是 pytorch.cuda
```

### 示例 3：创建张量
```python
import torch

# 创建张量
x = torch.tensor([1, 2, 3])
print(x)
# 输出: tensor([1, 2, 3])

# 注意：使用 torch.tensor，不是 pytorch.tensor
```

---

## 常见问题

### Q1: 为什么包名不叫 pytorch？

**A**: 历史原因：
- 继承自 Lua Torch 框架
- `torch` 更短，代码中更易用
- 保持与原始 Torch 的一致性

### Q2: 可以重命名导入吗？

**A**: 可以，但不推荐：
```python
# ✅ 标准用法
import torch

# ⚠️ 可以但不推荐
import torch as pytorch  # 可以，但容易混淆

# ❌ 错误
import pytorch  # 这会失败
```

### Q3: 如何确认安装的是 PyTorch？

**A**: 检查版本和功能：
```python
import torch

print(torch.__version__)  # 应该显示版本号
print(torch.cuda.is_available())  # 检查 CUDA
print(torch.__file__)  # 查看安装路径
```

### Q4: torch 和 torchvision 的关系？

**A**: 都是 PyTorch 生态的一部分：
- `torch`：核心框架
- `torchvision`：计算机视觉工具
- `torchaudio`：音频处理工具
- `torchtext`：文本处理工具

```python
import torch
import torchvision  # 需要单独安装
```

---

## 总结

### 关键要点

1. **PyTorch 和 torch 是同一个东西**
   - PyTorch：框架名称（品牌名）
   - torch：Python 包名（代码中使用）

2. **安装和导入都使用 `torch`**
   ```bash
   pip install torch
   ```
   ```python
   import torch
   ```

3. **不要使用 `pytorch` 作为包名**
   - `pip install pytorch`：❌ 错误
   - `import pytorch`：❌ 错误

4. **文档中可能说 "PyTorch"，但代码中用 `torch`**
   - 文档：PyTorch 框架
   - 代码：`import torch`

### 记忆技巧

- **框架名**：PyTorch（大写 P，大写 T）
- **包名**：torch（全小写）
- **导入**：`import torch`（全小写）
- **安装**：`pip install torch`（全小写）

### 类比

就像：
- **公司名**：Apple Inc.
- **产品名**：iPhone
- **代码中**：使用 `iphone`（如果是一个包）

或者：
- **框架名**：PyTorch
- **包名**：torch
- **代码中**：`import torch`

---

## 相关资源

- **官方网站**：https://pytorch.org/
- **GitHub**：https://github.com/pytorch/pytorch
- **文档**：https://pytorch.org/docs/
- **安装指南**：https://pytorch.org/get-started/locally/

---

## 项目中的使用总结

在 `llm-infer-baseline` 项目中：

1. **requirements.txt**：`torch`
2. **代码导入**：`import torch`
3. **使用**：`torch.cuda.is_available()`, `torch.no_grad()`, `torch.tensor()` 等
4. **框架名称**：在文档和讨论中说 "PyTorch"

**记住**：代码中始终使用 `torch`，文档中可以说 "PyTorch"。


