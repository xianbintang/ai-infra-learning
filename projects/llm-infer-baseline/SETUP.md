# 工程初始化与基本命令

## 初始化步骤

### 1. 创建虚拟环境（如果还没有）
```bash
cd projects/llm-infer-baseline
python3 -m venv venv
```

### 2. 激活虚拟环境
```bash
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

## 基本命令

### 运行 Day1 基线推理脚本
```bash
# 确保在虚拟环境中
source venv/bin/activate
python3 infer.py
```

### 检查环境配置
```bash
python3 -c "from config import describe_environment; print(describe_environment())"
```

### 测试模型加载
```bash
python3 -c "from model import load_model; from config import MODEL_NAME; m = load_model(MODEL_NAME); print('Model loaded successfully')"
```

### 退出虚拟环境
```bash
deactivate
```

## 项目结构说明

- `config.py` - 全局配置（模型名称、设备、数据类型）
- `model.py` - 模型加载工具
- `tokenizer.py` - Tokenizer 加载工具
- `metrics.py` - 性能指标采集工具
- `infer.py` - Day1 基线推理脚本
- `requirements.txt` - Python 依赖包列表

## 注意事项

- 首次运行会从 Hugging Face 下载模型（distilgpt2），需要网络连接
- 如果使用 GPU，确保已安装 CUDA 版本的 PyTorch
- 在 macOS 上默认使用 CPU，模型会以 float32 精度运行

