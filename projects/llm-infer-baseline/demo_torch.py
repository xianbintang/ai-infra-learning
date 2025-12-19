"""演示 torch 和 PyTorch 的关系。"""
from __future__ import annotations

import torch


def demo_torch_import() -> None:
    """演示 torch 的导入和使用。"""
    print("=" * 60)
    print("🔍 演示 1: torch 导入和基本信息")
    print("=" * 60)
    
    # 导入 torch（注意：不是 import pytorch）
    print("\n1. 导入 torch:")
    print("   import torch  # ✅ 正确")
    print("   # import pytorch  # ❌ 错误，不存在")
    
    # 检查版本
    print(f"\n2. PyTorch 版本: {torch.__version__}")
    print(f"   注意：虽然框架叫 PyTorch，但属性是 torch.__version__")
    
    # 检查包信息
    print(f"\n3. torch 包信息:")
    print(f"   包名: torch")
    print(f"   框架名: PyTorch")
    print(f"   安装路径: {torch.__file__}")


def demo_torch_usage() -> None:
    """演示 torch 的实际使用。"""
    print("\n" + "=" * 60)
    print("💻 演示 2: torch 的实际使用")
    print("=" * 60)
    
    # 创建张量
    print("\n1. 创建张量:")
    x = torch.tensor([1, 2, 3])
    print(f"   x = torch.tensor([1, 2, 3])")
    print(f"   结果: {x}")
    print(f"   注意：使用 torch.tensor，不是 pytorch.tensor")
    
    # 检查设备
    print("\n2. 检查 CUDA:")
    cuda_available = torch.cuda.is_available()
    print(f"   torch.cuda.is_available() = {cuda_available}")
    if cuda_available:
        print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"   注意：使用 torch.cuda，不是 pytorch.cuda")
    
    # 数据类型
    print("\n3. 数据类型:")
    print(f"   torch.float32 = {torch.float32}")
    print(f"   torch.float16 = {torch.float16}")
    print(f"   注意：使用 torch.float32，不是 pytorch.float32")


def demo_torch_vs_pytorch() -> None:
    """演示 torch 和 PyTorch 的关系。"""
    print("\n" + "=" * 60)
    print("📚 演示 3: torch 和 PyTorch 的关系")
    print("=" * 60)
    
    print("\n关系总结:")
    print("  PyTorch（框架名称）")
    print("    ↓")
    print("  torch（Python 包名）")
    print("    ↓")
    print("  import torch（代码中使用）")
    
    print("\n命名对比:")
    print("  ┌─────────────┬──────────────┬─────────────┐")
    print("  │ 场景        │ 使用名称     │ 示例        │")
    print("  ├─────────────┼──────────────┼─────────────┤")
    print("  │ 框架名称    │ PyTorch      │ 文档、讨论  │")
    print("  │ 包名        │ torch        │ pip install  │")
    print("  │ 导入        │ torch        │ import torch │")
    print("  │ 代码中使用  │ torch        │ torch.tensor │")
    print("  └─────────────┴──────────────┴─────────────┘")
    
    print("\n安装和导入:")
    print("  ✅ pip install torch")
    print("  ✅ import torch")
    print("  ❌ pip install pytorch  # 错误")
    print("  ❌ import pytorch       # 错误")


def demo_common_mistakes() -> None:
    """演示常见错误。"""
    print("\n" + "=" * 60)
    print("⚠️  演示 4: 常见错误")
    print("=" * 60)
    
    print("\n错误 1: 尝试导入 pytorch")
    print("  ❌ import pytorch")
    print("     NameError: name 'pytorch' is not defined")
    print("  ✅ import torch")
    
    print("\n错误 2: 尝试安装 pytorch 包")
    print("  ❌ pip install pytorch")
    print("     可能失败或安装错误的包")
    print("  ✅ pip install torch")
    
    print("\n错误 3: 在代码中使用 pytorch")
    print("  ❌ pytorch.tensor([1, 2, 3])")
    print("  ✅ torch.tensor([1, 2, 3])")
    
    print("\n正确用法:")
    print("  ✅ 安装: pip install torch")
    print("  ✅ 导入: import torch")
    print("  ✅ 使用: torch.tensor(), torch.cuda.is_available() 等")


def demo_torch_ecosystem() -> None:
    """演示 PyTorch 生态系统。"""
    print("\n" + "=" * 60)
    print("🌐 演示 5: PyTorch 生态系统")
    print("=" * 60)
    
    print("\nPyTorch 生态系统包含:")
    print("  1. torch - 核心框架（我们使用的）")
    print("  2. torchvision - 计算机视觉工具")
    print("  3. torchaudio - 音频处理工具")
    print("  4. torchtext - 文本处理工具")
    
    print("\n安装:")
    print("  pip install torch          # 核心框架")
    print("  pip install torchvision     # 需要单独安装")
    print("  pip install torchaudio      # 需要单独安装")
    
    print("\n导入:")
    print("  import torch")
    print("  import torchvision  # 如果安装了")
    print("  import torchaudio   # 如果安装了")
    
    print("\n注意:")
    print("  - 所有包名都使用 'torch' 前缀")
    print("  - 但框架整体叫 'PyTorch'")
    print("  - 这是命名约定，不是错误")


def main() -> None:
    """主函数。"""
    print("=" * 60)
    print("🔬 torch 和 PyTorch 关系演示")
    print("=" * 60)
    
    # 演示 1: 基本导入
    demo_torch_import()
    
    # 演示 2: 实际使用
    demo_torch_usage()
    
    # 演示 3: 关系说明
    demo_torch_vs_pytorch()
    
    # 演示 4: 常见错误
    demo_common_mistakes()
    
    # 演示 5: 生态系统
    demo_torch_ecosystem()
    
    print("\n" + "=" * 60)
    print("✅ 演示完成")
    print("=" * 60)
    print("\n💡 关键要点:")
    print("   1. PyTorch 是框架名称（品牌名）")
    print("   2. torch 是 Python 包名（代码中使用）")
    print("   3. 安装和导入都使用 torch")
    print("   4. 不要使用 pytorch 作为包名")
    print("   5. 代码中始终使用 torch，文档中可以说 PyTorch")


if __name__ == "__main__":
    main()


