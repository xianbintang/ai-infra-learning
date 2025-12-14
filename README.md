# AI Infra · 推理优化学习工程

面向已经掌握基础模型训练/调参与 Transformer 结构的工程师，构建一个 1–3 天可落地、可观测的推理优化学习工程。核心目标是亲手搭建“性能基准台”，用实测数据理解推理慢在哪里、怎样权衡 latency / throughput、batching、KV Cache 与调度等关键概念。

## 仓库结构

```
.
├── README.md                  # 仓库总览与导航
├── docs/
│   └── ai-infra-roadmap.md    # 系统学习大纲 + 进度跟踪
└── projects/
    └── llm-infer-baseline/
        └── README.md          # Day1-3 实战项目计划、Checklist
```

## 如何使用本仓库

1. 先阅读 `docs/ai-infra-roadmap.md`，了解完整阶段性学习路线、指标与里程碑。
2. 按照 `projects/llm-infer-baseline/README.md` 的 Day1 → Day3 任务推进，确保记录指标与问题。
3. 每次实验后在 Checklist 中勾选完成情况、更新观察记录，形成自己的性能数据集。
4. 若要动手实现代码，可在 `projects/llm-infer-baseline/` 目录中创建对应的 Python 模块（例如 `infer.py` 等），并与文档约定保持一致。

## 推荐工作流

- **你负责执行：** 运行脚本、采集日志、更新文档；
- **文档负责指导：** 告诉你每一步该观察什么、如何判断瓶颈；
- **持续记录：** 让每次优化都有可复用的基准数据，后续可叠加更激进的优化（如 TensorRT、vLLM、Flash-Attention 等）。

## 下一步

- 进入 `docs/ai-infra-roadmap.md`，确认整体路线；
- 依照路线执行 `projects/llm-infer-baseline/README.md` 中的 Day1 任务，从最小推理基线出发。
