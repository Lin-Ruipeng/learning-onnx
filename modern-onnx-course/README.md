# Modern ONNX Course

这是一套面向有 PyTorch 基础、希望深入理解模型部署的实验课程。它不把“生成了 `.onnx`”当作终点，而是要求 artifact 契约、数值正确性、运行时行为和性能结论都有可重现的证据。

## 使用方式

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

不需要手工执行 `source .venv/bin/activate`。所有命令从本目录运行，并通过 `uv run` 进入锁定环境。

## Core Course

| 章节 | 主题 | 状态 |
|---|---|---|
| [00](core/ch00_course_protocol/README.md) | 课程协议、环境验证与最小现代 ONNX 闭环 | 已完成 |
| 01 | 部署系统分层与 Artifact Contract | 计划中 |
| 02 | `torch.export` 与 graph capture | 计划中 |
| 03 | 现代 `torch.onnx` 导出器 | 计划中 |
| 04 | Dynamic Shapes 与控制流 | 计划中 |
| 05 | 导出报告、失败分类与最小复现 | 计划中 |
| 06 | ONNX IR、opset、checker、shape inference 与 external data | 计划中 |
| 07 | 数值等价、容差和 property-based 输入矩阵 | 计划中 |
| 08 | ONNX Runtime、Execution Provider 与 fallback | 计划中 |
| 09 | 图优化、profiling 与 benchmark | 计划中 |
| 10 | FP16、INT8 PTQ 与量化实验 | 计划中 |
| 11 | Python 部署、持久 session 与 artifact packaging | 计划中 |
| 12 | ONNX Runtime C++、CMake 与接口契约 | 计划中 |
| 13 | SRCNN Migration Bridge 与核心课综合验收 | 计划中 |

## 验收规则

每章都应同时具有中文讲义、可运行实验、练习、参考答案、自动测试、Teaching Fixtures 和 artifact 说明。测试通过只证明实验结果符合契约；学习者仍应完成章节清单并能解释结果。

## 版本策略

- 当前 Verified Baseline 由 `uv.lock` 确定。
- 每章记录最后验证日期和官方一手资料。
- 导出器、opset、runtime 或后端版本升级时，必须重跑数值、结构和性能验证。
