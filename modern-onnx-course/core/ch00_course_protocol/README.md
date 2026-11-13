# Chapter 00：课程协议与环境验证

> 最后验证：2026-11-13
> 验证路径：PyTorch → ONNX → ONNX checker → ONNX Runtime CPU

本章建立整套课程的最小验收协议：环境必须能从锁文件重建，导出产物必须能被结构检查，动态 batch 必须真正由 runtime 执行，数值结果必须和 PyTorch 的 Golden Reference 一致。

## 1. 准备环境

从 `modern-onnx-course/` 目录运行：

```bash
uv sync --locked
uv run python core/ch00_course_protocol/src/verify_environment.py
```

默认产物写入 `artifacts/generated/ch00_dynamic.onnx`。这个目录被 Git 忽略，因为产物可以由脚本重建。

## 2. 这个闭环验证了什么

1. 使用默认的 `dynamo=True` 路径调用现代 `torch.onnx.export` 导出器。
2. 用 `dynamic_shapes` 声明取值范围为 1–8 的动态 batch，而不是 legacy `dynamic_axes`。
3. 用 `onnx.checker.check_model(..., full_check=True)` 检查产物。
4. 明确指定 `CPUExecutionProvider`，并分别用 batch 1、4、8 执行。
5. 比较 ORT 输出和 PyTorch Golden Reference，拒绝“能加载就算成功”。

## 3. 自动验收

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest core/ch00_course_protocol
```

测试会使用 Hypothesis 生成 1–8 之间的 batch size，验证同一 ONNX artifact 不只对导出时的样例输入有效。

## 4. 完成清单

- [ ] 我能解释为什么“导出成功”不等于“部署正确”。
- [ ] 我能在没有手工激活 `.venv` 的情况下运行所有命令。
- [ ] 我能在 Netron 或 ONNX API 中找到动态 `batch` 维。
- [ ] 我能说出 checker、runtime 执行和数值差分各自能发现哪类问题。

## 官方资料

- [PyTorch ONNX](https://docs.pytorch.org/docs/stable/onnx)
- [torch.export-based ONNX Exporter](https://docs.pytorch.org/docs/stable/onnx_export.html)
- [ONNX checker API](https://onnx.ai/onnx/api/checker.html)
- [ONNX Runtime Python quickstart](https://onnxruntime.ai/docs/get-started/with-python.html)
