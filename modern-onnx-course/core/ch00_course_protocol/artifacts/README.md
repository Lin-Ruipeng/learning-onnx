# Generated Artifacts

`generated/ch00_dynamic.onnx` 由以下命令生成：

```bash
uv run python core/ch00_course_protocol/src/verify_environment.py
```

该产物使用 ONNX opset 18，输入名为 `features`，输出名为 `transformed`，第一维是符号 batch。产物可重建，因此不提交到 Git。
