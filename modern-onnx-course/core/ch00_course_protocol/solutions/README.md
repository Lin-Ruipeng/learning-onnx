# 参考答案

1. `compare_runtime(model_path, 2)` 和 `compare_runtime(model_path, 7)` 应当直接通过；动态 batch 属于同一 artifact 的契约，不应为每个 batch 重新导出。
2. `dim_param="batch"` 表示符号维度；它不是一个具体长度。`torch.export.Dim` 中的 1–8 范围是导出约束，不等于 ONNX Runtime 会自动强制这个上界。
3. 错误的 Golden Reference 应让 `np.testing.assert_allclose` 报告失配数量、最大绝对/相对差异及样例值。
4. 当前 `sample_input(9)` 先按课程声明的 Artifact Contract 抛出 `ValueError`。这是课程层的 fail-closed 输入检查，不能用来推断 ORT 是否会拒绝 batch 9。
