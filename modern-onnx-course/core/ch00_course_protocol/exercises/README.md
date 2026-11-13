# 练习

1. 先不看参考答案，把默认验证 batch 改成 `2` 和 `7`，确认无需重新导出就能推理。
2. 打印 ONNX 输入的维度信息，解释 `dim_param="batch"` 与固定 `dim_value` 的差异。
3. 故意把 ORT 输出与错误的参考公式比较，观察 `assert_allclose` 的失败信息。
4. 尝试 batch `9`：记录它是在课程输入检查、ONNX Runtime 还是导出约束的哪一层失败。

完成后再对照 [`../solutions/README.md`](../solutions/README.md)。
