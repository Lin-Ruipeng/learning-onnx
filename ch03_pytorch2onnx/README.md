# 第 3 章：PyTorch 转 ONNX 详解

来源：[`03_pytorch2onnx.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/03_pytorch2onnx.md)

## 本章要点

- **两种导出方式**：跟踪法（`torch.jit.trace`，只记录实际走过的分支，带控制流会丢）与脚本化（`torch.jit.script`，能保留 `if/for`）。`torch.onnx.export` 默认走跟踪。
- **`torch.onnx.export` 常用参数**：`opset_version`、`input_names` / `output_names`、`dynamic_axes`（不指定就固化 shape，换 batch size 会报 `INVALID_ARGUMENT`）、`do_constant_folding`。
- **导出期的行为差异**：`torch.onnx.is_in_onnx_export()` 让模型在导出时走不同分支。
- **算子支持查询**：ONNX 官方 `Operators.md` 查算子与 opset，PyTorch 侧看 `torch/onnx/symbolic_opset*.py` 里对应 opset 的 `@_onnx_symbolic("aten::xxx")` 映射。

## 代码与产物

| 文件 | 说明 |
|---|---|
| `export.py` | 下载权重与 `face.png` → torch 推理 → `torch.onnx.export`(opset 18) → `onnx.checker` → ORT 推理，并打印两条路径的最大像素差 |
| `outputs/srcnn.onnx` + `srcnn.onnx.data` | 导出结果（权重 >2GB 阈值以下，但 torch 仍按外部数据写出 `data` 文件，两者必须同目录） |
| `outputs/face_torch.png` | PyTorch 的 3 倍超分结果 |
| `outputs/face_ort.png` | ONNX Runtime 的同一张图，应与上一张肉眼无差 |

```bash
uv run python -m ch03_pytorch2onnx.export
```

实测（torch 2.15 + onnxruntime 1.31，CPU）：`Model correct`，两条路径输出最大像素差 1/255，肉眼无差。

## 笔记

<!-- 待补：dynamic_axes 的写法与踩坑（导出后如何用 onnx.shape_inference 确认）、opset 18 与 11 的差别 -->
