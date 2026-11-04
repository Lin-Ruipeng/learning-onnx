# 第 3 章：PyTorch 转 ONNX 详解

来源：[`03_pytorch2onnx.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/03_pytorch2onnx.md)

## 本章要点

- **两种导出方式**：跟踪法（`torch.jit.trace`，只记录实际走过的分支，带控制流会丢）与脚本化（`torch.jit.script`，能保留 `if/for`）。`torch.onnx.export` 默认走跟踪。
- **`torch.onnx.export` 常用参数**：`opset_version`、`input_names` / `output_names`、`dynamic_axes`（不指定就固化 shape，换 batch size 会报 `INVALID_ARGUMENT`）、`do_constant_folding`。
- **导出期的行为差异**：`torch.onnx.is_in_onnx_export()` 让模型在导出时走不同分支。
- **算子支持查询**：ONNX 官方 `Operators.md` 查算子与 opset，PyTorch 侧看 `torch/onnx/symbolic_opset*.py` 里对应 opset 的 `@_onnx_symbolic("aten::xxx")` 映射。

## 代码与产物

本章重点在原理，用第 1 章的现成产物当案例，暂不需要新脚本：

| 文件 | 说明 |
|---|---|
| `export.py`（第 1 章） | 下载权重与 `face.png` → torch 推理 → `torch.onnx.export`(opset 18) → `onnx.checker` → ORT 推理 |
| `outputs/srcnn.onnx` + `srcnn.onnx.data`（第 1 章） | 待分析的对象（`Resize → Conv → Relu → Conv → Relu → Conv`，6 个节点） |

练习用的脚本计划放在本目录：

```text
ch03_pytorch2onnx/
├── export_dynamic.py      # dynamic_axes 导出 + onnx.shape_inference 验证
└── outputs/
```

## 笔记

<!-- 待补：dynamic_axes 的写法与踩坑、trace 与 script 在同一模型上的差异、opset 18 与 11 的差别 -->
