# 第 4 章：在 PyTorch 中支持更多 ONNX 算子

来源：[`04_onnx_custom_op.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/04_onnx_custom_op.md)

## 本章要点

要让一个 PyTorch 算子顺利变成 ONNX 里的节点，三个环节都不能掉：

1. **ATen 算子** —— `torch.nn` / `torch.nn.functional` 里的算子（如 `aten::interpolate`）。给某个 opset 注册 symbolic function（`@_onnx_symbolic("aten::xxx")` + `@symbolic_helper.parse_args`），或干脆用 `torch.onnx.register_custom_op_symbolic` 改映射。
2. **TorchScript 算子** —— 自定义的 `@torch.jit.script` 函数，需要注册对应的 symbolic（`@_onnx_symbolic("::my_op")`）。
3. **`torch.autograd.Function`** —— 最通用的兜底手段：自定义 `forward` 里用 ONNX 原生 API（`g.op("...")`）直接拼节点，可以在导出时做任意的图改写。

三种方式的取舍：优先查 PyTorch 是否已支持 → 只是映射不对就用 `register_custom_op_symbolic` → 语义上真的缺算子才写 `autograd.Function`。

## 代码与产物

```text
ch04_onnx_custom_op/
├── aten_symbolic.py       # 给 aten 算子注册/覆盖 symbolic（第 2 章的动态 interpolate）
├── torchscript_op.py      # TorchScript 自定义算子 + symbolic
├── autograd_function.py   # torch.autograd.Function 里用 g.op 拼 ONNX 节点
└── outputs/
```

运行方式同样是 `uv run python -m ch04_onnx_custom_op.<脚本名>`。

## 笔记

<!-- 待补：register_custom_op_symbolic 在 PyTorch 2.x 是否被 onnxscript 取代、各方式实测能否通过 onnx.checker -->
