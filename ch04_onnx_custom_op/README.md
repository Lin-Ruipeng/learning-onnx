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
├── op_Asinh.py            # 给 aten 算子注册符号函数（torch.asinh -> ONNX Asinh）+ 数值验证
└── outputs/
    └── asinh.onnx
```

运行方式同样是 `uv run python -m ch04_onnx_custom_op.<脚本名>`（需在仓库根目录执行）。

## 笔记

`op_Asinh.py` 在 torch 2.15 + onnxruntime 1.31 下的实测结论：

- **旧写法不可用**：教程里的 `torch.onnx.symbolic_registry.register_op` 在 PyTorch 2.x 已被删除，import 时会直接 `ModuleNotFoundError`。替代 API 是 `torch.onnx.register_custom_op_symbolic`，算子名必须写成 `aten::asinh`（裸名 `"asinh"` 不行）。
- **只对旧导出器生效**：`register_custom_op_symbolic` 注册进的是 TorchScript 导出器的符号函数表；torch ≥ 2.9 默认 `dynamo=True`，必须显式传 `dynamo=False`，运行时会出现一条 `DeprecationWarning`，属预期现象。
- **这个例子仍然有意义**：torch 2.15 的 TorchScript 导出器只内置了 `asin/acos/atan` 的符号函数，没有 `aten::asinh`，不注册就会报 `UnsupportedOperatorError`。注册后导出图里是单个 `Asinh` 节点，`onnx.checker` 可过。
- **opset 不必等于注册值**：符号函数注册在 opset 9，导出用 opset 20 也能命中 —— 查找规则是“不超过目标 opset 的最大已注册版本”。
- **输入名别再写死 `"0"`**：新版导出器会沿用 traced graph 的输入名。导出时显式传 `input_names=["input"]`，ORT 侧用 `sess.get_inputs()[0].name` 取真实输入名。
- **新导出器路线**（`dynamo=True`）走的是 onnxscript 的分解表，对应写法是 `torch.onnx.export(..., custom_translation_table={...})`，与 `register_custom_op_symbolic` 是两套互不相干的机制。

下面是 mmdeploy 教程原文的笔记，供对照：

要使 PyTorch 算子顺利转换到 ONNX ，我们需要保证以下三个环节都不出错：

- 算子在 PyTorch 中有实现
- 有把该 PyTorch 算子映射成一个或多个 ONNX 算子的方法
- ONNX 有相应的算

在实际部署中，这三部分的内容都可能有所缺失。其中最坏的情况是：我们定义了一个全新的算子，它不仅缺少 PyTorch 实现，还缺少 PyTorch 到 ONNX 的映射关系。但所谓车到山前必有路，对于这三个环节，我们也分别都有以下的添加支持的方法：

- PyTorch 算子
  - 组合现有算子
  - 添加 TorchScript 算子
  - 添加普通 C++ 拓展算子
- 映射方法
  - 为 ATen 算子添加符号函数
  - 为 TorchScript 算子添加符号函数
  - 封装成 torch.autograd.Function 并添加符号函数
- ONNX 算子
  - 使用现有 ONNX 算子
  - 定义新 ONNX 算子

