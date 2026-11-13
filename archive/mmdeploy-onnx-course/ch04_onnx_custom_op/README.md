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
├── DeformConv.py          # 给 torchvision 算子注册符号函数（custom::deform_conv2d）+ 结构验证
├── myCppOp/
│   ├── my_add.cpp         # 自定义 C++ 算子（2a+b），用 pybind11 绑定成 my_lib
│   ├── setup.py           # 编译 my_lib 扩展
│   ├── myadd.py           # autograd.Function 封装 + symbolic（Mul + Add）+ 数值验证
│   └── outputs/
│       └── my_add.onnx
└── outputs/
    ├── asinh.onnx
    └── dcn.onnx
```

运行方式同样是 `uv run python -m ch04_onnx_custom_op.<脚本名>`（需在仓库根目录执行）。

`myCppOp` 需要先编译 C++ 扩展（**必须在 `myCppOp` 目录下执行**，产物 `my_lib.*.so` 会落在该目录）：

```bash
cd ch04_onnx_custom_op/myCppOp
uv run python setup.py build_ext --inplace
# 然后回仓库根目录
uv run python -m ch04_onnx_custom_op.myCppOp.myadd
```

## 笔记

`op_Asinh.py` 在 torch 2.15 + onnxruntime 1.31 下的实测结论：

- **旧写法不可用**：教程里的 `torch.onnx.symbolic_registry.register_op` 在 PyTorch 2.x 已被删除，import 时会直接 `ModuleNotFoundError`。替代 API 是 `torch.onnx.register_custom_op_symbolic`，算子名必须写成 `aten::asinh`（裸名 `"asinh"` 不行）。
- **只对旧导出器生效**：`register_custom_op_symbolic` 注册进的是 TorchScript 导出器的符号函数表；torch ≥ 2.9 默认 `dynamo=True`，必须显式传 `dynamo=False`，运行时会出现一条 `DeprecationWarning`，属预期现象。
- **这个例子仍然有意义**：torch 2.15 的 TorchScript 导出器只内置了 `asin/acos/atan` 的符号函数，没有 `aten::asinh`，不注册就会报 `UnsupportedOperatorError`。注册后导出图里是单个 `Asinh` 节点，`onnx.checker` 可过。
- **opset 不必等于注册值**：符号函数注册在 opset 9，导出用 opset 20 也能命中 —— 查找规则是“不超过目标 opset 的最大已注册版本”。
- **输入名别再写死 `"0"`**：新版导出器会沿用 traced graph 的输入名。导出时显式传 `input_names=["input"]`，ORT 侧用 `sess.get_inputs()[0].name` 取真实输入名。
- **新导出器路线**（`dynamo=True`）走的是 onnxscript 的分解表，对应写法是 `torch.onnx.export(..., custom_translation_table={...})`，与 `register_custom_op_symbolic` 是两套互不相干的机制。

`DeformConv.py` 的实测结论：

- **自定义域算子**：ONNX 官方没有 `deform_conv2d`，符号函数里用 `g.op("custom::deform_conv2d", input, offset)` 造了一个占位节点。导出器会自动给 `custom` 域补上 `opset_import`（version 1），所以 `onnx.checker` 能过。
- **`parse_args` 仍可用**：`from torch.onnx.symbolic_helper import parse_args`（顶层是兼容 shim，实现在 `_internal/torchscript_exporter/symbolic_helper.py`）。14 个描述符与 torchvision 0.29.2 的 `deform_conv2d(input, weight, offset, mask, bias, ...8 个 int..., use_mask)` 顺序完全一致。
- **offset 从哪来**：`DeformConv2d(3, 3, 3)` 需要 `2 * offset_groups * kh * kw = 18` 通道的 offset，所以 `forward` 必须是 `self.conv2(x, self.conv1(x))`（conv1 输出 18 通道），写成 `self.conv2(x, self.conv2(x))` 会因通道数不匹配/缺参而失败。
- **不能做数值验证**：`custom::deform_conv2d` 在 ONNX Runtime 里没有实现，直接推理会报找不到算子；本章只做结构校验（节点 domain / op_type / 输入数）。
- **预期警告**：导出时会打印 `The shape inference of custom::deform_conv2d type is missing`，因为符号函数没有用 `setType` 标注输出形状；自定义算子出现这条警告属正常现象。

`myCppOp/myadd.py` 的实测结论：

- **必须先编译扩展**：`setup.py build_ext --inplace` 要在 `myCppOp` 目录下执行（产物 `my_lib.*.so` 落在当前目录）。在仓库根目录执行 `setup.py develop` 会因 setuptools 的 flat-layout 自动发现（根目录下有多个顶层目录）而失败，这也是教程原命令在新环境下跑不通的原因。
- **`symbolic` 优先于 inlining**：PyTorch 的 `NodeToONNX` 只有在 autograd.Function **没有** `symbolic` 时才走 autograd inlining，所以这里定义了 `symbolic` 就会生效，不需要 `autograd_inlining=False`。
- **`dynamo=False` 必须显式传**：新的 `torch.export` 导出器不支持 `torch.autograd.Function`。
- **常量类型不用手动 Cast**：`torch.tensor([2])` 是 int64，导出时 `_jit_pass_onnx_scalar_type_analysis` 会把它转成与输入一致的浮点类型，`Mul` 不会类型不匹配。
- **实测产物**：导出图节点为 `Constant + Mul + Add`，ORT 与 PyTorch 数值一致；`sys.path` 里插入脚本目录，保证 `.so` 在任意 cwd 下都能 import。

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

---

为算子添加符号函数一般要经过以下几步：
1. 获取原算子的前向推理接口。(就是pytorch算子)
2. 获取目标 ONNX 算子的定义。(就是onnx算子)
3. 编写符号函数并绑定。(自己手动注册把他俩绑定到一起)


