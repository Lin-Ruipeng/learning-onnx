# 第 3 章：PyTorch 转 ONNX 详解

来源：[`03_pytorch2onnx.md`](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/03_pytorch2onnx.md)

## 本章要点

- **两种导出方式**：跟踪法（`torch.jit.trace`，只记录实际走过的分支，带控制流会丢）与脚本化（`torch.jit.script`，能保留 `if/for`）。`torch.onnx.export` 默认走跟踪。
- **`torch.onnx.export` 常用参数**：`opset_version`、`input_names` / `output_names`、`dynamic_axes`（不指定就固化 shape，换 batch size 会报 `INVALID_ARGUMENT`）、`do_constant_folding`。
- **导出期的行为差异**：`torch.onnx.is_in_onnx_export()` 让模型在导出时走不同分支；`.item()`、遍历张量这类操作会中断张量跟踪、把中间结果固化成常量（是坑，反过来也可用于静态化）。
- **算子支持查询**：ONNX 官方 `Operators.md` 查算子与 opset，PyTorch 侧看 `torch/onnx/symbolic_opset*.py` 里对应 opset 的 `@_onnx_symbolic("aten::xxx")` 映射。

## 代码与产物

分析对象沿用第 1 章的现成产物，另配三个小脚本做对照实验：

| 文件 | 说明 |
|---|---|
| `export.py`（第 1 章） | 下载权重与 `face.png` → torch 推理 → `torch.onnx.export`(opset 18) → `onnx.checker` → ORT 推理 |
| `outputs/srcnn.onnx` + `srcnn.onnx.data`（第 1 章） | 待分析的对象（`Resize → Conv → Relu → Conv → Relu → Conv`，6 个节点） |
| `trace_and_script.py` | 带 `for` 循环的模型（n=2/3）分别 trace / script 导出，对比控制流被展开成固定图还是保留为 `Loop` 节点 |
| `dynamic_axes.py` | 同一模型导出静态 / 第 0 维动态 / 第 2、3 维动态三份，再用 ORT 喂三种形状验证 |
| `export_tips.py`（原 `condition_infer.py`） | 两个导出技巧：`is_in_onnx_export` 条件推理、`.item()` 等中断张量跟踪；纯阅读示例，无 `__main__` |
| `outputs/` | `model_{2,3}_{trace,script}.onnx` 4 个 + `model_static.onnx`、`model_dynamic_0.onnx`、`model_dynamic_23.onnx` 3 个 |

```bash
# 从仓库根目录运行
uv run python -m ch03_pytorch2onnx.trace_and_script
uv run python -m ch03_pytorch2onnx.dynamic_axes
```

## 笔记

<!-- 待补：dynamic_axes 的写法与踩坑、trace 与 script 在同一模型上的差异、opset 18 与 11 的差别 -->


在这个翻译(.pt -> .onnx)过程中，可能会碰到以下情况：
- 该算子可以一对一地翻译成一个 ONNX 算子。
- 该算子在 ONNX 中没有直接对应的算子，会翻译成一至多个 ONNX 算子。
- 该算子没有定义翻译成 ONNX 的规则，报错。

在 PyTorch 中，和 ONNX 有关的定义全部放在 torch.onnx 目录中，
我们可以按照代码的调用逻辑，逐步跳转直到最底层的 ONNX 映射函数

比如说在本项目中，可以有这样：
1. 打开 `.venv/lib/python3.12/site-packages/torch/onnx` 这个路径
2. 算子版本11就是这个文件： `.venv/lib/python3.12/site-packages/torch/onnx/symbolic_opset11.py`
3. 从这个文件的 form import 语句进行跳转就可以打开文件： `.venv/lib/python3.12/site-packages/torch/onnx/_internal/torchscript_exporter/symbolic_opset11.py` 
4. 补充，如果你需要找到指定的算子，请在 `torch/onnx` 这个文件夹下直接搜索你要找的算子名称比如说 `bicubic` 
5. 这里举例的 `bicubic` 这个算子就是在版本11引入的，所以可以在上面打开的这个库文件里搜索到这个算子
6. 

- 算子注册源码
```python
@_onnx_symbolic(  # 这是一个装饰器
    "aten::upsample_bicubic2d",  # 第一个属性是 作用域::算子名字 也就是注册一个算子
    decorate=[symbolic_helper._apply_params("upsample_bicubic2d", 4, "cubic")],
)  # 第二个属性是 先执行 _apply(_interpolate)，得到绑定参数的具体函数
#  下面这个是函数是一个工厂函数 调用上面说的那个注册算子的时候其实是这个函数干活 返回一个实例
def _interpolate(name: str, dim: int, interpolate_mode: str):
    # 返回 回去的这个函数 才是真正干活的函数算子! 点击进去看源码!
    return symbolic_helper._interpolate_helper(name, dim, interpolate_mode)
```

- 算子实现源码
```python
# 这里就是上面转跳过来的函数，真正返回的内容又是一个 g.op() 也就是这个才是算子
def _interpolate_helper(name, dim, interpolate_mode):
    @quantized_args(True, False, False)
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = _get_interpolate_attributes(g, interpolate_mode, args)
        align_corners = _maybe_get_scalar(align_corners)
        coordinate_transformation_mode = (
            "asymmetric"
            if interpolate_mode == "nearest"
            else "align_corners"
            if align_corners
            else "half_pixel"
        )

        if scales is None:
            input_size = g.op("Shape", input)
            input_size_beg = _slice_helper(
                g, input_size, axes=[0], ends=[2], starts=[0]
            )
            output_size = g.op(
                "Cast", output_size, to_i=_C_onnx.TensorProtoDataType.INT64
            )
            output_size = g.op("Concat", input_size_beg, output_size, axis_i=0)

            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
                empty_scales = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )
                empty_scales = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )

            return g.op(
                "Resize",
                input,
                empty_roi,
                empty_scales,
                output_size,
                coordinate_transformation_mode_s=coordinate_transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s="floor",
            )  # only valid when mode="nearest"
        else:
            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )

            # 比如说这里就是返回了一个 Resize 算子, 所以在 Netron.app 上看到的算子是 Resize
            return g.op(
                "Resize",
                input,
                empty_roi,
                scales,
                coordinate_transformation_mode_s=coordinate_transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s="floor",
            )  # only valid when mode="nearest"

    return symbolic_fn
```