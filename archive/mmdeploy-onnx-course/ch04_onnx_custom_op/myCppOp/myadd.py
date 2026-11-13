"""把 C++ 扩展 my_lib.my_add 包装成 torch.autograd.Function，并用 symbolic 导出 ONNX。

运行前需要先编译 C++ 扩展（在 ch04_onnx_custom_op/myCppOp 目录下）::

    uv run python setup.py build_ext --inplace

然后从仓库根目录运行::

    uv run python -m ch04_onnx_custom_op.myCppOp.myadd
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

HERE = Path(__file__).resolve().parent
# 编译产物 my_lib.*.so 与脚本在同一目录，先把它加进 sys.path，
# 这样无论从哪个工作目录运行都能 import 到 my_lib
sys.path.insert(0, str(HERE))

try:
    import my_lib
except ImportError as exc:
    raise ImportError(
        "找不到 my_lib，请先编译 C++ 扩展（在 ch04_onnx_custom_op/myCppOp 目录下执行）:\n"
        "    uv run python setup.py build_ext --inplace\n"
        "或按教程执行 `uv run python setup.py develop` 安装到当前环境"
    ) from exc


class MyAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return my_lib.my_add(a, b)

    # 只要 autograd.Function 定义了 symbolic，导出时就会调用它把这个算子映射成 ONNX 节点，
    # 不需要额外设置 autograd_inlining（PyTorch 的 NodeToONNX 会优先使用 symbolic）
    @staticmethod
    def symbolic(g, a, b):
        two = g.op("Constant", value_t=torch.tensor([2]))
        a = g.op("Mul", a, two)
        return g.op("Add", a, b)


# 上面只是把模块封装起来以便于调用
# 下面开始调用

my_add = MyAddFunction.apply


class MyAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return my_add(a, b)


# 接下来测试正确性

output_dir = HERE / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
onnx_path = output_dir / "my_add.onnx"

model = MyAdd()
a = torch.rand(1, 3, 10, 10)
b = torch.rand(1, 3, 10, 10)
torch.onnx.export(
    model,
    (a, b),
    onnx_path,
    # 显式命名，后面 ORT 才知道该喂给哪个输入
    input_names=["a", "b"],
    output_names=["output"],
    opset_version=20,
    # 只有旧版 TorchScript 导出器支持 autograd.Function 的 symbolic
    # (torch 会发一条 DeprecationWarning，属预期现象)
    dynamo=False,
)

# 先确认导出的图里是我们定义的 Mul + Add 节点
onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)
op_types = [node.op_type for node in onnx_model.graph.node]
assert {"Mul", "Add"} <= set(op_types), f"导出图不符合预期: {op_types}"

with torch.no_grad():
    torch_output = model(a, b).numpy()

sess = onnxruntime.InferenceSession(str(onnx_path))
ort_output = sess.run(["output"], {"a": a.numpy(), "b": b.numpy()})[0]

np.testing.assert_allclose(torch_output, ort_output, rtol=1e-5, atol=1e-5)
print(f"导出完成: {onnx_path}，节点: {op_types}，PyTorch 与 ONNX Runtime 输出一致")
