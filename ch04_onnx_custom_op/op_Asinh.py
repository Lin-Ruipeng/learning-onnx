from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 这里调用的是 pt 的 asinh 算子
        return torch.asinh(x)
        # 在转 onnx 时会自动调用符号函数 转成 Asinh 这个 onnx 算子


# 这里导入的是用于注册算子的函数。
# 旧教程用的 torch.onnx.symbolic_registry.register_op 在 PyTorch 2.x 已被删除
# (import 会直接 ModuleNotFoundError)，现在的等价 API 是 register_custom_op_symbolic
from torch.onnx import register_custom_op_symbolic


# 这里用于创建一个符号函数，.pt -> .onnx 的过程本质上就是调用算子的符号函数
# 把每一个算子依次用符号函数表示成 onnx 算子的形式
# 所以这里创建一个符号函数用于把一个 pt 算子绑定到 Asinh 这个 onnx 算子上
def asinh_symbolic(
    g,  # 第一个参数必须是 g 且 g 必定有一个 op() 方法!
    input,  # 第二个参数是此算子的输入，也可以有第三四五个参数都作为算子输入
    *,
    out=None,
):
    return g.op(
        "Asinh",  # 第一个参数是 onnx 算子的名字
        input,  # 第二个参数开始都是算子的输入，这里是单输入，可以有多输入
    )


# 这里就是把手动创建的符号函数 asinh_symbolic 和 pt 算子 asinh 绑定到一起
# 注意: 算子名必须是 <domain>::<op> 形式，裸名 "asinh" 是不行的
# 该注册只对旧版 TorchScript 导出器生效 (torch>=2.9 默认 dynamo=True，要显式关掉)
register_custom_op_symbolic(
    "aten::asinh",  # pt 算子的名字
    asinh_symbolic,  # 绑定的符号函数
    9,  # 绑定到的 opset 版本 (ONNX 从 opset 9 开始提供 Asinh)
)

# 产物目录以脚本所在目录为基准，避免依赖运行时的当前工作目录
output_dir = Path(__file__).with_suffix("").parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
onnx_path = output_dir / "asinh.onnx"

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(
    model,
    input,
    onnx_path,
    # 显式命名，后面 ORT 推理时才知道该喂给哪个输入
    input_names=["input"],
    output_names=["output"],
    opset_version=20,
    # 符号函数注册只对旧版 TorchScript 导出器生效，因此这里必须显式关闭 dynamo
    # (torch 会发一条 DeprecationWarning，属预期现象)
    dynamo=False,
)

# 上面都是注册算子和转换 (这里的注册算子是: 其实已经有 onnx 算子的实现，只是没有绑定到 pt 算子上)
# 下面是对算子的正确性的测试

# 先确认导出的图里真的是 Asinh 节点，而不是别的兜底形式
onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)
op_types = [node.op_type for node in onnx_model.graph.node]
assert "Asinh" in op_types, f"导出的图里没有 Asinh 节点，实际节点: {op_types}"
print(f"导出完成: {onnx_path}，节点: {op_types}")


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)


model = Model()
input = torch.rand(1, 3, 10, 10)
with torch.no_grad():  # 这里用 pt 模型推理得到参考结果
    torch_output = model(input).numpy()

# 再用 onnx 模型跑一个结果
sess = onnxruntime.InferenceSession(str(onnx_path))
# 输入名不再是旧教程里的 "0": torch 2.x 会沿用 traced graph 的输入名，
# 直接从模型里取，避免写死
input_name = sess.get_inputs()[0].name
ort_output = sess.run(["output"], {input_name: input.numpy()})[0]

# 断言对比
np.testing.assert_allclose(torch_output, ort_output, rtol=1e-5, atol=1e-5)
print("PyTorch 与 ONNX Runtime 输出一致")
