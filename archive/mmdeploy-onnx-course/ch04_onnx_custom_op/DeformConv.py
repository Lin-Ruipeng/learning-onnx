from pathlib import Path

import onnx
import torch
import torchvision

# 旧教程的 torch.onnx.symbolic_registry.register_op 在 PyTorch 2.x 已删除，
# 现在统一用 register_custom_op_symbolic；parse_args 仍由 symbolic_helper 提供
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, 3)
        # 这是 torchvision 的 TorchScript 算子，ONNX 没有对应的官方算子，
        # 需要先用多个 ONNX 组合算子实现（或在推理引擎侧自定义），再注册符号函数绑定。
        # 18 = 2 * offset_groups(1) * kernel_h(3) * kernel_w(3)，正好是 offset 的通道数
        self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)

    def forward(self, x):
        # DeformConv2d.forward(input, offset)：offset 由 conv1 生成
        return self.conv2(x, self.conv1(x))


# 手搓 ONNX 算子，这里是为了给 DeformConv2d 算子实现，所以接口定义直接抄这个算子的 forward() 定义
# 参数顺序来自 torchvision::deform_conv2d 的 schema：
# (input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w,
#  dil_h, dil_w, n_weight_grps, n_offset_grps, use_mask)
@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none")
def symbolic(
    g,
    input,
    weight,
    offset,
    mask,
    bias,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    n_weight_grps,
    n_offset_grps,
    use_mask,
):
    # ONNX 没有官方 deform_conv2d 算子，这里用自定义域 "custom" 造一个占位节点。
    # 真正要用的话，需要在推理引擎里实现这个算子，或者在这里用多个 ONNX 算子拼出来。
    return g.op("custom::deform_conv2d", input, offset)


# 注册（和 asinh 一样：只对旧版 TorchScript 导出器生效）
register_custom_op_symbolic("torchvision::deform_conv2d", symbolic, 9)

# 产物目录以脚本所在目录为基准，避免依赖运行时的当前工作目录
output_dir = Path(__file__).with_suffix("").parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
onnx_path = output_dir / "dcn.onnx"

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(
    model,
    input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=20,
    # 符号函数注册只对旧版 TorchScript 导出器生效，因此必须显式关闭 dynamo
    # (torch 会发一条 DeprecationWarning，属预期现象)
    dynamo=False,
)

# 结构校验：确认导出的图里真的有 custom::deform_conv2d 节点
onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)
custom_nodes = [node for node in onnx_model.graph.node if node.domain == "custom"]
assert custom_nodes, "导出的图里没有 custom 域的节点"
for node in custom_nodes:
    assert node.op_type == "deform_conv2d", node.op_type
    assert len(node.input) == 2, node.input
    print(
        f"导出完成: {onnx_path}，自定义节点: "
        f"{node.domain}::{node.op_type}，输入数: {len(node.input)}"
    )

# 这里不能像 asinh 那样用 ONNX Runtime 做数值对比：
# custom::deform_conv2d 是自定义算子，ORT 没有它的实现，必须先由推理引擎支持。
# 本章只验证「能把算子导出成 ONNX 节点」这一点。
