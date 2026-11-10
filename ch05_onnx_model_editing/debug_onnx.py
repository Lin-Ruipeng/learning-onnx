"""第 5 章：用 ``onnx.utils.extract_model`` 从大模型里切出子图，方便单独调试中间结果。

先导出一个 9 层卷积 + 1 个 Add 的模型（用旧版 TorchScript 导出器，与教程一致），
再按边界张量名切出若干子模型。教程原文用的是 "22"/"28" 这类导出器内部编号，
但不同 torch 版本编号会变，所以这里按模型结构定位等价的张量名。

运行（``-m`` 形式或直接执行本文件都可以）::

    uv run python -m ch05_onnx_model_editing.debug_onnx
"""

import sys
from pathlib import Path

import onnx
import torch

# 直接执行本文件时 sys.path[0] 是脚本所在目录，仓库根不在其中；
# 先把仓库根插进 sys.path，``common`` 才能被 import（用 -m 运行时这行也无害）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import outputs

# 产物目录由 common.paths 统一给出（绝对路径，不存在会自动创建），不依赖当前工作目录
OUT = outputs("ch05_onnx_model_editing")
whole_path = OUT / "whole_model.onnx"
partial_path = OUT / "partial_model.onnx"
sub1_path = OUT / "submodel_1.onnx"
sub2_path = OUT / "submodel_2.onnx"


# 1. 构建一个大的完整的模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3),
            torch.nn.Conv2d(3, 3, 3),
            torch.nn.Conv2d(3, 3, 3),
        )
        self.convs2 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3),
            torch.nn.Conv2d(3, 3, 3),
        )
        self.convs3 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3),
            torch.nn.Conv2d(3, 3, 3),
        )
        self.convs4 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3),
            torch.nn.Conv2d(3, 3, 3),
        )

    def forward(self, x):
        x = self.convs1(x)
        x1 = self.convs2(x)
        x2 = self.convs3(x)
        x = x1 + x2
        x = self.convs4(x)
        return x


model = Model()

dummy_input = torch.randn(1, 3, 20, 20)
torch.onnx.export(
    model,
    dummy_input,
    whole_path,
    # torch>=2.9 默认走 dynamo 导出器，中间张量会被命名成 x / conv1/Conv_output_0 之类；
    # 后面按 "22"/"28" 这类 TorchScript 编号切子图就找不到了，所以显式使用旧版导出器
    dynamo=False,
)
print(f"已导出: {whole_path}")


# 2. 用 onnx 来提取出一个子模型

# 教程原文按 "22"/"28" 这类导出器内部编号切子图，但新版本 torch 导出的名字是
# /convs1/convs1.2/Conv_output_0 这种形式，编号未必存在。
# 这里按模型结构取等价的边界张量：卷积按出现顺序编号，Add 是分支汇合点。
graph_proto = onnx.load(whole_path).graph
conv_outputs = [node.output[0] for node in graph_proto.node if node.op_type == "Conv"]
add_output = next(node.output[0] for node in graph_proto.node if node.op_type == "Add")

branch_start = conv_outputs[2]             # convs1 的最后一个卷积 (教程里的 "22")
branch_a_end = conv_outputs[4]             # convs2 的最后一个卷积 (教程里的 "25")
branch_b_end = conv_outputs[6]             # convs3 的最后一个卷积 (教程里的 "27")
merged = add_output                        # 两支相加 (教程里的 "28")
model_output = graph_proto.output[0].name  # 整个模型的输出 (教程里的 "31")
print("边界张量:", branch_start, merged, model_output)

# 提取出从边22到边28的子图, 最好看可视化结果
onnx.utils.extract_model(whole_path, partial_path, [branch_start], [merged])
print(f"已提取: {partial_path}")


# 3. 添加额外输出
onnx.utils.extract_model(whole_path, sub1_path, [branch_start], [branch_b_end, model_output])
print(f"已提取: {sub1_path}")

# 4. 添加冗余输入（input.1 是模型输入名，它不在子图里，提取后会成为悬空输入）
onnx.utils.extract_model(whole_path, sub2_path, [branch_start, "input.1"], [merged])
print(f"已提取: {sub2_path}")

# 5. 可以用这样方式查看所有的中间输出（取消注释即可，路径同样基于 OUT）
# onnx.utils.extract_model(
#     whole_path,
#     OUT / "more_output_model.onnx",
#     ["input.1"],
#     [model_output, conv_outputs[3], branch_a_end, branch_b_end],
# )
# onnx.utils.extract_model(whole_path, OUT / "debug_model_1.onnx", ["input.1"], [conv_outputs[3]])
# onnx.utils.extract_model(whole_path, OUT / "debug_model_2.onnx", [conv_outputs[3]], [branch_a_end])
# onnx.utils.extract_model(whole_path, OUT / "debug_model_3.onnx", [conv_outputs[3]], [branch_b_end])
# onnx.utils.extract_model(
#     whole_path,
#     OUT / "debug_model_4.onnx",
#     [branch_a_end, branch_b_end],
#     [model_output],
# )

