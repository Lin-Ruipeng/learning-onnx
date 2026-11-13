"""第 5 章：读入已有的 ONNX 模型，查看其 GraphProto 结构，并直接改算子类型。

依赖 ``create_onnx.py`` 先跑出 ``linear_func.onnx``（把 Add 改成 Sub 后另存为 ``linear_func_2.onnx``）。

运行（``-m`` 形式或直接执行本文件都可以）::

    uv run python -m ch05_onnx_model_editing.modificate_onnx
"""

import sys
from pathlib import Path

import onnx

# 直接执行本文件时 sys.path[0] 是脚本所在目录，仓库根不在其中；
# 先把仓库根插进 sys.path，``common`` 才能被 import（用 -m 运行时这行也无害）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import outputs

# 产物目录由 common.paths 统一给出（绝对路径，不存在会自动创建），不依赖当前工作目录
OUT = outputs("ch05_onnx_model_editing")
onnx_path = OUT / "linear_func.onnx"
modified_path = OUT / "linear_func_2.onnx"

# 修改 onnx 模型

# 载入已有的 onnx 模型
if not onnx_path.exists():
    raise FileNotFoundError(f"找不到 {onnx_path}，请先运行 ch05_onnx_model_editing.create_onnx")
model = onnx.load(onnx_path)

# 打印完整结构
print(model)


# 获取 计算图
graph = model.graph
# 获取 节点
gnode = graph.node
# 获取输入
ginput = graph.input
# 获取输出
goutput = graph.output

print("节点Proto: ", gnode)
print("输入Proto: ", ginput)
print("输出Proto: ", goutput)

# 打印一个图节点的输入输出名字

node_0 = gnode[0]
node_0_inputs = node_0.input
node_0_outputs = node_0.output

input_0 = node_0_inputs[0]
input_1 = node_0_inputs[1]

output = node_0_outputs[0]
op_type = node_0.op_type

print("节点输入: ", input_0)
print("节点输入: ", input_1)
print("节点输出: ", output)
print("算子类型: ", op_type)


# 前面能读入 onnx 模型了，后续就非常方便修改了

model = onnx.load(onnx_path)
node = model.graph.node
node[1].op_type = "Sub"  # 直接修改算子类型!

onnx.checker.check_model(model)
onnx.save(model, modified_path)
print(f"修改后的模型已保存: {modified_path}")
