"""第 5 章：不依赖 PyTorch，纯用 onnx.helper 拼一个 ONNX 模型 (output = a*x + b)。

运行（``-m`` 形式或直接执行本文件都可以）::

    uv run python -m ch05_onnx_model_editing.create_onnx

产物落在 ``ch05_onnx_model_editing/outputs/linear_func.onnx``。
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import helper
from onnx import TensorProto

# 直接执行本文件时 sys.path[0] 是脚本所在目录，仓库根不在其中；
# 先把仓库根插进 sys.path，``common`` 才能被 import（用 -m 运行时这行也无害）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import outputs

# 产物目录由 common.paths 统一给出（绝对路径，不存在会自动创建），不依赖当前工作目录
OUT = outputs("ch05_onnx_model_editing")
onnx_path = OUT / "linear_func.onnx"

# 纯粹用 onnx API 来创建一个 onnx 模型 (output=a*x+b)

# 1. (边) 先创建数据节点 (更加准确的是，把数据理解成计算图的边!)
a = helper.make_tensor_value_info(
    "a",
    TensorProto.FLOAT,
    [10, 10],
)
b = helper.make_tensor_value_info(
    "b",
    TensorProto.FLOAT,
    [10, 10],
)
x = helper.make_tensor_value_info(
    "x",
    TensorProto.FLOAT,
    [10, 10],
)
output = helper.make_tensor_value_info(
    "output",
    TensorProto.FLOAT,
    [10, 10],
)

# 2. (子图) 用算子节点连接数据节点构成子图 (小的计算图)
# 这里的 c 是中间节点，名字相同就默认是相连的
mul = helper.make_node(
    "Mul",  # 算子节点
    ["a", "x"],  # 算子输入
    ["c"],  # 算子输出
)  # c = a*x
add = helper.make_node(
    "Add",
    ["c", "b"],
    ["output"],
)  # output = c+b = a*x+b

# 3. (图) 包装成一个完整的图
graph = helper.make_graph(
    [mul, add],  # 完整的图包含了哪些子图
    "linear_fun",  # 完整的图的名字
    [a, x, b],  # 输入
    [output],  # 输出
)

# 4. 创建模型，检查并导出

# 创建一个实例
# 显式指定 opset 18（和第 1 章导出用的 opset 一致），
# 并用 make_model_gen_version 按 opset 推导出最低可用的 IR 版本（opset 18 -> IR 8）。
# 如果直接用 helper.make_model(graph)，ir_version 会跟着当前 onnx 取最新值（如 IR 14），
# onnxruntime 未必认识，加载时会报 "Unsupported model IR version"。
model = helper.make_model_gen_version(
    graph,
    opset_imports=[helper.make_opsetid("", 18)],
)
# 检查器，检查我们手搓的这个图是否符合onnx模型协议 (非必须步骤)
onnx.checker.check_model(model)

# 打印出这个 protobuf 文件内容
print(model)
onnx.save(model, onnx_path)
print(f"导出完成: {onnx_path}")


# 运行!
sess = onnxruntime.InferenceSession(onnx_path)
a = np.random.rand(10, 10).astype(np.float32)
b = np.random.rand(10, 10).astype(np.float32)
x = np.random.rand(10, 10).astype(np.float32)

output = sess.run(["output"], {"a": a, "b": b, "x": x})[0]

assert np.allclose(output, a * x + b)
