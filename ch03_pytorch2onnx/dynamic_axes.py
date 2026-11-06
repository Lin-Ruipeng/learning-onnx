from pathlib import Path

import numpy as np
import onnxruntime
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)
        return x


model = Model()
dummy_input = torch.rand(1, 3, 10, 10)

# 输出目录：以脚本所在目录为基准，避免依赖运行时的当前工作目录
output_dir = Path(__file__).with_suffix("").parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 不仅要说明动态维度，还需要显示的写出名字
# 比如说，在输入中有一个动态维度，下标是0,名字是batch
model_configs = [
    {"path": output_dir / "model_static.onnx", "dynamic_axes": None},
    {
        "path": output_dir / "model_dynamic_0.onnx",
        "dynamic_axes": {
            "in": {0: "batch"},
            "out": {0: "batch"},
        },
    },
    {
        "path": output_dir / "model_dynamic_23.onnx",
        "dynamic_axes": {
            "in": [2, 3],
            "out": [2, 3],
        },
    },
]

for config in model_configs:
    torch.onnx.export(
        model,
        dummy_input,
        config["path"],
        input_names=["in"],
        output_names=["out"],
        dynamic_axes=config["dynamic_axes"],  # 指定动态维度
        dynamo=False,
    )

# 上面是导出代码
# 下面是推理

# 创建输入
origin_tensor = np.random.rand(1, 3, 10, 10).astype(np.float32)
mult_batch_tensor = np.random.rand(2, 3, 10, 10).astype(np.float32)
big_tensor = np.random.rand(1, 3, 20, 20).astype(np.float32)

inputs = [origin_tensor, mult_batch_tensor, big_tensor]
exceptions = {}

# 对每个模型遍历
for config in model_configs:
    model_path = config["path"]
    model_name = model_path.name
    # 每个模型遍历输入
    for i, input in enumerate(inputs):
        try:
            ort_session = onnxruntime.InferenceSession(str(model_path))
            ort_inputs = {"in": input}
            ort_session.run(["out"], ort_inputs)
        except Exception as e:  # noqa: BLE001
            exceptions[(i, model_name)] = e
            print(f"Input[{i}] on model {model_name} error.")
        else:
            print(f"Input[{i}] on model {model_name} succeed.")
