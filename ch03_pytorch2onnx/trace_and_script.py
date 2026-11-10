from pathlib import Path

import torch


class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.conv = torch.nn.Conv2d(3, 3, 3)

    # 构建了一个 for 循环的前向传播图
    def forward(self, x):
        for i in range(self.n):
            x = self.conv(x)
        return x


# 故意让 for 循环次数不一样，这样子就可以看到 trace 和 script 的差异
models = [Model(2), Model(3)]
model_names = ["model_2", "model_3"]

# 输出目录以脚本所在目录为基准，避免依赖运行时的当前工作目录
output_dir = Path(__file__).with_suffix("").parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

for model, model_name in zip(models, model_names):
    dummy_input = torch.rand(1, 3, 10, 10)
    dummy_output = model(dummy_input)

    # 这里是 trace 方法导出静态图! 所以需要输入!
    model_trace = torch.jit.trace(model, dummy_input)

    # 这里是 script 方法记录控制流程!
    model_script = torch.jit.script(model)

    # 新版 PyTorch 默认走 dynamo 导出, 不再接受 torch.jit.ScriptModule,
    # 也不再支持 example_outputs 参数。为了保留原本的 trace/script 对比,
    # 这里显式使用旧版 TorchScript 导出器 (dynamo=False)。
    
    # trace 方法和 torch.onnx.export(model, ...) 等价
    torch.onnx.export(
        model_trace,
        dummy_input,
        output_dir / f"{model_name}_trace.onnx",
        dynamo=False,
    )

    # script 必须先调用 torch.jit.script
    torch.onnx.export(
        model_script,
        dummy_input,
        output_dir / f"{model_name}_script.onnx",
        dynamo=False,
    )
