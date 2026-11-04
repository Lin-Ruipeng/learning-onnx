"""第二章：解决模型部署中的难题 —— 让放大倍数变成推理时的输入。

第 1 章的 SRCNN 把放大倍数写死在 ``nn.Upsample`` 里，导出的 ONNX 只能出 3 倍图。
本章用 ``torch.autograd.Function`` 手写插值算子，在导出时把它拼成 ONNX 的 ``Resize``
节点、把缩放系数变成模型的第二个输入，于是同一个 ``.onnx`` 能出任意整数倍。

运行（必须从仓库根目录，这样 ``common`` 才在 sys.path 里）::

    uv run python -m ch02_deployment_challenges.export_dynamic

产物落在 ``ch02_deployment_challenges/outputs/``：srcnn_dynamic.onnx、
face_torch_x3.png（3 倍，应与第 1 章结果一致）、face_ort_x4.png（4 倍，ONNX Runtime）
"""

import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn
from torch.nn.functional import interpolate

from common.image import to_hwc, to_nchw
from common.paths import ASSETS, outputs
from common.srcnn import load_weights

OUT = outputs("ch02_deployment_challenges")


class NewInterpolate(torch.autograd.Function):
    """手搓的插值算子，把动态的缩放倍数变成模型输入。

    autograd.Function 有两个生效时机，正好各管一边：

    - ``forward``：PyTorch 里实际算的时候走这里；
    - ``symbolic``：导出 ONNX 时走这里，用 ``g.op`` 直接拼出对应的 ONNX 节点。
    """

    @staticmethod
    def symbolic(g, input, scales):
        # 拼成 ONNX 的 Resize 节点：input / roi / scales 三个输入 + 四个属性。
        # roi 传空张量表示不用它（Resize-11 之后 scales 与 roi 二选一）。
        return g.op(
            "Resize",
            input,
            g.op("Constant", value_t=torch.tensor([], dtype=torch.float32)),
            scales,
            coordinate_transformation_mode_s="pytorch_half_pixel",
            cubic_coeff_a_f=-0.75,
            mode_s="cubic",
            nearest_mode_s="floor",
        )

    @staticmethod
    def forward(ctx, input, scales):
        # scales 形如 [1, 1, h, w]，只有后两维对 interpolate 有意义
        return interpolate(
            input,
            scale_factor=scales.tolist()[-2:],
            mode="bicubic",
            align_corners=False,
        )


class SuperResolutionNet(nn.Module):
    """第 1 章的模型去掉 ``nn.Upsample``，放大倍数改由第二个输入给出。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, upscale_factor: torch.Tensor) -> torch.Tensor:
        x = NewInterpolate.apply(x, upscale_factor)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def make_scales(factor: float) -> torch.Tensor:
    """缩放系数张量，形如 [1, 1, factor, factor]（N/C 不放大）。"""
    return torch.tensor([1.0, 1.0, factor, factor], dtype=torch.float32)


def run_torch(model: nn.Module, input_img: np.ndarray, factor: float) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(input_img), make_scales(factor)).numpy()


def export_onnx(model: nn.Module, onnx_path) -> None:
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        torch.onnx.export(
            model,
            # 第二个输入是自定义算子要的缩放系数，导出时必须一起传进去
            (dummy, make_scales(3)),
            str(onnx_path),
            # torch>=2.9 默认的 torch.export(dynamo) 导出器不支持 autograd.Function
            # 的 symbolic()，会真的去 trace forward 里的 interpolate 并报 SymFloat
            # 错误，所以本章必须显式切回 TorchScript 导出器（代价是一条
            # DeprecationWarning）。详见本章 README。
            dynamo=False,
            opset_version=18,
            input_names=["input", "factor"],
            output_names=["output"],
        )

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    print(f"导出完成: {onnx_path}")
    print(f"节点: {[node.op_type for node in graph.graph.node]}")
    print(f"输入: {[i.name for i in graph.graph.input]}")
    print("Model correct")


def run_ort(onnx_path, input_img: np.ndarray, factor: float) -> np.ndarray:
    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    scales = np.array([1.0, 1.0, factor, factor], dtype=np.float32)
    return session.run(["output"], {"input": input_img, "factor": scales})[0]


def main() -> None:
    input_img = to_nchw(cv2.imread(str(ASSETS / "face.png")))
    model = load_weights(SuperResolutionNet())

    # 3 倍：对照第 1 章的固定倍数结果
    torch_x3 = run_torch(model, input_img, factor=3)
    cv2.imwrite(str(OUT / "face_torch_x3.png"), to_hwc(torch_x3))

    onnx_path = OUT / "srcnn_dynamic.onnx"
    export_onnx(model, onnx_path)

    # 同一个 .onnx，换一组缩放系数出 4 倍图 —— 本章的目的
    ort_x4 = run_ort(onnx_path, input_img, factor=4)
    cv2.imwrite(str(OUT / "face_ort_x4.png"), to_hwc(ort_x4))

    torch_x4 = run_torch(model, input_img, factor=4)
    diff = float(np.abs(torch_x4 - ort_x4).max())
    print(f"4 倍放大：PyTorch 与 ONNX Runtime 的最大像素差 {diff:.3f}")


if __name__ == "__main__":
    main()
