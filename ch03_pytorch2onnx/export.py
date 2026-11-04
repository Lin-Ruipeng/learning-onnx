"""第三章：PyTorch 转 ONNX 详解 —— 导出、校验、用 ONNX Runtime 跑一遍。

运行（必须从仓库根目录，这样 ``common`` 才在 sys.path 里）::

    uv run python -m ch03_pytorch2onnx.export

产物落在 ``ch03_pytorch2onnx/outputs/``：srcnn.onnx(+.data)、face_torch.png、face_ort.png
"""

import cv2
import numpy as np
import onnx
import onnxruntime
import torch

from common.download import ensure
from common.paths import ASSETS, outputs
from common.srcnn import init_torch_model

FACE_URL = (
    "https://raw.githubusercontent.com/open-mmlab/mmagic/master/tests/data/face/000001.png"
)

OUT = outputs("ch03_pytorch2onnx")


def to_nchw(img: np.ndarray) -> np.ndarray:
    """HWC(uint8) -> NCHW(float32)。"""
    return np.expand_dims(np.transpose(img.astype(np.float32), [2, 0, 1]), 0)


def to_hwc(tensor: np.ndarray) -> np.ndarray:
    """NCHW -> HWC(uint8)，并把超分结果截断回 [0, 255]。"""
    img = np.squeeze(tensor, 0)
    img = np.clip(img, 0, 255)
    return np.transpose(img, [1, 2, 0]).astype(np.uint8)


def main() -> None:
    ensure(FACE_URL, ASSETS / "face.png")
    input_img = to_nchw(cv2.imread(str(ASSETS / "face.png")))

    model = init_torch_model(upscale_factor=3)

    # PyTorch 推理，作为后面 ONNX Runtime 结果的参照
    with torch.no_grad():
        torch_output = model(torch.from_numpy(input_img)).numpy()
    cv2.imwrite(str(OUT / "face_torch.png"), to_hwc(torch_output))

    # 导出：dummy input 的形状会被固化，所以后面 ORT 也用同样形状的输入
    x = torch.randn(1, 3, 256, 256)
    onnx_path = OUT / "srcnn.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            str(onnx_path),
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
        )
    print(f"导出完成: {onnx_path}")

    # 校验
    onnx_model = onnx.load(str(onnx_path))
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as exc:
        print(f"Model incorrect: {exc}")
        raise
    else:
        print("Model correct")

    # ONNX Runtime 推理
    ort_session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    ort_output = ort_session.run(["output"], {"input": input_img})[0]
    cv2.imwrite(str(OUT / "face_ort.png"), to_hwc(ort_output))

    diff = float(np.abs(to_hwc(ort_output).astype(np.int16) - to_hwc(torch_output).astype(np.int16)).max())
    print(f"PyTorch 与 ONNX Runtime 输出的最大像素差: {diff}")


if __name__ == "__main__":
    main()
