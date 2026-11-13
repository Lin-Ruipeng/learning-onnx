"""第一章：模型部署简介 —— 把 PyTorch 的 SRCNN 导出成 ONNX，再用 ONNX Runtime 跑一遍。

运行（必须从仓库根目录，这样 ``common`` 才在 sys.path 里）::

    uv run python -m ch01_introduction_to_deployment.export

产物落在 ``ch01_introduction_to_deployment/outputs/``：srcnn.onnx(+.data)、face_torch.png、face_ort.png
"""

import cv2
import numpy as np
import onnx
import onnxruntime
import torch

from common.download import ensure
from common.image import to_hwc, to_nchw
from common.paths import ASSETS, outputs
from common.srcnn import init_torch_model

FACE_URL = (
    "https://raw.githubusercontent.com/open-mmlab/mmagic/master/tests/data/face/000001.png"
)

OUT = outputs("ch01_introduction_to_deployment")


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
