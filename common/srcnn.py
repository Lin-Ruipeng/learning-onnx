"""教程第 1~2 章共用的超分辨率模型与权重加载（第 3 章只分析其导出产物）。"""

from pathlib import Path

import torch
from torch import nn

from .download import ensure
from .paths import MODELS

# MMEditing 官方提供的 SRCNN 权重（放大 3 倍）
SRCNN_WEIGHTS_URL = (
    "https://download.openmmlab.com/mmediting/restorers/srcnn/"
    "srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth"
)


class SuperResolutionNet(nn.Module):
    """第 1 章的版本：放大倍数写死在 ``nn.Upsample`` 里。"""

    def __init__(self, upscale_factor: int = 3) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
        )

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def load_weights(model: nn.Module, weights: Path | None = None) -> nn.Module:
    """把官方 checkpoint 载进任意 SRCNN 变体（第 2 章的改版也能用同一份权重）。

    官方 checkpoint 的 key 带 ``generator.`` 前缀（MMEditing 多卡训练留下的），
    这里统一去掉第一段后再 ``load_state_dict``。
    """
    weights = Path(weights) if weights is not None else MODELS / "srcnn.pth"
    ensure(SRCNN_WEIGHTS_URL, weights)

    state_dict = torch.load(weights, map_location="cpu", weights_only=False)["state_dict"]
    for old_key in list(state_dict.keys()):
        state_dict[".".join(old_key.split(".")[1:])] = state_dict.pop(old_key)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def init_torch_model(
    upscale_factor: int = 3, weights: Path | None = None
) -> SuperResolutionNet:
    """构造第 1 章的模型并载入官方权重。"""
    return load_weights(SuperResolutionNet(upscale_factor=upscale_factor), weights)
