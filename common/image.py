"""图像与张量之间的形状/取值范围转换，各章都要用。"""

import numpy as np


def to_nchw(img: np.ndarray) -> np.ndarray:
    """HWC(uint8) -> NCHW(float32)。"""
    return np.expand_dims(np.transpose(img.astype(np.float32), [2, 0, 1]), 0)


def to_hwc(tensor: np.ndarray) -> np.ndarray:
    """NCHW(float) -> HWC(uint8)，并把颜色值截断回 [0, 255]。"""
    img = np.squeeze(tensor, 0)
    img = np.clip(img, 0, 255)
    return np.transpose(img, [1, 2, 0]).astype(np.uint8)
