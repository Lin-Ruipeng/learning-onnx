"""项目内路径的唯一来源。

所有章节脚本都从这里取路径，脚本里不出现 ``"face.png"`` 这类相对当前工作目录
的字面量 —— 否则换个目录执行就会把文件写到别处。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ASSETS = ROOT / "assets"
MODELS = ROOT / "models"


def outputs(chapter: str) -> Path:
    """返回某章的产物目录（不存在则创建），如 ``outputs("ch03_pytorch2onnx")``。"""
    path = ROOT / chapter / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path
