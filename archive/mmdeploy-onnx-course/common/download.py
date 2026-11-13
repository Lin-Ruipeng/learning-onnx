"""教程里散落的 ``requests.get(...).content`` 收敛成一个可复用的下载函数。"""

from pathlib import Path
from urllib.parse import urlparse

import requests


def ensure(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
    """确保 ``dest`` 存在且非空，缺失时下载。

    不用 ``open(dest, "wb").write(resp.content)`` 直接写盘，是因为中途失败会留下
    0 字节或半个文件，而下次 ``if not os.path.exists`` 又会跳过它，导致静默损坏。
    这里先写 ``.part``，确认非空后再 rename（同文件系统内为原子操作）。
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(resp.content)
    if tmp.stat().st_size == 0:
        raise RuntimeError(f"下载到的文件为空: {url}")
    tmp.replace(dest)
    return dest


def filename_of(url: str) -> str:
    """从 URL 里取出文件名，例如 ``.../000001.png`` -> ``000001.png``。"""
    return Path(urlparse(url).path).name
