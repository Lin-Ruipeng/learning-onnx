from __future__ import annotations

import sys
from pathlib import Path

CHAPTER_SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(CHAPTER_SRC))
