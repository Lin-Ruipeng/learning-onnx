from __future__ import annotations

import sys
from pathlib import Path

CHAPTER_SRC = Path(__file__).parents[1] / "src"
COURSE_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(COURSE_ROOT))
sys.path.insert(0, str(CHAPTER_SRC))
