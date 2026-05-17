from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    src_text = str(SRC)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)

from video_to_text.cli import main


if __name__ == "__main__":
    main()
