"""python -m App 启动网页后端。"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from yg_quant_repo import relaunch_as_module

relaunch_as_module("App")

from .server import main

if __name__ == "__main__":
    main()
