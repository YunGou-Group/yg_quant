"""可注册的 asof 股票池。评估、回测、实盘、风格拥挤共用；不约束数据拉取。"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .base import Universe
from .catalog import build_mask, get, listing, names

__all__ = [
    "Universe",
    "build_mask",
    "get",
    "listing",
    "names",
]
