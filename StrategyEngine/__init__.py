"""日频策略运行时：score(ctx) → 目标权重；回测与实盘共用合同。"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .backtest import Engine, EngineResult, OpenFillExecutor, PanelStore, summarize, trades_frame
from .context import DayContext
from .fill import Broker, FillReport
from .holdings import TargetHoldings
from .allocators import Allocator, EqualWeight, MinVol, get_allocator
from .strategy import Strategy
from .live import QmtOrderPlan, plan_live, to_qmt_code

__all__ = [
    "Broker",
    "DayContext",
    "Engine",
    "EngineResult",
    "FillReport",
    "OpenFillExecutor",
    "EqualWeight",
    "MinVol",
    "PanelStore",
    "Allocator",
    "Strategy",
    "TargetHoldings",
    "get_allocator",
    "summarize",
    "trades_frame",
    "QmtOrderPlan",
    "plan_live",
    "to_qmt_code",
]
