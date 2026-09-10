"""策略库：只实现 score(ctx) -> TargetHoldings。回测与以后的 QMT 共用。"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .icir import IcirWeightedTopK
from .multifactor import MultiFactorTopK
from .small_cap import CANDIDATE_N, HOLD_N, FinRow, SmallCapData, SmallCapStrategy
from .topk import FactorTopK

__all__ = [
    "CANDIDATE_N",
    "FactorTopK",
    "FinRow",
    "HOLD_N",
    "IcirWeightedTopK",
    "MultiFactorTopK",
    "SmallCapData",
    "SmallCapStrategy",
]
