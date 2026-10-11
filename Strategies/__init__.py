"""策略库：只实现 score(ctx) -> TargetHoldings。回测与以后的 QMT 共用。"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .equal import EqualWeightTopK
from .icir import IcirWeightedTopK
from .lgbm import LgbmWeightedTopK
from .lgbm_barra import LgbmBarraTopK
from .lgbm_classification_specific_return import LgbmClassificationSpecificReturnTopK
from .lgbm_deep import LgbmDeepTopK
from .lgbm_index_enhance import LgbmIndexEnhanceTopK
from .lgbm_shape import LgbmShapeTopK
from .lgbm_sn import LgbmSizeNlTopK
from .lgbm_snp import LgbmSnPathTopK
from .lgbm_tc import LgbmTcTopK
from .mlp import MlpTopK
from .small_cap import CANDIDATE_N, HOLD_N, FinRow, SmallCapData, SmallCapStrategy
from .topk import FactorTopK
from .wufu import WufuEtf
from .wufu_a088 import WufuAlpha088
from .wufu_mix import WufuMix
from .wufu_ns import WufuXinChun
from .wufu_rank import WufuRank

__all__ = [
    "CANDIDATE_N",
    "EqualWeightTopK",
    "FactorTopK",
    "FinRow",
    "HOLD_N",
    "IcirWeightedTopK",
    "LgbmBarraTopK",
    "LgbmClassificationSpecificReturnTopK",
    "LgbmDeepTopK",
    "LgbmIndexEnhanceTopK",
    "LgbmShapeTopK",
    "LgbmSizeNlTopK",
    "LgbmSnPathTopK",
    "LgbmTcTopK",
    "LgbmWeightedTopK",
    "MlpTopK",
    "SmallCapData",
    "SmallCapStrategy",
    "WufuAlpha088",
    "WufuEtf",
    "WufuMix",
    "WufuRank",
    "WufuXinChun",
]
