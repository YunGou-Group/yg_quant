"""因子评估：BaseMetric 插件、契约 B 远期收益、可插拔指标。网页进程在 App。"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .base_metric import BaseMetric
from .cross_section_ic_calculator import CrossSectionICCalculator
from .context import BatchEvalContext, EvalContext, LibraryEvalContext
from .exposure_engine import ExposureEngine, ExposureMatrix, RAW_STYLE_NAMES
from .factor_eval_service import FactorEvalService
from .factor_evaluator import FactorEvaluator
from .return_calculator import ReturnCalculator
from .field_doc import FieldDoc
from .metric_discoverer import MetricDiscoverer
from .metric_result import MetricResult
from .param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec, universe_param
from .summary_writer import SummaryWriter

__all__ = [
    "HORIZON_PARAM",
    "UNIVERSE_PARAM",
    "universe_param",
    "BaseMetric",
    "CrossSectionICCalculator",
    "EvalContext",
    "BatchEvalContext",
    "LibraryEvalContext",
    "ExposureEngine",
    "ExposureMatrix",
    "RAW_STYLE_NAMES",
    "FactorEvalService",
    "FactorEvaluator",
    "FieldDoc",
    "ReturnCalculator",
    "MetricDiscoverer",
    "MetricResult",
    "ParamSpec",
    "SummaryWriter",
]
