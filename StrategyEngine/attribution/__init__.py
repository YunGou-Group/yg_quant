"""回测事后业绩归因：读某次 strategy_runs 快照，不参与 score / 撮合。"""

from .evaluate import (
    attribute_for_web,
    attribute_snapshot,
    load_attribution_view,
)

__all__ = ["attribute_for_web", "attribute_snapshot", "load_attribution_view"]
