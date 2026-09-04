#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨日派生调度：公式在 metric.compute_crossday，这里只按指标调用。"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..base_metric import BaseMetric


class CrossDayEngine:
    def run(
        self,
        arrays: Dict[str, np.ndarray],
        params: dict,
        metrics: Optional[Sequence[BaseMetric]] = None,
    ) -> Dict[str, np.ndarray]:
        if metrics is None:
            from ..metric_discoverer import MetricDiscoverer

            metrics = [m for m in MetricDiscoverer().metrics if m.has_crossday()]
        out = dict(arrays)
        for metric in metrics:
            produced = metric.compute_crossday(out, params) or {}
            out.update(produced)
        return out
