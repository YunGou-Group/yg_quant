#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""库级引擎：调用 eval_scope=library 指标。"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..base_metric import BaseMetric
from ..context import LibraryEvalContext
from .family_key import family_map


class LibraryMetricEngine:
    def run(
        self,
        metrics: Sequence[BaseMetric],
        names: List[str],
        dates: List[str],
        daily_rank_ic: np.ndarray,
        mean_corr: np.ndarray,
        params: dict,
    ) -> Dict[str, object]:
        ctx = LibraryEvalContext(
            factor_names=list(names),
            dates=list(dates),
            daily_rank_ic=daily_rank_ic,
            mean_factor_corr=mean_corr,
            family_of=family_map(names),
        )
        payload = {}
        for metric in metrics:
            if metric.get_eval_scope() != "library":
                continue
            payload[metric.get_name()] = metric.compute_library(ctx, params)
        return payload
