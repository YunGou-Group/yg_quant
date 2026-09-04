#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子评估者：单因子、单 horizon，组上下文并跑已发现指标。"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from .base_metric import BaseMetric
from .context import EvalContext
from .return_calculator import ReturnCalculator
from .metric_discoverer import MetricDiscoverer


class FactorEvaluator:
    def __init__(
        self,
        discoverer: Optional[MetricDiscoverer] = None,
        calculator: Optional[ReturnCalculator] = None,
    ):
        self.discoverer = discoverer if discoverer is not None else MetricDiscoverer()
        self.calculator = calculator

    def evaluate(
        self,
        factor: pd.DataFrame,
        open_panel: pd.DataFrame,
        *,
        horizon: int = 5,
        metric_names: Optional[Sequence[str]] = None,
        params: Optional[Mapping[str, Any]] = None,
        universe_mask: Optional[pd.DataFrame] = None,
        metrics: Optional[Sequence[BaseMetric]] = None,
        calculator: Optional[ReturnCalculator] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        factor_name: Optional[str] = None,
        universe: Optional[str] = None,
        intermediates: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        catalog = list(metrics) if metrics is not None else self.discoverer.metrics
        if not catalog:
            raise ValueError("没有发现可运行的评估指标")
        if metric_names:
            self.discoverer.reject_library_metrics(metric_names)
        ordered = self.discoverer.resolve(
            metric_names, catalog, eval_scope="single"
        )
        incoming = dict(params or {})
        incoming.setdefault("horizon", horizon)

        engine = calculator if calculator is not None else self.calculator
        t0 = time.perf_counter()
        print(f"[eval] 计算远期收益 N={horizon} …", flush=True)
        ctx = EvalContext.build(
            factor,
            open_panel,
            horizon=horizon,
            universe_mask=universe_mask,
            calculator=engine,
        )
        if intermediates:
            ctx.intermediates.update(intermediates)
        print(f"[eval] 远期收益完成 ({time.perf_counter() - t0:.1f}s)", flush=True)
        payload: Dict[str, Any] = {
            "factor": factor_name,
            "horizon": ctx.horizon,
            "label": ctx.label,
            "asof": ctx.asof,
            "start": start,
            "end": end,
            "universe": universe,
            "metrics": {},
        }
        for metric in ordered:
            merged = metric.merge_params(incoming)
            t0 = time.perf_counter()
            print(f"[eval] 指标 {metric.get_name()} …", flush=True)
            result = metric.compute(ctx, merged)
            print(
                f"[eval] 指标 {metric.get_name()} 完成 ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
            payload["metrics"][metric.get_name()] = {
                "scalars": self._sanitize_scalars(result.scalars),
                "series": {
                    key: series.astype("float64")
                    for key, series in result.series.items()
                },
            }
        return payload

    @staticmethod
    def _json_number(value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, AttributeError):
                pass
        if isinstance(value, float):
            if value != value:
                return None
            return float(value)
        return value

    @classmethod
    def _sanitize_scalars(cls, scalars: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: cls._json_number(val) for key, val in scalars.items()}
