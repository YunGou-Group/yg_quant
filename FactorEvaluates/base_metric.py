#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估指标抽象基类。子类禁止自己 shift 行情。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .field_doc import FieldDoc
from .metric_result import MetricResult
from .param_spec import ParamSpec


class BaseMetric(ABC):
    name: str = ""
    description: str = ""
    dimension: str = ""
    cost: str = "panel"  # panel | derived
    eval_scope: str = "single"  # single | library
    produces: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()
    engine_requires: Tuple[str, ...] = ()

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_dimension(self) -> str:
        return self.dimension

    def get_eval_scope(self) -> str:
        scope = str(self.eval_scope or "single").strip().lower()
        if scope not in {"single", "library"}:
            raise ValueError(
                f"指标 {self.get_name()!r} 的 eval_scope 必须是 single 或 library，收到 {self.eval_scope!r}"
            )
        return scope

    def params(self) -> Sequence[ParamSpec]:
        return ()

    def fields(self) -> Sequence[FieldDoc]:
        return ()

    def params_schema(self) -> List[Dict[str, Any]]:
        return [spec.to_dict() for spec in self.params()]

    def merge_params(self, incoming: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        incoming = dict(incoming or {})
        merged: Dict[str, Any] = {}
        for spec in self.params():
            if spec.name in incoming and incoming[spec.name] is not None:
                value = incoming[spec.name]
            else:
                value = spec.default
            merged[spec.name] = spec.coerce(value)
        return merged

    @abstractmethod
    def compute(self, ctx: Any, params: Mapping[str, Any]) -> MetricResult:
        """派生类应把 produces 写入 ctx.intermediates。"""

    def compute_matrix(self, batch_ctx: Any, params: Mapping[str, Any]) -> Dict[str, Any]:
        """日度多因子矩阵接口。未实现的指标不进全库默认集。"""
        raise NotImplementedError(self.get_name())

    def has_matrix(self) -> bool:
        return type(self).compute_matrix is not BaseMetric.compute_matrix

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """跨日派生：输入已是 (日期 × 因子) 日度数组。公式必须写在指标类里。"""
        raise NotImplementedError(self.get_name())

    def has_crossday(self) -> bool:
        return type(self).compute_crossday is not BaseMetric.compute_crossday

    def compute_from_labels(
        self, labels: Any, params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """跨日、依赖历史分位标签的指标（如换手）。"""
        raise NotImplementedError(self.get_name())

    def has_label_history(self) -> bool:
        return type(self).compute_from_labels is not BaseMetric.compute_from_labels

    def compute_library(self, library_ctx: Any, params: Mapping[str, Any]) -> Dict[str, Any]:
        """库级接口。仅 eval_scope=library 的指标实现。"""
        raise NotImplementedError(self.get_name())

    def __str__(self) -> str:
        return (
            f"Metric(name={self.get_name()!r}, eval_scope={self.get_eval_scope()!r}, "
            f"cost={self.cost!r}, requires={self.requires}, produces={self.produces})"
        )
