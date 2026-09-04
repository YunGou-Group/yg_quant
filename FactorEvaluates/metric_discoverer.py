#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指标发现者：加载 metrics/ 下 BaseMetric 子类，并按 requires 拓扑排序。"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .base_metric import BaseMetric
from .param_spec import HORIZON_PARAM, UNIVERSE_PARAM, universe_param

logger = logging.getLogger("FactorEvaluates")


class MetricDiscoverer:
    def __init__(self, metrics_dir: Optional[Path] = None):
        self.metrics_dir = (
            Path(metrics_dir)
            if metrics_dir is not None
            else Path(__file__).parent / "metrics"
        )
        self.metrics: List[BaseMetric] = self.discover()

    def discover(self) -> List[BaseMetric]:
        root = self.metrics_dir
        if not root.is_dir():
            logger.warning("指标目录不存在: %s", root)
            return []

        found: List[BaseMetric] = []
        seen_names = set()
        for path in sorted(root.rglob("*.py")):
            if path.name.startswith("_") or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).with_suffix("")
            package = (__package__ or "FactorEvaluates") + ".metrics"
            module_name = package + "." + ".".join(rel.parts)
            try:
                module = importlib.import_module(module_name)
            except Exception:
                logger.exception("导入指标模块失败: %s", module_name)
                continue
            loaded = 0
            for attr in dir(module):
                candidate = getattr(module, attr)
                if (
                    isinstance(candidate, type)
                    and issubclass(candidate, BaseMetric)
                    and candidate is not BaseMetric
                    and candidate.__module__ == module.__name__
                ):
                    metric = candidate()
                    name = metric.get_name()
                    if not name:
                        raise ValueError(f"{candidate.__name__} 未设置 name")
                    if name in seen_names:
                        raise ValueError(f"重复的指标名: {name}")
                    seen_names.add(name)
                    found.append(metric)
                    loaded += 1
            if loaded:
                logger.info("加载指标模块 %s: %s 个", module_name, loaded)
        return found

    def by_name(self) -> Dict[str, BaseMetric]:
        return {metric.get_name(): metric for metric in self.metrics}

    def metrics_for(self, eval_scope: Optional[str] = "single") -> List[BaseMetric]:
        """按 eval_scope 过滤。None / 'all' 返回全部已发现指标。"""
        if eval_scope in (None, "", "all"):
            return list(self.metrics)
        wanted = str(eval_scope).strip().lower()
        if wanted not in {"single", "library"}:
            raise ValueError(f"eval_scope 必须是 single、library 或 all，收到 {eval_scope!r}")
        return [metric for metric in self.metrics if metric.get_eval_scope() == wanted]

    def reject_library_metrics(self, names: Sequence[str]) -> None:
        """单因子路径禁止库级指标。"""
        catalog = self.by_name()
        blocked = [
            name
            for name in names
            if name in catalog and catalog[name].get_eval_scope() == "library"
        ]
        if blocked:
            raise ValueError(f"库级指标不能用于单因子评估: {sorted(blocked)}")

    def producer_map(self, metrics: Optional[Sequence[BaseMetric]] = None) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for metric in metrics if metrics is not None else self.metrics:
            for key in metric.produces:
                if key in mapping:
                    raise ValueError(
                        f"中间量 {key!r} 被 {mapping[key]} 和 {metric.get_name()} 同时 produces"
                    )
                mapping[key] = metric.get_name()
        return mapping

    def resolve(
        self,
        selected: Optional[Sequence[str]] = None,
        metrics: Optional[Sequence[BaseMetric]] = None,
        eval_scope: Optional[str] = "single",
    ) -> List[BaseMetric]:
        if selected is not None and eval_scope == "single":
            self.reject_library_metrics(selected)
        if metrics is not None:
            catalog = list(metrics)
            if eval_scope in {"single", "library"}:
                catalog = [
                    metric
                    for metric in catalog
                    if metric.get_eval_scope() == eval_scope
                ]
        else:
            catalog = self.metrics_for(eval_scope)
        by_name = {metric.get_name(): metric for metric in catalog}
        producers = self.producer_map(catalog)
        if selected is None:
            wanted = set(by_name)
        else:
            wanted = set(selected)
            missing = wanted - set(by_name)
            if missing:
                raise ValueError(f"未知指标: {sorted(missing)}")

        stack = list(wanted)
        while stack:
            name = stack.pop()
            metric = by_name[name]
            for key in metric.requires:
                producer = producers.get(key)
                if producer is None:
                    raise ValueError(
                        f"指标 {name} 需要中间量 {key!r}，但没有指标 produces 它"
                    )
                if producer not in wanted:
                    wanted.add(producer)
                    stack.append(producer)

        pending = {name: by_name[name] for name in wanted}
        ordered: List[BaseMetric] = []
        available = set()
        while pending:
            ready = [
                name
                for name, metric in pending.items()
                if all(producers[key] in available for key in metric.requires)
            ]
            if not ready:
                cycle = ", ".join(sorted(pending))
                raise ValueError(f"指标依赖成环: {cycle}")
            ready.sort()
            for name in ready:
                ordered.append(pending.pop(name))
                available.add(name)
        return ordered

    def schema(self, eval_scope: Optional[str] = "single") -> Dict[str, Any]:
        shared: Dict[str, Dict] = {
            HORIZON_PARAM.name: HORIZON_PARAM.to_dict(),
            UNIVERSE_PARAM.name: universe_param().to_dict(),
        }
        metrics = []
        for metric in self.metrics_for(eval_scope):
            own = []
            for spec in metric.params():
                item = spec.to_dict()
                if spec.scope == "shared":
                    shared[spec.name] = item
                else:
                    own.append(item)
            metrics.append(
                {
                    "name": metric.get_name(),
                    "dimension": metric.get_dimension(),
                    "description": metric.get_description(),
                    "cost": metric.cost,
                    "eval_scope": metric.get_eval_scope(),
                    "requires": list(metric.requires),
                    "produces": list(metric.produces),
                    "params": own,
                    "fields": [item.to_dict() for item in metric.fields()],
                }
            )
        return {"shared": list(shared.values()), "metrics": metrics}
