#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指标文档：由 schema FieldDoc 生成。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..metric_discoverer import MetricDiscoverer


def list_docs(discoverer: Optional[MetricDiscoverer] = None) -> List[Dict[str, Any]]:
    engine = discoverer or MetricDiscoverer()
    pages = [
        {
            "slug": "contract",
            "title": "评估口径",
            "body": (
                "契约 B：信号在收盘 T 可得，远期收益为 open[T+1+N]/open[T+1]-1。\n"
                "单因子路径使用 EvalContext；全库路径用 BatchEvalContext / LibraryEvalContext，不改标签定义。"
            ),
        }
    ]
    for metric in engine.metrics:
        lines = [
            f"# {metric.get_name()}",
            "",
            f"范围：`{metric.get_eval_scope()}`　维度：{metric.get_dimension() or '-'}",
            "",
            metric.get_description() or "",
            "",
        ]
        for field in metric.fields():
            lines.append(f"- **{field.label}** (`{field.name}`)：{field.meaning}")
        pages.append(
            {
                "slug": metric.get_name(),
                "title": metric.get_name(),
                "body": "\n".join(lines),
            }
        )
    return pages


def get_doc(slug: str, discoverer: Optional[MetricDiscoverer] = None) -> Dict[str, Any]:
    for page in list_docs(discoverer):
        if page["slug"] == slug:
            return page
    raise FileNotFoundError(slug)