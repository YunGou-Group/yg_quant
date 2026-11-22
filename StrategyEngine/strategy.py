#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""策略合同：只返回目标权重。实现放 Strategies/，继承本类并实现 score。

CLI 发现另看 name / from_cli / panel_kwargs / run_tag / cli_fields。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .context import DayContext
from .holdings import TargetHoldings

# 最长产物是 ``{tag}_YYYYMMDD_HHMMSS_attribution.json``。
# Excel 打不开路径长于 218 的工作簿，报「文件格式或扩展名无效」。
_EXCEL_PATH_MAX = 218
_RUN_SUFFIX = "_YYYYMMDD_HHMMSS_attribution.json"


def run_tag_max() -> int:
    """run tag 上限：目录 + tag + 最长后缀不超过 Excel 的路径长度。"""
    from yg_quant_repo import strategy_runs_dir

    folder = len(str(strategy_runs_dir("backtest"))) + 1
    room = _EXCEL_PATH_MAX - folder - len(_RUN_SUFFIX) - 8
    return max(48, room)


def shorten_run_tag(tag: str, max_len: int | None = None) -> str:
    """超长 run tag 保留前缀并接 sha1，避免保存后 Excel 认不出文件。"""
    if max_len is None:
        max_len = run_tag_max()
    text = str(tag or "").strip()
    limit = max(16, int(max_len))
    if len(text) <= limit:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    keep = limit - len(digest) - 1
    if keep < 8:
        return digest
    return f"{text[:keep].rstrip('_')}_{digest}"


def cli_field(
    key: str,
    label: str,
    type: str = "str",
    *,
    default: Any = None,
    required: bool = False,
    placeholder: Optional[str] = None,
) -> Dict[str, Any]:
    """策略参数的一项。写在 cli_fields 里，CLI 与网页都会自动发现。"""
    spec: Dict[str, Any] = {"key": str(key), "label": str(label), "type": str(type)}
    if default is not None:
        spec["default"] = default
    if required:
        spec["required"] = True
    if placeholder:
        spec["placeholder"] = str(placeholder)
    return spec


def n_field(default: int, *, label: str = "持仓数") -> Dict[str, Any]:
    return cli_field("n", label, "int", default=int(default))


def rebalance_field(
    default: str = "daily",
    *,
    placeholder: str = "daily / weekly / 5",
) -> Dict[str, Any]:
    return cli_field("rebalance", "调仓频率", "str", default=str(default), placeholder=placeholder)


class Strategy(ABC):
    """日频策略。引擎只调用 score；不要在这里加选股骨架。"""

    name: str = ""

    @classmethod
    def cli_fields(cls) -> List[Dict[str, Any]]:
        """策略自己的参数。默认没有；新策略在本类声明即可，不用改 run / CLI。"""
        return []

    @abstractmethod
    def score(self, ctx: DayContext) -> Optional[TargetHoldings]:
        """当天唯一入口：asof 收盘可见信息 → 目标权重（T+1 开盘成交）。

        None 与全 0 都是空仓。标的列表和目标权重与上次相同、且已经持有这组股票时，
        引擎不再按次日开盘配平，价格涨跌只改变市值、不改股数。
        想换仓就给出不同的股票或权重。
        """
        ...
