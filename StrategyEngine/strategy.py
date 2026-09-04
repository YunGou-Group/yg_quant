#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""策略合同：只返回目标权重。实现放 Strategies/，继承本类并实现 score。

CLI 发现另看 name / from_cli / panel_kwargs / run_tag，不写进抽象方法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .context import DayContext
from .holdings import TargetHoldings


class Strategy(ABC):
    """日频策略。引擎只调用 score；不要在这里加选股骨架。"""

    name: str = ""

    @abstractmethod
    def score(self, ctx: DayContext) -> Optional[TargetHoldings]:
        """当天唯一入口：asof 收盘可见信息 → 目标权重（T+1 开盘成交）。

        None 与全 0 都是空仓。标的列表和目标权重与上次相同、且已经持有这组股票时，
        引擎不再按次日开盘配平，价格涨跌只改变市值、不改股数。
        想换仓就给出不同的股票或权重。
        """
        ...
