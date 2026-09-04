#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成交合同：回测模拟与以后的 QMT 下单共用。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from .holdings import TargetHoldings


@dataclass
class FillReport:
    date: str
    symbols: List[str]
    intended: np.ndarray
    filled: np.ndarray
    price: np.ndarray
    reason: List[str] = field(default_factory=list)
    traded_notional: float = 0.0
    nav_before: float = 0.0
    fee: float = 0.0

    def to_rows(self) -> List[dict]:
        qty = np.asarray(self.filled, dtype=np.float64)
        want = np.asarray(self.intended, dtype=np.float64)
        n = qty.size
        reason = list(self.reason) if self.reason else [""] * n
        if len(reason) < n:
            reason = reason + [""] * (n - len(reason))
        keep = (np.abs(qty) >= 1e-9) | (np.abs(want) >= 1e-9)
        rows = []
        notionals = []
        for i in np.flatnonzero(keep):
            q = float(qty[i])
            w = float(want[i])
            why = reason[i]
            if abs(q) < 1e-9 and not why:
                continue
            if q > 0 or (q == 0 and w > 0):
                side = "buy"
            elif q < 0 or w < 0:
                side = "sell"
            else:
                side = ""
            px = self.price[i]
            price = float(px) if np.isfinite(px) else None
            notionals.append(abs(q) * price if price is not None else 0.0)
            rows.append(
                {
                    "date": self.date,
                    "symbol": self.symbols[i],
                    "side": side,
                    "intended": w,
                    "filled": q,
                    "price": price,
                    "reason": why,
                    "fee": 0.0,
                }
            )
        total_n = float(sum(notionals))
        fee_total = float(self.fee or 0.0)
        if fee_total and total_n > 0:
            for row, notion in zip(rows, notionals):
                row["fee"] = fee_total * notion / total_n
        elif fee_total and rows:
            rows[0]["fee"] = fee_total
        return rows


@runtime_checkable
class Broker(Protocol):
    """把目标权重变成仓位。回测是 OpenFillExecutor，实盘再接 QMT。"""

    def fill(
        self,
        shares: np.ndarray,
        cash: float,
        target: TargetHoldings,
        symbols: Sequence[str],
        price: np.ndarray,
        tradable: Optional[np.ndarray] = None,
        up_limit: Optional[np.ndarray] = None,
        down_limit: Optional[np.ndarray] = None,
        date: str = "",
        last_px: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, FillReport]: ...
