#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""目标权重 + 分配资金 → QMT 对接 JSON（代码与手数）。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..holdings import TargetHoldings
from .codes import to_qmt_code

LOT = 100


@dataclass
class QmtOrder:
    code: str
    side: str
    volume: int
    symbol: str
    weight: float
    price: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "code": self.code,
            "symbol": self.symbol,
            "side": self.side,
            "volume": int(self.volume),
            "weight": float(self.weight),
        }
        if self.price is not None and np.isfinite(self.price):
            row["price"] = float(self.price)
        return row


@dataclass
class QmtOrderPlan:
    asof: str
    execute_on: Optional[str]
    capital: float
    strategy: str
    allocator: str
    account: str = ""
    lot_size: int = LOT
    holdings: List[Dict[str, Any]] = field(default_factory=list)
    orders: List[QmtOrder] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": "live",
            "created_at": datetime.now().astimezone().isoformat(),
            "strategy": self.strategy,
            "allocator": self.allocator,
            "asof": self.asof,
            "execute_on": self.execute_on,
            "capital": float(self.capital),
            "account": self.account,
            "lot_size": int(self.lot_size),
            "holdings": list(self.holdings),
            "orders": [row.as_dict() for row in self.orders],
            "skipped": list(self.skipped),
            "params": dict(self.params),
        }

    def write(self, path: Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, dest)
        return dest


def lots_from_weight(weight: float, capital: float, price: float, lot_size: int = LOT) -> int:
    if weight <= 0 or capital <= 0 or not np.isfinite(price) or price <= 0:
        return 0
    shares = float(weight) * float(capital) / float(price)
    lot = max(1, int(lot_size))
    return int(np.floor(shares / lot + 1e-9) * lot)


def plan_from_target(
    target: TargetHoldings,
    prices: Mapping[str, float],
    *,
    capital: float,
    strategy: str,
    allocator: str,
    account: str = "",
    lot_size: int = LOT,
    params: Optional[Dict[str, Any]] = None,
    current: Optional[Mapping[str, float]] = None,
) -> QmtOrderPlan:
    cash = float(capital)
    if cash <= 0:
        raise ValueError("实盘必须指定正的分配资金 --cash")
    lot = max(1, int(lot_size))
    held = {
        str(symbol): float(weight)
        for symbol, weight in (current or {}).items()
        if float(weight) > 1e-9
    }
    holdings: List[Dict[str, Any]] = []
    orders: List[QmtOrder] = []
    skipped: List[Dict[str, Any]] = []
    for symbol in sorted(set(target.weights) | set(held)):
        code = to_qmt_code(symbol)
        price = float(prices.get(symbol, np.nan))
        want_w = float(target.weights.get(symbol, 0.0) or 0.0)
        have_w = float(held.get(symbol, 0.0) or 0.0)
        want = lots_from_weight(want_w, cash, price, lot) if want_w > 1e-9 else 0
        have = lots_from_weight(have_w, cash, price, lot) if have_w > 1e-9 else 0
        px = float(price) if np.isfinite(price) and price > 0 else None
        if want_w > 1e-9 and want <= 0:
            skipped.append(
                {
                    "symbol": symbol,
                    "code": code,
                    "weight": want_w,
                    "price": px,
                    "reason": "no_price" if px is None else "below_lot",
                }
            )
            continue
        if want > 0:
            holdings.append(
                {
                    "code": code,
                    "symbol": symbol,
                    "volume": want,
                    "weight": want_w,
                    "price": px,
                }
            )
        delta = want - have
        if delta == 0:
            continue
        orders.append(
            QmtOrder(
                code=code,
                side="buy" if delta > 0 else "sell",
                volume=abs(int(delta)),
                symbol=symbol,
                weight=want_w,
                price=px,
            )
        )
    return QmtOrderPlan(
        asof=target.asof,
        execute_on=target.execute_on,
        capital=cash,
        strategy=strategy,
        allocator=allocator,
        account=str(account or ""),
        lot_size=lot,
        holdings=holdings,
        orders=orders,
        skipped=skipped,
        params=dict(params or {}),
    )


def default_order_path() -> Path:
    from yg_quant_repo import strategy_runs_dir

    return strategy_runs_dir("live") / "qmt_orders.json"
