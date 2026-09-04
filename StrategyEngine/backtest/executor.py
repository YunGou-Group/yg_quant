#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T+1 开盘模拟撮合：涨跌停、整手、印花税、隔夜可卖。"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..fill import FillReport
from ..holdings import TargetHoldings

TICK = 0.01
LOT = 100


class OpenFillExecutor:
    def __init__(
        self,
        commission: float = 0.0003,
        stamp: float = 0.0005,
        lot_size: int = LOT,
        tick: float = TICK,
    ):
        self.commission = max(0.0, float(commission))
        self.stamp = max(0.0, float(stamp))
        self.lot_size = max(1, int(lot_size))
        self.tick = float(tick)

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
    ) -> Tuple[np.ndarray, float, FillReport]:
        shares = np.asarray(shares, dtype=np.float64).copy()
        overnight = shares.copy()
        px = np.asarray(price, dtype=np.float64)
        n = shares.size
        names = [str(s) for s in symbols]
        intended = np.zeros(n, dtype=np.float64)
        filled = np.zeros(n, dtype=np.float64)
        live = np.isfinite(px) & (px > 0)
        if tradable is not None:
            live = live & np.asarray(tradable, dtype=bool)
        mark = np.where(live, px, np.nan)
        if last_px is not None:
            lp = np.asarray(last_px, dtype=np.float64)
            halted_hold = (overnight > 1e-12) & ~live & np.isfinite(lp) & (lp > 0)
            mark = np.where(halted_hold, lp, mark)
        marked = np.where(np.isfinite(mark) & (mark > 0), shares * mark, 0.0)
        value = float(cash) + float(marked.sum())
        report = FillReport(
            date=str(date),
            symbols=names,
            intended=intended,
            filled=filled,
            price=px,
            reason=[],
            nav_before=value,
        )
        if value <= 0:
            return shares, float(cash), report

        target_w = target.aligned(names)
        desired = overnight.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            desired[live] = target_w[live] * value / px[live]
        intended[:] = desired - overnight

        limit_up = self._hit_limit(px, up_limit, side="up")
        limit_down = self._hit_limit(px, down_limit, side="down")
        sell, buy, reasons = self._book(intended, overnight, live, limit_up, limit_down)

        shares, cash, filled_sell, fee_sell = self._apply_delta(shares, cash, px, sell, sell=True)
        filled += filled_sell

        lot = float(self.lot_size)
        buy_notional = float(np.sum(np.where(np.isfinite(px), buy * px, 0.0)))
        buy_cost = buy_notional * (1.0 + self.commission)
        if buy_cost > cash + 1e-9 and buy_cost > 0:
            scale = max(0.0, cash / buy_cost)
            buy = np.floor(buy * scale / lot) * lot
            starved = (intended > 1e-9) & live & ~limit_up & (buy <= 0) & (reasons == "")
            reasons[starved] = "cash"
        shares, cash, filled_buy, fee_buy = self._apply_delta(shares, cash, px, buy, sell=False)
        filled += filled_buy
        traded = float(np.sum(np.abs(np.where(np.isfinite(px), filled * px, 0.0))))
        report.intended = intended
        report.filled = filled
        report.reason = reasons.tolist()
        report.traded_notional = traded
        report.fee = float(fee_sell + fee_buy)
        return shares, cash, report

    def _book(self, intended, overnight, live, limit_up, limit_down):
        n = intended.size
        lot = float(self.lot_size)
        reasons = np.empty(n, dtype=object)
        reasons[:] = ""
        sell = np.zeros(n, dtype=np.float64)
        buy = np.zeros(n, dtype=np.float64)
        active = np.abs(intended) >= 1e-9
        if not np.any(active):
            return sell, buy, reasons

        halted = active & ~live
        reasons[halted] = "halt"

        selling = active & live & (intended < 0)
        blocked_dn = selling & limit_down
        reasons[blocked_dn] = "limit_down"
        sell_ok = selling & ~limit_down
        want = np.where(sell_ok, -intended, 0.0)
        cap = np.asarray(overnight, dtype=np.float64)
        clipped = sell_ok & (want > cap + 1e-9)
        reasons[clipped] = "t1"
        want = np.minimum(want, cap)
        flatten = sell_ok & (want + 1e-9 >= cap)
        sell[flatten] = -cap[flatten]
        partial = sell_ok & ~flatten
        lots = np.floor(want / lot) * lot
        lot_fail = partial & (lots <= 0)
        reasons[lot_fail] = "lot"
        take = partial & (lots > 0)
        sell[take] = -lots[take]

        buying = active & live & (intended > 0)
        blocked_up = buying & limit_up
        reasons[blocked_up] = "limit_up"
        buy_ok = buying & ~limit_up
        buy_lots = np.floor(intended / lot) * lot
        buy_lot_fail = buy_ok & (buy_lots <= 0)
        reasons[buy_lot_fail] = "lot"
        buy[buy_ok & (buy_lots > 0)] = buy_lots[buy_ok & (buy_lots > 0)]
        return sell, buy, reasons

    def _hit_limit(
        self,
        px: np.ndarray,
        band: Optional[np.ndarray],
        *,
        side: str,
    ) -> np.ndarray:
        out = np.zeros(px.shape, dtype=bool)
        if band is None:
            return out
        lim = np.asarray(band, dtype=np.float64)
        ok = np.isfinite(lim) & np.isfinite(px)
        if side == "up":
            out[ok] = px[ok] + 1e-12 >= lim[ok]
        else:
            out[ok] = px[ok] - 1e-12 <= lim[ok]
        return out

    def _apply_delta(
        self,
        shares: np.ndarray,
        cash: float,
        price: np.ndarray,
        delta: np.ndarray,
        *,
        sell: bool,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        notional = np.where(np.isfinite(price) & np.isfinite(delta), delta * price, 0.0)
        abs_n = float(np.sum(np.abs(notional)))
        fee = abs_n * self.commission
        if sell:
            fee += abs_n * self.stamp
        shares = shares + delta
        cash = float(cash) - float(np.sum(notional)) - fee
        return shares, cash, delta.copy(), float(fee)
