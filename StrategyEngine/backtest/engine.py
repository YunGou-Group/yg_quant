#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按日切 ctx → 策略目标 → T+1 开盘撮合。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import logging
import numpy as np
import pandas as pd

from ..fill import Broker, FillReport
from ..holdings import TargetHoldings
from ..strategy import Strategy
from .executor import OpenFillExecutor
from .panel import PanelStore

_LOG = logging.getLogger("StrategyEngine")
_HOLD_EPS = 1e-9


@dataclass
class EngineResult:
    dates: List[str]
    nav: np.ndarray
    cash: np.ndarray
    weights: np.ndarray
    turnover: np.ndarray
    targets: List[Optional[TargetHoldings]] = field(default_factory=list)
    fills: List[Optional[FillReport]] = field(default_factory=list)
    # 退市导致的强制变现，不走 Broker，所以单独记一份
    forced_fills: List[FillReport] = field(default_factory=list)

    def equity(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"nav": self.nav, "cash": self.cash, "turnover": self.turnover},
            index=self.dates,
        )


class Engine:
    def __init__(
        self,
        panel: PanelStore,
        *,
        executor: Optional[Broker] = None,
        initial_cash: float = 1_000_000.0,
        commission: float = 0.0003,
        stamp: float = 0.0005,
        lot_size: int = 100,
    ):
        self.panel = panel
        self.executor = executor or OpenFillExecutor(
            commission=commission, stamp=stamp, lot_size=lot_size
        )
        self.initial_cash = float(initial_cash)

    def run(self, strategy: Strategy) -> EngineResult:
        t0 = int(self.panel.active_index)
        n_dates = len(self.panel.dates)
        n_run = n_dates - t0
        n_stocks = len(self.panel.symbols)
        shares = np.zeros(n_stocks, dtype=np.float64)
        cash = self.initial_cash
        navs = np.zeros(n_run, dtype=np.float64)
        cashes = np.zeros(n_run, dtype=np.float64)
        weights = np.zeros((n_run, n_stocks), dtype=np.float64)
        turnover = np.zeros(n_run, dtype=np.float64)
        targets: List[Optional[TargetHoldings]] = []
        fills: List[Optional[FillReport]] = [None] * n_run
        forced_fills: List[FillReport] = []
        opens = self.panel.market["open"]
        up = self.panel.market.get("up_limit")
        down = self.panel.market.get("down_limit")

        _LOG.info(
            "开始回测 %s 天 × %s 只（面板 %s 天，预热 %s）",
            n_run,
            n_stocks,
            n_dates,
            t0,
        )
        last_w: Optional[np.ndarray] = None
        last_px = np.full(n_stocks, np.nan, dtype=np.float64)
        live_open = np.isfinite(opens) & (opens > 0)
        delist = np.asarray(getattr(self.panel, "delist_on", np.full(n_stocks, "")), dtype="U10")
        if delist.size != n_stocks:
            delist = np.full(n_stocks, "", dtype="U10")
        has_delist = delist != ""
        asofs = np.asarray(self.panel.dates, dtype="U10")
        expired = has_delist[None, :] & (asofs[:, None] >= delist[None, :])
        for k, t in enumerate(range(t0, n_dates)):
            if k == 0 or (k + 1) % 40 == 0 or k + 1 == n_run:
                _LOG.info("回测 %s/%s  %s", k + 1, n_run, self.panel.dates[t])
            px = opens[t]
            live_px = live_open[t]
            last_px = np.where(live_px, px, last_px)
            priced = np.isfinite(last_px) & (last_px > 0)
            holding = shares > _HOLD_EPS
            gone = holding & expired[t] & priced
            if np.any(gone):
                nav_before = float(cash) + float(
                    np.where(holding & priced, shares * last_px, 0.0).sum()
                )
                idx = np.flatnonzero(gone)
                proceeds = float((shares[gone] * last_px[gone]).sum())
                abs_n = abs(proceeds)
                fee = abs_n * float(getattr(self.executor, "commission", 0.0) or 0.0)
                fee += abs_n * float(getattr(self.executor, "stamp", 0.0) or 0.0)
                forced_fills.append(
                    FillReport(
                        date=self.panel.dates[t],
                        symbols=[str(self.panel.symbols[i]) for i in idx],
                        intended=np.zeros(idx.size, dtype=np.float64),
                        filled=-shares[idx].copy(),
                        price=last_px[idx].copy(),
                        reason=["delist"] * int(idx.size),
                        traded_notional=proceeds,
                        nav_before=nav_before,
                        fee=fee,
                    )
                )
                cash += proceeds - fee
                shares[gone] = 0.0
                last_w = None
                if nav_before > 0:
                    turnover[k] = float(turnover[k]) + proceeds / nav_before
                _LOG.info("退市，按最后价变现 %s 只 %s", int(gone.sum()), self.panel.dates[t])
            mark_px = np.where(live_px, px, last_px)
            marked = np.where(
                (shares > _HOLD_EPS) & np.isfinite(mark_px) & (mark_px > 0),
                shares * mark_px,
                0.0,
            )
            value = float(cash) + float(marked.sum())
            if value <= 0:
                value = 0.0
                pos = np.zeros(n_stocks, dtype=np.float64)
            else:
                pos = marked / value
            navs[k] = value
            cashes[k] = cash
            weights[k] = pos
            execute_on = self.panel.dates[t + 1] if t + 1 < n_dates else None
            ctx = self.panel.day_context(
                t,
                position=pos,
                cash=cash,
                value=value,
                execute_on=execute_on,
            )
            raw = strategy.score(ctx)
            target = self._as_target(raw, ctx.asof, execute_on, ctx.symbols, n_stocks)
            targets.append(None if raw is None else target)
            if execute_on is None:
                continue
            want = target.aligned(self.panel.symbols)
            if _hold_shares(shares, want, last_w):
                continue
            next_px = opens[t + 1]
            up_row = None if up is None else up[t + 1]
            down_row = None if down is None else down[t + 1]
            shares, cash, report = self.executor.fill(
                shares,
                cash,
                target,
                self.panel.symbols,
                next_px,
                up_limit=up_row,
                down_limit=down_row,
                date=execute_on,
                last_px=last_px,
            )
            last_w = want.copy()
            fills[k + 1] = report
            if report.nav_before > 0:
                turnover[k + 1] = report.traded_notional / report.nav_before

        return EngineResult(
            dates=list(self.panel.dates[t0:]),
            nav=navs,
            cash=cashes,
            weights=weights,
            turnover=turnover,
            targets=targets,
            fills=fills,
            forced_fills=forced_fills,
        )

    @staticmethod
    def _as_target(
        raw: Union[TargetHoldings, np.ndarray, Sequence[float], None],
        asof: str,
        execute_on: Optional[str],
        symbols: Sequence[str],
        n_stocks: int,
    ) -> TargetHoldings:
        if raw is None:
            return TargetHoldings.from_array(
                asof, execute_on, symbols, np.zeros(n_stocks, dtype=np.float64)
            )
        return Engine._coerce_target(raw, asof, execute_on, symbols)

    @staticmethod
    def _coerce_target(
        raw: Union[TargetHoldings, np.ndarray, Sequence[float]],
        asof: str,
        execute_on: Optional[str],
        symbols: Sequence[str],
    ) -> TargetHoldings:
        if isinstance(raw, TargetHoldings):
            if raw.asof != asof:
                raw = TargetHoldings(
                    asof=asof,
                    execute_on=execute_on if raw.execute_on is None else raw.execute_on,
                    weights=raw.weights,
                )
            elif raw.execute_on is None and execute_on is not None:
                raw = TargetHoldings(asof=raw.asof, execute_on=execute_on, weights=raw.weights)
            return raw
        return TargetHoldings.from_array(asof, execute_on, symbols, np.asarray(raw, dtype=np.float64))


def _hold_shares(shares: np.ndarray, want: np.ndarray, last_w: Optional[np.ndarray]) -> bool:
    """标的集合和目标权重都没变，且已经持有这组股票（或都已空仓）时，不因股价再调仓。"""
    holding = np.asarray(shares, dtype=np.float64) > _HOLD_EPS
    wanted = np.asarray(want, dtype=np.float64) > _HOLD_EPS
    if last_w is None:
        return (not bool(wanted.any())) and (not bool(holding.any()))
    if not np.allclose(want, last_w, atol=1e-12, rtol=0.0):
        return False
    return bool(np.array_equal(holding, wanted))
