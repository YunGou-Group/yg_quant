#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多因子滚动 ICIR 加权 + Top N。

用法：--strategy icir --factor a,b,c
因子名前加 '-' 表示取负。

每个调仓日：
1. 回看 lookback 个已实现持有期，逐日算 RankIC（因子 vs horizon 日收盘收益）
2. ICIR_k = mean(IC)/std(IC)；负 ICIR 置 0 后归一化
3. 当日得分 = Σ w_k · z(factor_k)，再选 Top N

只用 asof 及之前已实现的 t→t+H 收益，无前视。样本不足时回退等权 z-score。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import Strategy
from Strategies.multifactor import (
    DEFAULT_HORIZON,
    DEFAULT_LOOKBACK,
    FactorLeg,
    cross_section_z,
    equal_weight_scores,
    parse_factor_list,
)
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli

DEFAULT_MIN_OBS = 30
DEFAULT_MIN_IC_DAYS = 20


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.float64)
    return ranks


def spearman_ic(factor: np.ndarray, returns: np.ndarray, min_obs: int) -> float:
    valid = np.isfinite(factor) & np.isfinite(returns)
    n = int(valid.sum())
    if n < max(3, int(min_obs)):
        return float("nan")
    rx = _rank(np.asarray(factor[valid], dtype=np.float64))
    ry = _rank(np.asarray(returns[valid], dtype=np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if den < 1e-15:
        return float("nan")
    return float(np.sum(rx * ry) / den)


def icir_weights(ics: np.ndarray, min_days: int) -> Optional[np.ndarray]:
    """ics: (n_days, n_factors)。负 ICIR 置 0；全非正则 None（调用方回退等权）。"""
    mat = np.asarray(ics, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat[:, None]
    n_f = mat.shape[1]
    raw = np.zeros(n_f, dtype=np.float64)
    need = max(2, int(min_days))
    for k in range(n_f):
        col = mat[:, k]
        clean = col[np.isfinite(col)]
        if clean.size < need:
            continue
        mu = float(np.mean(clean))
        sd = float(np.std(clean, ddof=1))
        if not np.isfinite(sd) or sd < 1e-12:
            # IC 几乎常数：稳定为正则给很大 ICIR，否则不参与
            if abs(mu) > 1e-12:
                raw[k] = 1e6 if mu > 0 else -1e6
            continue
        raw[k] = mu / sd
    clipped = np.maximum(raw, 0.0)
    total = float(clipped.sum())
    if total <= 0:
        return None
    return clipped / total


def combine_scores_icir(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    min_obs: int,
    min_ic_days: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    need = int(lookback) + int(horizon)
    n_factors = len(factor_hist)
    if n_factors == 0:
        return np.full(close_hist.shape[1], np.nan, dtype=np.float64), None
    today_uni = np.asarray(mask_hist[-1], dtype=bool)
    today_panels = [np.asarray(p)[-1] for p in factor_hist]
    for panel in factor_hist:
        if np.asarray(panel).shape[0] < need:
            return equal_weight_scores(today_panels, signs, today_uni), None

    close = np.asarray(close_hist, dtype=np.float64)
    mask = np.asarray(mask_hist, dtype=bool)
    ics = np.full((int(lookback), n_factors), np.nan, dtype=np.float64)
    for i in range(int(lookback)):
        j = i + int(horizon)
        c0 = close[i]
        c1 = close[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            y = c1 / c0 - 1.0
        y = np.where(np.isfinite(c0) & (c0 > 0) & np.isfinite(c1) & (c1 > 0), y, np.nan)
        uni = mask[i]
        for k, (panel, sign) in enumerate(zip(factor_hist, signs)):
            raw = np.asarray(panel[i], dtype=np.float64) * float(sign)
            z = cross_section_z(raw, uni)
            ics[i, k] = spearman_ic(z, y, min_obs=min_obs)

    w = icir_weights(ics, min_ic_days)
    if w is None:
        return equal_weight_scores(today_panels, signs, today_uni), None

    z_cols = []
    for panel, sign in zip(today_panels, signs):
        raw = np.asarray(panel, dtype=np.float64) * float(sign)
        z_cols.append(cross_section_z(raw, today_uni))
    z = np.column_stack(z_cols)
    score = np.full(today_uni.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(z)
    w_row = np.where(finite, w[None, :], 0.0)
    w_sum = w_row.sum(axis=1)
    contrib = np.where(finite, z * w[None, :], 0.0).sum(axis=1)
    ok = today_uni & (w_sum > 0)
    score[ok] = contrib[ok] / w_sum[ok]
    return score, w


class IcirWeightedTopK(Strategy):
    name = "icir"

    @classmethod
    def from_cli(cls, args, allocator):
        if not args.factor:
            raise SystemExit("icir 需要 --factor，逗号分隔，如 a,b,c")
        try:
            legs = parse_factor_list(args.factor)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        n = args.n if args.n is not None else 50
        spec = spec_from_cli(getattr(args, "rebalance", None) or "daily")
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        if lookback < 10:
            raise SystemExit("icir --lookback 至少为 10")
        if horizon < 1:
            raise SystemExit("icir --horizon 至少为 1")
        return cls(
            legs,
            n=n,
            allocator=allocator,
            rebalance=spec,
            lookback=lookback,
            horizon=horizon,
        )

    @classmethod
    def panel_kwargs(cls, args) -> dict:
        legs = parse_factor_list(args.factor)
        return {"factors": [name for name, _sign in legs]}

    @classmethod
    def warmup(cls, args) -> int:
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        return lookback + horizon + 5

    @classmethod
    def run_tag(cls, args) -> str:
        legs = parse_factor_list(args.factor)
        parts = [("m" if s < 0 else "") + n for n, s in legs]
        tag = "icir_" + "_".join(parts)
        spec = parse_rebalance(getattr(args, "rebalance", None) or "daily")
        tag += spec.tag_suffix()
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        tag += f"_icir{lookback}h{horizon}"
        return tag

    def __init__(
        self,
        factors: Sequence[FactorLeg] | Sequence[str],
        n: int = 50,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = "daily",
        lookback: int = DEFAULT_LOOKBACK,
        horizon: int = DEFAULT_HORIZON,
        min_obs: int = DEFAULT_MIN_OBS,
        min_ic_days: int = DEFAULT_MIN_IC_DAYS,
    ):
        legs: List[FactorLeg] = []
        for item in factors:
            if isinstance(item, str):
                legs.extend(parse_factor_list(item))
            else:
                name, sign = item
                legs.append((str(name), float(sign)))
        if not legs:
            raise ValueError("icir 至少需要一个因子")
        self.legs = legs
        self.n = max(1, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = (
            parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        )
        self.lookback = max(10, int(lookback))
        self.horizon = max(1, int(horizon))
        self.min_obs = max(10, int(min_obs))
        self.min_ic_days = max(5, int(min_ic_days))
        self._gate = RebalanceGate(self.rebalance)
        self.last_weights: Optional[np.ndarray] = None

    @property
    def factor_names(self) -> List[str]:
        return [name for name, _ in self.legs]

    def _icir_score(self, ctx: DayContext) -> np.ndarray:
        need = self.lookback + self.horizon
        close = ctx.history("close", need)
        t1 = int(ctx._t) + 1
        t0 = max(0, t1 - need)
        mask = np.asarray(ctx._store.mask[t0:t1], dtype=bool)
        if mask.shape[0] < need:
            pad = np.zeros((need - mask.shape[0], mask.shape[1]), dtype=bool)
            mask = np.vstack([pad, mask])
        signs = [sign for _, sign in self.legs]
        factor_hist = [ctx.history(name, need) for name, _ in self.legs]
        score, weights = combine_scores_icir(
            factor_hist,
            signs,
            close,
            mask,
            lookback=self.lookback,
            horizon=self.horizon,
            min_obs=self.min_obs,
            min_ic_days=self.min_ic_days,
        )
        self.last_weights = weights
        return score

    def score(self, ctx: DayContext) -> TargetHoldings:
        held = self._gate.skip(ctx)
        if held is not None:
            return held

        values = self._icir_score(ctx)
        tradable = ctx.universe() & np.isfinite(values)
        ranked = np.where(tradable, values, -np.inf)
        k = int(min(self.n, int(tradable.sum())))
        if k <= 0:
            weights = np.zeros(len(ctx.symbols))
            return self._gate.remember(
                TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
            )
        pick = np.argpartition(ranked, -k)[-k:]
        pick = pick[tradable[pick]]
        weights = self.allocator.size(ctx, pick)
        return self._gate.remember(
            TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
        )
