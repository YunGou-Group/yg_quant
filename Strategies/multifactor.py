#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多因子 OLS 合成 + Top N。

用法：--strategy multifactor --factor a,b,c
因子名前加 '-' 表示取负，如 alpha001,-turnover。

每个调仓日：
1. 回看 lookback 个已实现持有期，对每日截面做 OLS：fwd_ret ~ 1 + z(factors)
2. 将各日因子系数取均值，得到权重 w
3. 当日得分 = z(factors) · w，再选 Top N

远期收益用收盘价：r_t = close[t+horizon]/close[t] - 1，仅用 asof 及之前已实现的收益，无前视。
样本不足时回退到等权 z-score。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import Strategy
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli

# (factor_name, sign)  sign ∈ {+1, -1}
FactorLeg = Tuple[str, float]

DEFAULT_LOOKBACK = 60
DEFAULT_HORIZON = 5
DEFAULT_MIN_OBS = 30


def parse_factor_list(text: str) -> List[FactorLeg]:
    """解析 'a,b,-c' → [('a',1),('b',1),('c',-1)]。"""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("需要至少一个因子，例如 --factor a,b,c")
    legs: List[FactorLeg] = []
    seen = set()
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        sign = 1.0
        if item.startswith("-"):
            sign = -1.0
            item = item[1:].strip()
        elif item.startswith("+"):
            item = item[1:].strip()
        if not item:
            raise ValueError(f"无效因子项: {token!r}")
        if item in seen:
            raise ValueError(f"重复因子: {item}")
        seen.add(item)
        legs.append((item, sign))
    if not legs:
        raise ValueError("需要至少一个因子，例如 --factor a,b,c")
    return legs


def cross_section_z(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """在 mask 有效样本上做截面 z-score；无效处置 NaN。"""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    sel = np.asarray(mask, dtype=bool) & np.isfinite(values)
    n = int(sel.sum())
    if n < 2:
        if n == 1:
            out[sel] = 0.0
        return out
    x = values[sel]
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if sd < 1e-12:
        out[sel] = 0.0
    else:
        out[sel] = (x - mu) / sd
    return out


def equal_weight_scores(
    panels: Sequence[np.ndarray],
    signs: Sequence[float],
    universe: np.ndarray,
) -> np.ndarray:
    """各因子截面 z 后等权 nanmean（OLS 失败时的回退）。"""
    if not panels:
        return np.full(universe.shape, np.nan, dtype=np.float64)
    zs = []
    for panel, sign in zip(panels, signs):
        raw = np.asarray(panel, dtype=np.float64) * float(sign)
        zs.append(cross_section_z(raw, universe))
    stacked = np.vstack(zs)
    count = np.sum(np.isfinite(stacked), axis=0)
    total = np.nansum(np.where(np.isfinite(stacked), stacked, 0.0), axis=0)
    score = np.full(universe.shape, np.nan, dtype=np.float64)
    ok = count > 0
    score[ok] = total[ok] / count[ok]
    return score


def cs_ols_beta(y: np.ndarray, x: np.ndarray, min_obs: int) -> Optional[np.ndarray]:
    """截面 OLS：y = a + X β + ε，返回 β（不含截距）。"""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    n = int(valid.sum())
    if n < max(int(min_obs), x.shape[1] + 2):
        return None
    xv = np.column_stack([np.ones(n, dtype=np.float64), x[valid]])
    yv = y[valid]
    try:
        beta, _residuals, rank, _s = np.linalg.lstsq(xv, yv, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if int(rank) < xv.shape[1]:
        return None
    return np.asarray(beta[1:], dtype=np.float64)


def combine_scores_ols(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    min_obs: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """用历史窗口 Fama–MacBeth OLS 估权重，对末日截面打分。

    factor_hist[k] / close_hist / mask_hist 形状均为 (lookback+horizon, n_stocks)，
    最后一行对应 asof。返回 (score, mean_beta)；估权失败时 score 为等权回退且 beta 为 None。
    """
    need = int(lookback) + int(horizon)
    n_factors = len(factor_hist)
    if n_factors == 0:
        return np.full(close_hist.shape[1], np.nan, dtype=np.float64), None
    for panel in factor_hist:
        if np.asarray(panel).shape[0] < need:
            today = [np.asarray(p)[-1] for p in factor_hist]
            return equal_weight_scores(today, signs, mask_hist[-1]), None

    close = np.asarray(close_hist, dtype=np.float64)
    mask = np.asarray(mask_hist, dtype=bool)
    betas = []
    for i in range(int(lookback)):
        j = i + int(horizon)
        c0 = close[i]
        c1 = close[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            y = c1 / c0 - 1.0
        y = np.where(np.isfinite(c0) & (c0 > 0) & np.isfinite(c1) & (c1 > 0), y, np.nan)
        uni = mask[i]
        cols = []
        for panel, sign in zip(factor_hist, signs):
            raw = np.asarray(panel[i], dtype=np.float64) * float(sign)
            cols.append(cross_section_z(raw, uni))
        x = np.column_stack(cols)
        row_ok = uni & np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        beta = cs_ols_beta(y[row_ok], x[row_ok], min_obs=min_obs)
        if beta is not None:
            betas.append(beta)

    today_uni = mask[-1]
    today_panels = [np.asarray(p)[-1] for p in factor_hist]
    if len(betas) < max(5, int(lookback) // 10):
        return equal_weight_scores(today_panels, signs, today_uni), None

    w = np.nanmean(np.vstack(betas), axis=0)
    if not np.all(np.isfinite(w)):
        return equal_weight_scores(today_panels, signs, today_uni), None

    z_cols = []
    for panel, sign in zip(today_panels, signs):
        raw = np.asarray(panel, dtype=np.float64) * float(sign)
        z_cols.append(cross_section_z(raw, today_uni))
    z = np.column_stack(z_cols)
    score = np.full(today_uni.shape, np.nan, dtype=np.float64)
    ok = today_uni & np.all(np.isfinite(z), axis=1)
    score[ok] = z[ok] @ w
    return score, w


# 兼容旧名
combine_scores = equal_weight_scores


class MultiFactorTopK(Strategy):
    name = "multifactor"

    @classmethod
    def from_cli(cls, args, allocator):
        if not args.factor:
            raise SystemExit("multifactor 需要 --factor，逗号分隔，如 a,b,c")
        try:
            legs = parse_factor_list(args.factor)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        n = args.n if args.n is not None else 50
        spec = spec_from_cli(getattr(args, "rebalance", None) or "daily")
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        if lookback < 5:
            raise SystemExit("multifactor --lookback 至少为 5")
        if horizon < 1:
            raise SystemExit("multifactor --horizon 至少为 1")
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
        tag = "multifactor_" + "_".join(parts)
        spec = parse_rebalance(getattr(args, "rebalance", None) or "daily")
        tag += spec.tag_suffix()
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        tag += f"_ols{lookback}h{horizon}"
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
    ):
        legs: List[FactorLeg] = []
        for item in factors:
            if isinstance(item, str):
                legs.extend(parse_factor_list(item))
            else:
                name, sign = item
                legs.append((str(name), float(sign)))
        if not legs:
            raise ValueError("multifactor 至少需要一个因子")
        self.legs = legs
        self.n = max(1, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = (
            parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        )
        self.lookback = max(5, int(lookback))
        self.horizon = max(1, int(horizon))
        self.min_obs = max(len(legs) + 2, int(min_obs))
        self._gate = RebalanceGate(self.rebalance)
        self.last_beta: Optional[np.ndarray] = None

    @property
    def factor_names(self) -> List[str]:
        return [name for name, _ in self.legs]

    def _ols_score(self, ctx: DayContext) -> np.ndarray:
        need = self.lookback + self.horizon
        close = ctx.history("close", need)
        t1 = int(ctx._t) + 1
        t0 = max(0, t1 - need)
        mask = np.asarray(ctx._store.mask[t0:t1], dtype=bool)
        if mask.shape[0] < need:
            pad = np.zeros((need - mask.shape[0], mask.shape[1]), dtype=bool)
            mask = np.vstack([pad, mask])
        signs = [sign for _, sign in self.legs]
        factor_hist = []
        for name, _ in self.legs:
            factor_hist.append(ctx.history(name, need))
        score, beta = combine_scores_ols(
            factor_hist,
            signs,
            close,
            mask,
            lookback=self.lookback,
            horizon=self.horizon,
            min_obs=self.min_obs,
        )
        self.last_beta = beta
        return score

    def score(self, ctx: DayContext) -> TargetHoldings:
        held = self._gate.skip(ctx)
        if held is not None:
            return held

        values = self._ols_score(ctx)
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
