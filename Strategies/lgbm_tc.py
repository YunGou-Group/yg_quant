#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LGBM 打分 + 经济调仓：预测换仓增益超过成本才动。

用法：--strategy lgbm_tc --factor a,b,-c
打分与 lgbm 相同（LambdaRank、blend、smooth），不改 lgbm.py。
默认每天检查；当前持仓相对新目标的
  增益 = (w_new - w_old) · μ ，μ = IĈ × σ̂ × z(score)
  成本 = 0.5‖w_new - w_old‖₁ × --tc
仅当增益 ≥ hurdle × 成本时换仓。IC 衰减未翻负时持有仍有正期望，不等于调仓动作为负。
首次建仓、持有超过 --max-hold、或持仓落在股票池外，强制换仓。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import cli_field, n_field, rebalance_field
from Strategies._factor_combo import FactorLeg, cross_section_z, equal_weight_scores, parse_factor_list
from Strategies.lgbm import (
    DEFAULT_BLEND,
    DEFAULT_HORIZON,
    DEFAULT_LOOKBACK,
    DEFAULT_N,
    DEFAULT_REFIT,
    DEFAULT_SMOOTH,
    LgbmWeightedTopK,
    _day_stride,
)
from Strategies.rebalance_schedule import RebalanceSpec, parse_rebalance, spec_from_cli

DEFAULT_TC = 0.003
DEFAULT_HURDLE = 1.0
DEFAULT_MIN_HOLD = 1
DEFAULT_MAX_HOLD = 60
_MAX_SCALE_GROUPS = 80
_EPS = 1e-12


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 8:
        return float("nan")
    rx = np.argsort(np.argsort(x, kind="mergesort")).astype(np.float64)
    ry = np.argsort(np.argsort(y, kind="mergesort")).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if den < _EPS:
        return float("nan")
    return float(np.sum(rx * ry) / den)


def estimate_ic_sigma(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
) -> Tuple[float, float]:
    """用已实现 t→t+H 收益估 RankIC 与截面收益标准差，只读 asof 之前。"""
    close = np.asarray(close_hist, dtype=np.float64)
    mask = np.asarray(mask_hist, dtype=bool)
    step = _day_stride(horizon)
    last = int(lookback) - 1
    ics: List[float] = []
    sigmas: List[float] = []
    i = last
    while i >= 0 and len(ics) < _MAX_SCALE_GROUPS:
        j = i + int(horizon)
        c0 = close[i]
        c1 = close[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            y = c1 / c0 - 1.0
        y = np.where(np.isfinite(c0) & (c0 > 0) & np.isfinite(c1) & (c1 > 0), y, np.nan)
        uni = mask[i]
        day_panels = [np.asarray(panel[i], dtype=np.float64) for panel in factor_hist]
        eq = equal_weight_scores(day_panels, signs, uni)
        ok = np.asarray(uni, dtype=bool) & np.isfinite(y) & np.isfinite(eq)
        if int(ok.sum()) < 30:
            i -= step
            continue
        ic = _spearman(eq[ok], y[ok])
        sig = float(np.std(y[ok], ddof=1))
        if np.isfinite(ic) and np.isfinite(sig) and sig > _EPS:
            ics.append(ic)
            sigmas.append(sig)
        i -= step
    if not ics:
        return 0.0, 0.0
    return float(np.median(np.asarray(ics, dtype=np.float64))), float(
        np.median(np.asarray(sigmas, dtype=np.float64))
    )


def expected_mu(score: np.ndarray, tradable: np.ndarray, ic_hat: float, ret_std: float) -> np.ndarray:
    z = cross_section_z(np.asarray(score, dtype=np.float64), tradable)
    mu = np.zeros(z.shape, dtype=np.float64)
    ok = np.asarray(tradable, dtype=bool) & np.isfinite(z)
    mu[ok] = float(ic_hat) * float(ret_std) * z[ok]
    return mu


def rebalance_gain_cost(
    w_old: np.ndarray,
    w_new: np.ndarray,
    mu: np.ndarray,
    tc: float,
) -> Tuple[float, float, float]:
    """返回 (期望超额增益, 单边换手, 双边成本)。权重按多头归一。"""
    old = np.asarray(w_old, dtype=np.float64)
    new = np.asarray(w_new, dtype=np.float64)
    old = np.where(np.isfinite(old), np.maximum(old, 0.0), 0.0)
    new = np.where(np.isfinite(new), np.maximum(new, 0.0), 0.0)
    s_old = float(old.sum())
    s_new = float(new.sum())
    if s_old > _EPS:
        old = old / s_old
    if s_new > _EPS:
        new = new / s_new
    delta = new - old
    gain = float(np.dot(delta, np.where(np.isfinite(mu), mu, 0.0)))
    turn = 0.5 * float(np.abs(delta).sum())
    cost = turn * float(tc)
    return gain, turn, cost


def worth_rebalance(
    w_old: np.ndarray,
    w_new: np.ndarray,
    mu: np.ndarray,
    tc: float,
    hurdle: float,
) -> bool:
    gain, turn, cost = rebalance_gain_cost(w_old, w_new, mu, tc)
    if turn < 1e-8:
        return False
    return gain + 1e-15 >= float(hurdle) * cost


class LgbmTcTopK(LgbmWeightedTopK):
    name = "lgbm_tc"

    @classmethod
    def from_cli(cls, args, allocator):
        if not args.factor:
            raise SystemExit("lgbm_tc 需要 --factor，逗号分隔，如 a,b,c")
        try:
            legs = parse_factor_list(args.factor)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        n = args.n if args.n is not None else DEFAULT_N
        spec = spec_from_cli(getattr(args, "rebalance", None) or "daily")
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        refit = int(
            getattr(args, "refit", None)
            if getattr(args, "refit", None) is not None
            else DEFAULT_REFIT
        )
        blend = float(
            getattr(args, "blend", None) if getattr(args, "blend", None) is not None else DEFAULT_BLEND
        )
        smooth = float(
            getattr(args, "smooth", None) if getattr(args, "smooth", None) is not None else DEFAULT_SMOOTH
        )
        tc = float(getattr(args, "tc", None) if getattr(args, "tc", None) is not None else DEFAULT_TC)
        hurdle = float(
            getattr(args, "hurdle", None) if getattr(args, "hurdle", None) is not None else DEFAULT_HURDLE
        )
        min_hold = int(
            getattr(args, "min_hold", None)
            if getattr(args, "min_hold", None) is not None
            else DEFAULT_MIN_HOLD
        )
        max_hold = int(
            getattr(args, "max_hold", None)
            if getattr(args, "max_hold", None) is not None
            else DEFAULT_MAX_HOLD
        )
        if lookback < 60:
            raise SystemExit("lgbm_tc --lookback 至少为 60（默认 252）")
        if horizon < 1:
            raise SystemExit("lgbm_tc --horizon 至少为 1（默认 20）")
        if refit < 1:
            raise SystemExit("lgbm_tc --refit 至少为 1（默认 20）")
        if not 0.0 <= blend <= 1.0:
            raise SystemExit("lgbm_tc --blend 须在 0 与 1 之间")
        if not 0.0 <= smooth < 1.0:
            raise SystemExit("lgbm_tc --smooth 须在 [0, 1)")
        if tc < 0.0:
            raise SystemExit("lgbm_tc --tc 须 ≥ 0（默认 0.003，千三双边）")
        if hurdle < 0.0:
            raise SystemExit("lgbm_tc --hurdle 须 ≥ 0（默认 1，增益须盖过成本）")
        if min_hold < 1:
            raise SystemExit("lgbm_tc --min-hold 至少为 1")
        if max_hold < min_hold:
            raise SystemExit("lgbm_tc --max-hold 须 ≥ --min-hold")
        return cls(
            legs,
            n=n,
            allocator=allocator,
            rebalance=spec,
            lookback=lookback,
            horizon=horizon,
            refit=refit,
            blend=blend,
            smooth=smooth,
            tc=tc,
            hurdle=hurdle,
            min_hold=min_hold,
            max_hold=max_hold,
        )

    @classmethod
    def cli_fields(cls):
        return [
            cli_field("factor", "因子列表", "str", required=True, placeholder="a,b,-c"),
            n_field(DEFAULT_N),
            rebalance_field("daily", placeholder="daily / weekly / 5"),
            cli_field("lookback", "LGBM回看天数", "int", default=DEFAULT_LOOKBACK),
            cli_field("horizon", "LGBM持有期，亦为μ的期限", "int", default=DEFAULT_HORIZON),
            cli_field("refit", "同一模型沿用交易日数", "int", default=DEFAULT_REFIT),
            cli_field("blend", "树分权重", "float", default=DEFAULT_BLEND),
            cli_field("smooth", "上期分数平滑", "float", default=DEFAULT_SMOOTH),
            cli_field("tc", "双边费率，乘在单边换手上", "float", default=DEFAULT_TC),
            cli_field("hurdle", "增益/成本门槛", "float", default=DEFAULT_HURDLE),
            cli_field("min_hold", "两次调仓最少间隔交易日", "int", default=DEFAULT_MIN_HOLD),
            cli_field("max_hold", "超过则强制调仓", "int", default=DEFAULT_MAX_HOLD),
        ]

    @classmethod
    def run_tag(cls, args) -> str:
        legs = parse_factor_list(args.factor)
        parts = [("m" if s < 0 else "") + n for n, s in legs]
        tag = "lgbmtc_" + "_".join(parts)
        spec = parse_rebalance(getattr(args, "rebalance", None) or "daily")
        tag += spec.tag_suffix()
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        refit = DEFAULT_REFIT if getattr(args, "refit", None) is None else int(args.refit)
        blend = getattr(args, "blend", None)
        b = DEFAULT_BLEND if blend is None else float(blend)
        tc = DEFAULT_TC if getattr(args, "tc", None) is None else float(args.tc)
        hurdle = DEFAULT_HURDLE if getattr(args, "hurdle", None) is None else float(args.hurdle)
        max_hold = DEFAULT_MAX_HOLD if getattr(args, "max_hold", None) is None else int(args.max_hold)
        tag += f"_lgb{lookback}h{horizon}rf{refit}rk{int(round(b * 100))}"
        tag += f"_tc{int(round(tc * 10000))}_hu{int(round(hurdle * 100))}_mh{max_hold}"
        return tag

    def __init__(
        self,
        factors: Sequence[FactorLeg] | Sequence[str],
        n: int = DEFAULT_N,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = "daily",
        lookback: int = DEFAULT_LOOKBACK,
        horizon: int = DEFAULT_HORIZON,
        refit: int = DEFAULT_REFIT,
        min_rows: int = 400,
        blend: float = DEFAULT_BLEND,
        smooth: float = DEFAULT_SMOOTH,
        tc: float = DEFAULT_TC,
        hurdle: float = DEFAULT_HURDLE,
        min_hold: int = DEFAULT_MIN_HOLD,
        max_hold: int = DEFAULT_MAX_HOLD,
    ):
        super().__init__(
            factors,
            n=n,
            allocator=allocator or EqualWeight(),
            rebalance=rebalance,
            lookback=lookback,
            horizon=horizon,
            refit=refit,
            min_rows=min_rows,
            blend=blend,
            smooth=smooth,
        )
        self.tc = float(max(0.0, tc))
        self.hurdle = float(max(0.0, hurdle))
        self.min_hold = max(1, int(min_hold))
        self.max_hold = max(self.min_hold, int(max_hold))
        self._ic_hat = 0.0
        self._ret_std = 0.0
        self._scale_t: Optional[int] = None
        self._last_trade_t: Optional[int] = None

    def _refresh_scale(self, ctx: DayContext) -> None:
        if self._scale_t is not None and not self._need_refit(ctx._t) and self._scale_t == self._fitted_t:
            return
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
        self._ic_hat, self._ret_std = estimate_ic_sigma(
            factor_hist,
            signs,
            close,
            mask,
            lookback=self.lookback,
            horizon=self.horizon,
        )
        self._scale_t = int(ctx._t)

    def _candidate_weights(self, ctx: DayContext, values: np.ndarray) -> np.ndarray:
        tradable = ctx.universe() & np.isfinite(values)
        ranked = np.where(tradable, values, -np.inf)
        k = int(min(self.n, int(tradable.sum())))
        n_sym = len(ctx.symbols)
        if k <= 0:
            return np.zeros(n_sym, dtype=np.float64)
        pick = np.argpartition(ranked, -k)[-k:]
        pick = pick[tradable[pick]]
        return np.asarray(self.allocator.size(ctx, pick), dtype=np.float64)

    def _current_weights(self, ctx: DayContext) -> np.ndarray:
        w = np.asarray(ctx.position, dtype=np.float64)
        w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
        return w

    def _held_days(self, t: int) -> int:
        if self._last_trade_t is None:
            return 10**9
        return int(t) - int(self._last_trade_t)

    def score(self, ctx: DayContext) -> TargetHoldings:
        calendar = self._gate.skip(ctx)
        if calendar is not None:
            return calendar

        values = self._lgbm_score(ctx)
        self._refresh_scale(ctx)
        w_new = self._candidate_weights(ctx, values)
        w_old = self._current_weights(ctx)
        uni = ctx.universe()
        invested = float(w_old.sum())
        held = self._held_days(ctx._t)
        outside = bool(np.any((~uni) & (w_old > 1e-9)))
        empty = invested <= 1e-9
        stale = held >= int(self.max_hold)
        too_soon = (self._last_trade_t is not None) and (held < int(self.min_hold))
        tradable = uni & np.isfinite(values)
        mu = expected_mu(values, tradable, self._ic_hat, self._ret_std)

        trade = empty or outside or stale
        if not trade and not too_soon:
            trade = worth_rebalance(w_old, w_new, mu, self.tc, self.hurdle)
        if not trade:
            keep = np.where(uni, w_old, 0.0)
            if float(keep.sum()) > 1e-9:
                keep = keep / float(keep.sum())
            else:
                keep = np.zeros(len(ctx.symbols), dtype=np.float64)
            return self._gate.remember(
                TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, keep)
            )
        self._last_trade_t = int(ctx._t)
        return self._gate.remember(
            TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, w_new)
        )
