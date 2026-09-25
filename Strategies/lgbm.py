#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""滚动 LightGBM 合成 + Top N。

用法：--strategy lgbm --factor a,b,c
因子名前加 '-' 表示取负。

每个调仓日：
1. 回看 lookback 个已实现持有期，用当日截面 z(因子) 预测已实现远期收益
   （收益先在当日截面去均值，只学相对排序）
2. 用拟合模型对 asof 截面打分，再选 Top N

远期收益 r_t = close[t+horizon]/close[t] - 1，只用 asof 及之前已实现的收益。
样本不足或未安装 lightgbm 时回退等权 z-score。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import Strategy, cli_field, n_field, rebalance_field
from Strategies._factor_combo import (
    DEFAULT_HORIZON,
    DEFAULT_LOOKBACK,
    FactorLeg,
    cross_section_z,
    equal_weight_scores,
    parse_factor_list,
)
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli

logger = logging.getLogger("Strategies.lgbm")

DEFAULT_N = 50
DEFAULT_MIN_ROWS = 400
DEFAULT_MAX_ROWS = 120_000

_LGB_PARAMS = {
    "n_estimators": 80,
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_child_samples": 200,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": -1,
}


def _try_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    return lgb


def stack_realized_xy(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """拼已实现样本。X 只用到 index lookback-1，不含 asof 所在的最后 horizon 行。"""
    close = np.asarray(close_hist, dtype=np.float64)
    mask = np.asarray(mask_hist, dtype=bool)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
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
        if int(row_ok.sum()) < 8:
            continue
        yy = y[row_ok]
        yy = yy - float(np.mean(yy))
        xs.append(x[row_ok])
        ys.append(yy)
    if not xs:
        return np.empty((0, len(factor_hist)), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.vstack(xs), np.concatenate(ys)


def _subsample(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    if n <= max_rows:
        return x, y
    rng = np.random.default_rng(int(seed) % (2**31))
    pick = rng.choice(n, size=int(max_rows), replace=False)
    return x[pick], y[pick]


def combine_scores_lgbm(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> Tuple[np.ndarray, Optional[object]]:
    """用已实现窗口训练 LightGBM，对末日截面打分。失败则等权回退。"""
    need = int(lookback) + int(horizon)
    today_uni = np.asarray(mask_hist[-1], dtype=bool)
    today_panels = [np.asarray(p)[-1] for p in factor_hist]
    if not factor_hist:
        return np.full(close_hist.shape[1], np.nan, dtype=np.float64), None
    for panel in factor_hist:
        if np.asarray(panel).shape[0] < need:
            return equal_weight_scores(today_panels, signs, today_uni), None

    lgb = _try_lightgbm()
    if lgb is None:
        logger.warning("未安装 lightgbm，lgbm 策略回退等权 z-score")
        return equal_weight_scores(today_panels, signs, today_uni), None

    x, y = stack_realized_xy(
        factor_hist,
        signs,
        close_hist,
        mask_hist,
        lookback=lookback,
        horizon=horizon,
    )
    need_rows = max(int(min_rows), 80 * len(factor_hist))
    if x.shape[0] < need_rows:
        return equal_weight_scores(today_panels, signs, today_uni), None
    x, y = _subsample(x, y, DEFAULT_MAX_ROWS, seed=42 + int(lookback) + int(horizon))

    model = lgb.LGBMRegressor(**_LGB_PARAMS)
    try:
        model.fit(x, y)
    except Exception:
        logger.exception("LightGBM 拟合失败，回退等权")
        return equal_weight_scores(today_panels, signs, today_uni), None

    z_cols = []
    for panel, sign in zip(today_panels, signs):
        raw = np.asarray(panel, dtype=np.float64) * float(sign)
        z_cols.append(cross_section_z(raw, today_uni))
    z = np.column_stack(z_cols)
    score = np.full(today_uni.shape, np.nan, dtype=np.float64)
    ok = today_uni & np.all(np.isfinite(z), axis=1)
    if int(ok.sum()) == 0:
        return equal_weight_scores(today_panels, signs, today_uni), None
    pred = np.asarray(model.predict(z[ok]), dtype=np.float64)
    score[ok] = pred
    return score, model


class LgbmWeightedTopK(Strategy):
    name = "lgbm"

    @classmethod
    def from_cli(cls, args, allocator):
        if not args.factor:
            raise SystemExit("lgbm 需要 --factor，逗号分隔，如 a,b,c")
        try:
            legs = parse_factor_list(args.factor)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        n = args.n if args.n is not None else DEFAULT_N
        spec = spec_from_cli(getattr(args, "rebalance", None) or "20")
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        if lookback < 20:
            raise SystemExit("lgbm --lookback 至少为 20")
        if horizon < 1:
            raise SystemExit("lgbm --horizon 至少为 1")
        return cls(
            legs,
            n=n,
            allocator=allocator,
            rebalance=spec,
            lookback=lookback,
            horizon=horizon,
        )

    @classmethod
    def cli_fields(cls):
        return [
            cli_field("factor", "因子列表", "str", required=True, placeholder="a,b,-c"),
            n_field(DEFAULT_N),
            rebalance_field("20", placeholder="daily / weekly / 20"),
            cli_field("lookback", "LGBM回看天数", "int", default=DEFAULT_LOOKBACK),
            cli_field("horizon", "LGBM持有期", "int", default=DEFAULT_HORIZON),
        ]

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
        tag = "lgbm_" + "_".join(parts)
        spec = parse_rebalance(getattr(args, "rebalance", None) or "20")
        tag += spec.tag_suffix()
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        tag += f"_lgb{lookback}h{horizon}"
        return tag

    def __init__(
        self,
        factors: Sequence[FactorLeg] | Sequence[str],
        n: int = DEFAULT_N,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = "20",
        lookback: int = DEFAULT_LOOKBACK,
        horizon: int = DEFAULT_HORIZON,
        min_rows: int = DEFAULT_MIN_ROWS,
    ):
        legs: List[FactorLeg] = []
        for item in factors:
            if isinstance(item, str):
                legs.extend(parse_factor_list(item))
            else:
                name, sign = item
                legs.append((str(name), float(sign)))
        if not legs:
            raise ValueError("lgbm 至少需要一个因子")
        self.legs = legs
        self.n = max(1, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = (
            parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        )
        self.lookback = max(20, int(lookback))
        self.horizon = max(1, int(horizon))
        self.min_rows = max(80, int(min_rows))
        self._gate = RebalanceGate(self.rebalance)
        self.last_model: Optional[object] = None

    @property
    def factor_names(self) -> List[str]:
        return [name for name, _ in self.legs]

    def _lgbm_score(self, ctx: DayContext) -> np.ndarray:
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
        score, model = combine_scores_lgbm(
            factor_hist,
            signs,
            close,
            mask,
            lookback=self.lookback,
            horizon=self.horizon,
            min_rows=self.min_rows,
        )
        self.last_model = model
        return score

    def score(self, ctx: DayContext) -> TargetHoldings:
        held = self._gate.skip(ctx)
        if held is not None:
            return held

        values = self._lgbm_score(ctx)
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
