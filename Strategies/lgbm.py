#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""滚动 LightGBM 排序合成 + Top N。

用法：--strategy lgbm --factor a,b,c
因子名前加 '-' 表示取负。

每个调仓日：
1. 回看 lookback，按约 horizon/4 日抽样已实现截面，减轻 20 日收益重叠
2. 标签是截面收益五档相关度（0–4），用 LambdaRank 学排序，不是个股收益 MSE
3. 预测分截面 z 后与等权 z 按 blend 混合，再与上期分数平滑，压换手

只用 asof 及之前已实现的 t→t+H 收益。样本不足或未安装 lightgbm 时回退等权 z-score。
默认 lookback=252、horizon=20。
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
    FactorLeg,
    cross_section_z,
    equal_weight_scores,
    parse_factor_list,
)
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli

logger = logging.getLogger("Strategies.lgbm")

DEFAULT_N = 50
DEFAULT_LOOKBACK = 252
DEFAULT_HORIZON = 20
DEFAULT_MIN_ROWS = 400
DEFAULT_BLEND = 0.45
DEFAULT_SMOOTH = 0.40
MAX_GROUP = 512
MAX_GROUPS = 80
MIN_GROUPS = 12
N_GRADES = 5

_LGB_RANK_PARAMS = {
    "n_estimators": 60,
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 2.0,
    "lambdarank_truncation_level": 50,
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


def _day_stride(horizon: int) -> int:
    return max(1, int(horizon) // 4)


def _relevance_from_returns(y: np.ndarray, n_grades: int = N_GRADES) -> np.ndarray:
    """截面收益分档：0 最差，n_grades-1 最好。LambdaRank 默认 label_gain 只有 31 档，不能用 1..n 名次。"""
    y = np.asarray(y, dtype=np.float64)
    n = int(y.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.int32)
    order = np.argsort(y, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    grades = np.floor(ranks * int(n_grades) / n).astype(np.int32)
    return np.clip(grades, 0, int(n_grades) - 1)


def _cap_group(
    x: np.ndarray, y: np.ndarray, max_n: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(y.shape[0])
    grades = _relevance_from_returns(y)
    if n <= max_n:
        return x, grades
    top = np.where(grades == N_GRADES - 1)[0]
    bot = np.where(grades == 0)[0]
    keep = np.unique(np.concatenate([top, bot]))
    if int(keep.size) > max_n:
        keep = rng.choice(keep, size=int(max_n), replace=False)
    else:
        rest = np.setdiff1d(np.arange(n), keep, assume_unique=False)
        need = int(max_n) - int(keep.size)
        if need > 0 and rest.size > 0:
            pick = rng.choice(rest, size=min(need, int(rest.size)), replace=False)
            keep = np.concatenate([keep, pick])
    keep.sort()
    return x[keep], grades[keep]


def stack_realized_xy(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    stride: Optional[int] = None,
    group_cap: int = MAX_GROUP,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """拼已实现样本。从最近已实现日往回按 stride 抽样；y 是组内收益五档（0–4）。"""
    close = np.asarray(close_hist, dtype=np.float64)
    mask = np.asarray(mask_hist, dtype=bool)
    step = int(stride) if stride is not None else _day_stride(horizon)
    rng = np.random.default_rng(int(seed) % (2**31))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    groups: List[int] = []
    last = int(lookback) - 1
    i = last
    while i >= 0:
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
            i -= step
            continue
        xx, yy = _cap_group(x[row_ok], y[row_ok], int(group_cap), rng)
        xs.append(xx)
        ys.append(yy)
        groups.append(int(yy.shape[0]))
        i -= step
    if not xs:
        n_f = len(factor_hist)
        return (
            np.empty((0, n_f), dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    # 近端截面放前面，只留最近 MAX_GROUPS 组
    if len(xs) > MAX_GROUPS:
        xs = xs[:MAX_GROUPS]
        ys = ys[:MAX_GROUPS]
        groups = groups[:MAX_GROUPS]
    return np.vstack(xs), np.concatenate(ys), np.asarray(groups, dtype=np.int32)


def _label_gain(n_grades: int) -> list:
    return [float((1 << i) - 1) for i in range(max(1, int(n_grades)))]


def _fit_ranker(lgb, x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    y = np.asarray(y, dtype=np.int32)
    n_grades = int(y.max()) + 1 if y.size else N_GRADES
    n_grades = max(n_grades, N_GRADES)
    params = dict(_LGB_RANK_PARAMS)
    params["label_gain"] = _label_gain(n_grades)
    try:
        model = lgb.LGBMRanker(objective="lambdarank", **params)
        model.fit(x, y, group=groups)
        return model
    except TypeError:
        params.pop("lambdarank_truncation_level", None)
        params.pop("label_gain", None)
        model = lgb.LGBMRanker(**params)
        model.fit(x, y, group=groups)
        return model


def combine_scores_lgbm(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    min_rows: int = DEFAULT_MIN_ROWS,
    blend: float = DEFAULT_BLEND,
) -> Tuple[np.ndarray, Optional[object]]:
    """LambdaRank 拟合后与等权 z 混合。失败则等权回退。"""
    need = int(lookback) + int(horizon)
    today_uni = np.asarray(mask_hist[-1], dtype=bool)
    today_panels = [np.asarray(p)[-1] for p in factor_hist]
    eq = equal_weight_scores(today_panels, signs, today_uni)
    if not factor_hist:
        return eq, None
    for panel in factor_hist:
        if np.asarray(panel).shape[0] < need:
            return eq, None

    lgb = _try_lightgbm()
    if lgb is None:
        logger.warning("未安装 lightgbm，lgbm 策略回退等权 z-score")
        return eq, None

    x, y, groups = stack_realized_xy(
        factor_hist,
        signs,
        close_hist,
        mask_hist,
        lookback=lookback,
        horizon=horizon,
        seed=42 + int(lookback) + int(horizon),
    )
    need_rows = max(int(min_rows), 80 * len(factor_hist))
    if groups.size < MIN_GROUPS or x.shape[0] < need_rows:
        return eq, None

    try:
        model = _fit_ranker(lgb, x, y, groups)
    except Exception:
        logger.exception("LightGBM 排序拟合失败，回退等权")
        return eq, None

    z_cols = []
    for panel, sign in zip(today_panels, signs):
        raw = np.asarray(panel, dtype=np.float64) * float(sign)
        z_cols.append(cross_section_z(raw, today_uni))
    z = np.column_stack(z_cols)
    tree = np.full(today_uni.shape, np.nan, dtype=np.float64)
    ok = today_uni & np.all(np.isfinite(z), axis=1)
    if int(ok.sum()) == 0:
        return eq, None
    pred = np.asarray(model.predict(z[ok]), dtype=np.float64)
    tree[ok] = pred
    tz = cross_section_z(tree, ok)
    b = float(np.clip(blend, 0.0, 1.0))
    both = np.isfinite(tz) & np.isfinite(eq)
    out = eq.copy()
    out[both] = (1.0 - b) * eq[both] + b * tz[both]
    only_tree = np.isfinite(tz) & ~np.isfinite(eq)
    out[only_tree] = tz[only_tree]
    return out, model


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
        blend = float(getattr(args, "blend", None) if getattr(args, "blend", None) is not None else DEFAULT_BLEND)
        smooth = float(
            getattr(args, "smooth", None) if getattr(args, "smooth", None) is not None else DEFAULT_SMOOTH
        )
        if lookback < 60:
            raise SystemExit("lgbm --lookback 至少为 60（默认 252，约一年交易日）")
        if horizon < 1:
            raise SystemExit("lgbm --horizon 至少为 1（默认 20，与 20 日调仓对齐）")
        if not 0.0 <= blend <= 1.0:
            raise SystemExit("lgbm --blend 须在 0 与 1 之间（树分权重，其余为等权）")
        if not 0.0 <= smooth < 1.0:
            raise SystemExit("lgbm --smooth 须在 [0, 1)（上期分数权重）")
        return cls(
            legs,
            n=n,
            allocator=allocator,
            rebalance=spec,
            lookback=lookback,
            horizon=horizon,
            blend=blend,
            smooth=smooth,
        )

    @classmethod
    def cli_fields(cls):
        return [
            cli_field("factor", "因子列表", "str", required=True, placeholder="a,b,-c"),
            n_field(DEFAULT_N),
            rebalance_field("20", placeholder="daily / weekly / 20"),
            cli_field("lookback", "LGBM回看天数", "int", default=DEFAULT_LOOKBACK),
            cli_field("horizon", "LGBM持有期", "int", default=DEFAULT_HORIZON),
            cli_field("blend", "树分权重", "float", default=DEFAULT_BLEND),
            cli_field("smooth", "上期分数平滑", "float", default=DEFAULT_SMOOTH),
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
        blend = getattr(args, "blend", None)
        b = DEFAULT_BLEND if blend is None else float(blend)
        tag += f"_lgb{lookback}h{horizon}rk{int(round(b * 100))}"
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
        blend: float = DEFAULT_BLEND,
        smooth: float = DEFAULT_SMOOTH,
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
        self.lookback = max(60, int(lookback))
        self.horizon = max(1, int(horizon))
        self.min_rows = max(80, int(min_rows))
        self.blend = float(np.clip(blend, 0.0, 1.0))
        self.smooth = float(np.clip(smooth, 0.0, 0.95))
        self._gate = RebalanceGate(self.rebalance)
        self.last_model: Optional[object] = None
        self._prev_score: Optional[np.ndarray] = None

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
            blend=self.blend,
        )
        self.last_model = model
        if self._prev_score is not None and self._prev_score.shape == score.shape and self.smooth > 0:
            both = np.isfinite(score) & np.isfinite(self._prev_score)
            mixed = score.copy()
            mixed[both] = (1.0 - self.smooth) * score[both] + self.smooth * self._prev_score[both]
            score = mixed
        self._prev_score = np.asarray(score, dtype=np.float64)
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
