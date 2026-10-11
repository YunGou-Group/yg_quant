#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分类模型 + 特质收益标签 + 前后尾部样本。

对应课程里的三条：

1. 用二分类，不回归收益率，也不做 LambdaRank 五档排序。
2. 标签来自持有期收益对当日 Barra X_T（风格、NLSIZE、有行业时加上行业）
   的市值加权回归残差。截距去掉了当天的市场收益，其余列去掉风格，
   残差就是特质收益。
3. 每个已实现交易日只保留特质收益的前 ``tail`` 与后 ``tail``，默认前后 20%。
   前尾记 1，后尾记 0，中间不进训练集。

预测分是「未来落入特质收益前尾」的概率。只使用 asof 上已经走完的
t→t+H 收益。样本不足或未安装 lightgbm 时回退等权 z-score。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from FactorEvaluates.exposure_engine import RAW_STYLE_NAMES
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
from Strategies.lgbm import (
    DEFAULT_HORIZON,
    DEFAULT_LOOKBACK,
    DEFAULT_MIN_ROWS,
    DEFAULT_N,
    DEFAULT_REFIT,
    _day_stride,
    _try_lightgbm,
)
from Strategies.lgbm_barra import _window2, barra_book
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli

logger = logging.getLogger("Strategies.lgbm_classification_specific_return")

DEFAULT_TAIL = 0.20
DEFAULT_BLEND = 1.0
DEFAULT_SMOOTH = 0.0
MIN_SIDE = 8
MIN_DAYS = 12
MAX_DAYS = 80

_LGB_CLF_PARAMS = {
    "objective": "binary",
    "n_estimators": 60,
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": -1,
}


def _tail_class(specific: np.ndarray, universe: np.ndarray, tail: float) -> np.ndarray:
    """前尾为 1，后尾为 0，中间与无效样本为 NaN。分位在当天成分内计算。"""
    y = np.asarray(specific, dtype=np.float64)
    out = np.full(y.shape, np.nan, dtype=np.float64)
    ok = np.asarray(universe, dtype=bool) & np.isfinite(y)
    if int(ok.sum()) < max(20, MIN_SIDE * 2):
        return out
    lo = float(np.quantile(y[ok], float(tail)))
    hi = float(np.quantile(y[ok], 1.0 - float(tail)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return out
    out[ok & (y <= lo)] = 0.0
    out[ok & (y >= hi)] = 1.0
    return out


def stack_classification_xy(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    specific_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    tail: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """拼已实现的前后尾样本。窗口右端是决策日，训练停在 lookback-1，不含未走完的持有期。"""
    step = _day_stride(horizon)
    mask = np.asarray(mask_hist, dtype=bool)
    specific = np.asarray(specific_hist, dtype=np.float64)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    last = int(lookback) - 1
    i = last
    n_days = 0
    n_f = len(factor_hist)
    while i >= 0 and n_days < MAX_DAYS:
        uni = mask[i]
        label = _tail_class(specific[i], uni, tail)
        cols = []
        for panel, sign in zip(factor_hist, signs):
            raw = np.asarray(panel[i], dtype=np.float64) * float(sign)
            cols.append(cross_section_z(raw, uni))
        if not cols:
            break
        x = np.column_stack(cols)
        row_ok = np.isfinite(label) & np.all(np.isfinite(x), axis=1)
        if int(row_ok.sum()) < MIN_SIDE * 2:
            i -= step
            continue
        yy = label[row_ok]
        if int((yy == 0).sum()) < MIN_SIDE or int((yy == 1).sum()) < MIN_SIDE:
            i -= step
            continue
        xs.append(x[row_ok])
        ys.append(yy)
        n_days += 1
        i -= step
    if not xs:
        return (
            np.empty((0, n_f), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0,
        )
    return np.vstack(xs), np.concatenate(ys), n_days


def _fit_classifier(lgb, x: np.ndarray, y: np.ndarray):
    model = lgb.LGBMClassifier(**_LGB_CLF_PARAMS)
    model.fit(x, y.astype(np.int32))
    return model


def _positive_probability(model, x: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = [int(c) for c in getattr(model, "classes_", [0, 1])]
    if 1 not in classes:
        return np.full(x.shape[0], np.nan, dtype=np.float64)
    return proba[:, classes.index(1)]


def combine_scores_classification(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    specific_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    tail: float,
    min_rows: int,
    blend: float,
    model: Optional[object] = None,
) -> Tuple[np.ndarray, Optional[object]]:
    """二分类概率与等权 z 混合。失败则只用等权。传入 model 则只打分。"""
    today_uni = np.asarray(mask_hist[-1], dtype=bool)
    today_panels = [np.asarray(p)[-1] for p in factor_hist]
    eq = equal_weight_scores(today_panels, signs, today_uni)
    if not factor_hist:
        return eq, None
    fitted = model
    if fitted is None:
        need = int(lookback) + int(horizon)
        if any(np.asarray(p).shape[0] < need for p in factor_hist):
            return eq, None
        lgb = _try_lightgbm()
        if lgb is None:
            logger.warning("未安装 lightgbm，分类策略回退等权 z-score")
            return eq, None
        x, y, n_days = stack_classification_xy(
            factor_hist,
            signs,
            specific_hist,
            mask_hist,
            lookback=lookback,
            horizon=horizon,
            tail=tail,
        )
        need_rows = max(int(min_rows), 80 * len(factor_hist))
        if n_days < MIN_DAYS or x.shape[0] < need_rows or len(np.unique(y)) < 2:
            return eq, None
        try:
            fitted = _fit_classifier(lgb, x, y)
        except Exception:
            logger.exception("LightGBM 分类拟合失败，回退等权")
            return eq, None
        logger.info(
            "分类样本 %s 行 / %s 日，前尾占比 %.2f，tail=%.2f",
            int(x.shape[0]),
            int(n_days),
            float(np.mean(y == 1)),
            float(tail),
        )

    z_cols = []
    for panel, sign in zip(today_panels, signs):
        raw = np.asarray(panel, dtype=np.float64) * float(sign)
        z_cols.append(cross_section_z(raw, today_uni))
    z = np.column_stack(z_cols)
    tree = np.full(today_uni.shape, np.nan, dtype=np.float64)
    ok = today_uni & np.all(np.isfinite(z), axis=1)
    if int(ok.sum()) == 0:
        return eq, fitted
    try:
        tree[ok] = _positive_probability(fitted, z[ok])
    except Exception:
        logger.exception("LightGBM 分类打分失败，回退等权")
        return eq, fitted
    tz = cross_section_z(tree, ok)
    b = float(np.clip(blend, 0.0, 1.0))
    both = np.isfinite(tz) & np.isfinite(eq)
    out = eq.copy()
    out[both] = (1.0 - b) * eq[both] + b * tz[both]
    only_tree = np.isfinite(tz) & ~np.isfinite(eq)
    out[only_tree] = tz[only_tree]
    return out, fitted


class LgbmClassificationSpecificReturnTopK(Strategy):
    """特质收益前后尾上的 LightGBM 二分类，再取 Top N。"""

    name = "lgbm_classification_specific_return"

    @classmethod
    def from_cli(cls, args, allocator):
        if not args.factor:
            raise SystemExit("lgbm_classification_specific_return 需要 --factor，逗号分隔，如 a,b,c")
        try:
            legs = parse_factor_list(args.factor)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        n = args.n if args.n is not None else DEFAULT_N
        spec = spec_from_cli(getattr(args, "rebalance", None) or "20")
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
        tail = float(getattr(args, "tail", None) if getattr(args, "tail", None) is not None else DEFAULT_TAIL)
        if lookback < 60:
            raise SystemExit("lgbm_classification_specific_return --lookback 至少为 60")
        if horizon < 1:
            raise SystemExit("lgbm_classification_specific_return --horizon 至少为 1")
        if refit < 1:
            raise SystemExit("lgbm_classification_specific_return --refit 至少为 1")
        if not 0.0 <= blend <= 1.0:
            raise SystemExit("lgbm_classification_specific_return --blend 须在 0 与 1 之间")
        if not 0.0 <= smooth < 1.0:
            raise SystemExit("lgbm_classification_specific_return --smooth 须在 [0, 1)")
        if not 0.05 <= tail <= 0.45:
            raise SystemExit("lgbm_classification_specific_return --tail 须在 0.05 与 0.45 之间，默认 0.2")
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
            tail=tail,
        )

    @classmethod
    def cli_fields(cls):
        return [
            cli_field("factor", "因子列表", "str", required=True, placeholder="a,b,-c"),
            n_field(DEFAULT_N),
            rebalance_field("20", placeholder="daily / weekly / 20"),
            cli_field("lookback", "LGBM回看天数", "int", default=DEFAULT_LOOKBACK),
            cli_field("horizon", "持有期", "int", default=DEFAULT_HORIZON),
            cli_field("refit", "同一模型沿用交易日数", "int", default=DEFAULT_REFIT),
            cli_field("blend", "分类概率权重，其余为等权", "float", default=DEFAULT_BLEND),
            cli_field("smooth", "上期分数平滑", "float", default=DEFAULT_SMOOTH),
            cli_field("tail", "前后尾样本比例，0.2 表示各取 20%", "float", default=DEFAULT_TAIL),
        ]

    @classmethod
    def panel_kwargs(cls, args) -> dict:
        legs = parse_factor_list(args.factor)
        names = [name for name, _sign in legs]
        extra = [name for name in RAW_STYLE_NAMES if name not in names]
        return {"factors": names + extra}

    @classmethod
    def warmup(cls, args) -> int:
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        return lookback + horizon + 5

    @classmethod
    def run_tag(cls, args) -> str:
        legs = parse_factor_list(args.factor)
        parts = [("m" if s < 0 else "") + n for n, s in legs]
        tag = "lgbm_classification_specific_return_" + "_".join(parts)
        spec = parse_rebalance(getattr(args, "rebalance", None) or "20")
        tag += spec.tag_suffix()
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        refit = DEFAULT_REFIT if getattr(args, "refit", None) is None else int(args.refit)
        blend = getattr(args, "blend", None)
        b = DEFAULT_BLEND if blend is None else float(blend)
        tail = getattr(args, "tail", None)
        tail_v = DEFAULT_TAIL if tail is None else float(tail)
        tag += f"_lgb{lookback}h{horizon}rf{refit}rk{int(round(b * 100))}tail{int(round(tail_v * 100))}"
        return tag

    def __init__(
        self,
        factors: Sequence[FactorLeg] | Sequence[str],
        n: int = DEFAULT_N,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = "20",
        lookback: int = DEFAULT_LOOKBACK,
        horizon: int = DEFAULT_HORIZON,
        refit: int = DEFAULT_REFIT,
        min_rows: int = DEFAULT_MIN_ROWS,
        blend: float = DEFAULT_BLEND,
        smooth: float = DEFAULT_SMOOTH,
        tail: float = DEFAULT_TAIL,
    ):
        legs: List[FactorLeg] = []
        for item in factors:
            if isinstance(item, str):
                legs.extend(parse_factor_list(item))
            else:
                name, sign = item
                legs.append((str(name), float(sign)))
        if not legs:
            raise ValueError("lgbm_classification_specific_return 至少需要一个因子")
        self.legs = legs
        self.n = max(1, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = (
            parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        )
        self.lookback = max(60, int(lookback))
        self.horizon = max(1, int(horizon))
        self.refit = max(1, int(refit))
        self.min_rows = max(80, int(min_rows))
        self.blend = float(np.clip(blend, 0.0, 1.0))
        self.smooth = float(np.clip(smooth, 0.0, 0.95))
        self.tail = float(tail)
        self._gate = RebalanceGate(self.rebalance)
        self.last_model: Optional[object] = None
        self._fitted_t: Optional[int] = None
        self._prev_score: Optional[np.ndarray] = None

    @property
    def factor_names(self) -> List[str]:
        return [name for name, _ in self.legs]

    def _need_refit(self, t: int) -> bool:
        if self.last_model is None or self._fitted_t is None:
            return True
        return int(t) - int(self._fitted_t) >= int(self.refit)

    def _lgbm_score(self, ctx: DayContext) -> np.ndarray:
        need = self.lookback + self.horizon
        book = barra_book(ctx._store, self.horizon, 60, "resid", 2.0)
        t = int(ctx._t)
        mask = _window2(ctx._store.mask, t, need)
        specific = _window2(book.label, t, need)
        signs = [sign for _, sign in self.legs]
        factor_hist = [ctx.history(name, need) for name, _ in self.legs]
        reuse = None if self._need_refit(t) else self.last_model
        score, fitted = combine_scores_classification(
            factor_hist,
            signs,
            specific,
            mask,
            lookback=self.lookback,
            horizon=self.horizon,
            tail=self.tail,
            min_rows=self.min_rows,
            blend=self.blend,
            model=reuse,
        )
        if self._need_refit(t):
            if fitted is not None:
                self.last_model = fitted
                self._fitted_t = t
            elif self.last_model is not None:
                score, _ = combine_scores_classification(
                    factor_hist,
                    signs,
                    specific,
                    mask,
                    lookback=self.lookback,
                    horizon=self.horizon,
                    tail=self.tail,
                    min_rows=self.min_rows,
                    blend=self.blend,
                    model=self.last_model,
                )
        else:
            self.last_model = fitted if fitted is not None else self.last_model
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
