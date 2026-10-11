#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每个因子单独学非线性形状，仓位按树分加权。

每个因子一棵树，输入池内原始截面 z（不翻号、不先线性合成）。
多因子是把各树的截面标准化分数相加，不是对原始 z 加权。
默认不再等权 TopN：权重大于截面中位数的树分，可按 --n 截断只留头部。
--n 0 表示不截断。000852 只是 zz1000 的对照基准，不是策略构造的一部分。

用法：--strategy lgbm_shape --factor a
一次一个因子才能看到「这个形状」用到什么程度；多个就写 a,b,c。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import Strategy, cli_field, n_field, rebalance_field
from Strategies._factor_combo import FactorLeg, cross_section_z, parse_factor_list
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli

logger = logging.getLogger("Strategies.lgbm_shape")

DEFAULT_N = 50
DEFAULT_LOOKBACK = 252
DEFAULT_HORIZON = 20
DEFAULT_REFIT = 20
DEFAULT_MIN_ROWS = 400
DEFAULT_SMOOTH = 0.50
DEFAULT_BINS = 0
MAX_GROUP = 512
MAX_GROUPS = 80
MIN_GROUPS = 12
N_GRADES = 6
HEAD_GAIN = (0.0, 1.0, 3.0, 8.0, 20.0, 40.0)

_LGB_SHAPE_PARAMS = {
    "n_estimators": 120,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": 8,
    "min_child_samples": 48,
    "min_split_gain": 0.01,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 4.0,
    "reg_alpha": 0.2,
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


def _quantile_bins(z: np.ndarray, mask: np.ndarray, n_bins: int) -> np.ndarray:
    """股票池内按 z 等数量分档，1…n_bins；样本不够则全 NaN。"""
    out = np.full(np.asarray(z).shape, np.nan, dtype=np.float64)
    sel = np.asarray(mask, dtype=bool) & np.isfinite(z)
    n = int(sel.sum())
    k = max(3, int(n_bins))
    if n < k:
        return out
    ranks = np.empty(n, dtype=np.float64)
    order = np.argsort(z[sel], kind="mergesort")
    ranks[order] = np.arange(n, dtype=np.float64)
    bins = np.floor(ranks * k / n) + 1.0
    out[sel] = np.clip(bins, 1.0, float(k))
    return out


def expand_shape_features(z: np.ndarray, mask: np.ndarray, n_bins: int) -> np.ndarray:
    """单因子：原始截面 z；--bins≥3 时再加一列分位档，仍不假定单调。"""
    z = np.asarray(z, dtype=np.float64)
    if int(n_bins) >= 3:
        return np.column_stack([z, _quantile_bins(z, mask, int(n_bins))])
    return z.reshape(-1, 1)


def _features_from_panels(
    panels: Sequence[np.ndarray],
    mask: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    chunks = []
    for panel in panels:
        z = cross_section_z(np.asarray(panel, dtype=np.float64), mask)
        chunks.append(expand_shape_features(z, mask, n_bins))
    if not chunks:
        return np.empty((int(np.asarray(mask).shape[0]), 0), dtype=np.float64)
    return np.column_stack(chunks)


def _relevance_from_returns(y: np.ndarray, n_grades: int = N_GRADES) -> np.ndarray:
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
    keep = np.where((grades == 0) | (grades >= N_GRADES - 2))[0]
    keep = np.unique(keep)
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


def _realized_days(
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    close = np.asarray(close_hist, dtype=np.float64)
    mask = np.asarray(mask_hist, dtype=bool)
    step = _day_stride(horizon)
    last = int(lookback) - 1
    out: List[Tuple[int, np.ndarray, np.ndarray]] = []
    i = last
    while i >= 0 and len(out) < MAX_GROUPS:
        j = i + int(horizon)
        c0 = close[i]
        c1 = close[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            y = c1 / c0 - 1.0
        y = np.where(np.isfinite(c0) & (c0 > 0) & np.isfinite(c1) & (c1 > 0), y, np.nan)
        out.append((i, mask[i], y))
        i -= step
    return out


def _stack_realized_xy(
    factor_hist: Sequence[np.ndarray],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    n_bins: int,
    seed: int = 42,
    days: Optional[Sequence[Tuple[int, np.ndarray, np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) % (2**31))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    groups: List[int] = []
    n_f = len(factor_hist)
    n_cols = n_f * (2 if int(n_bins) >= 3 else 1)
    slices = days if days is not None else _realized_days(
        close_hist, mask_hist, lookback=lookback, horizon=horizon
    )
    for i, uni, y in slices:
        day_panels = [np.asarray(panel[i], dtype=np.float64) for panel in factor_hist]
        x = _features_from_panels(day_panels, uni, n_bins)
        row_ok = np.asarray(uni, dtype=bool) & np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        if int(row_ok.sum()) < 8:
            continue
        xx, yy = _cap_group(x[row_ok], y[row_ok], MAX_GROUP, rng)
        xs.append(xx)
        ys.append(yy)
        groups.append(int(yy.shape[0]))
    if not xs:
        return (
            np.empty((0, n_cols), dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    return np.vstack(xs), np.concatenate(ys), np.asarray(groups, dtype=np.int32)


def score_to_weights(score: np.ndarray, tradable: np.ndarray, n: int) -> np.ndarray:
    """树分大于截面中位数的多头，权重大小跟树分走；n>0 只留头部 n 只。"""
    n_sym = int(np.asarray(score).shape[0])
    weights = np.zeros(n_sym, dtype=np.float64)
    ok = np.asarray(tradable, dtype=bool) & np.isfinite(score)
    if int(ok.sum()) <= 0:
        return weights
    s = np.asarray(score, dtype=np.float64)
    mid = float(np.nanmedian(s[ok]))
    edge = np.where(ok, s - mid, 0.0)
    edge = np.maximum(edge, 0.0)
    if int(n) > 0:
        cand = np.where(edge > 0)[0]
        if cand.size > int(n):
            keep = cand[np.argpartition(edge[cand], -int(n))[-int(n):]]
            mask = np.zeros(n_sym, dtype=bool)
            mask[keep] = True
            edge = np.where(mask, edge, 0.0)
    tot = float(edge.sum())
    if tot <= 0:
        return weights
    weights = edge / tot
    return weights


def _fit_ranker(lgb, x: np.ndarray, y: np.ndarray, groups: np.ndarray, truncation: int):
    y = np.asarray(y, dtype=np.int32)
    n_grades = int(y.max()) + 1 if y.size else N_GRADES
    n_grades = max(n_grades, N_GRADES)
    gain = list(HEAD_GAIN)
    if len(gain) < n_grades:
        gain.extend([gain[-1]] * (n_grades - len(gain)))
    params = dict(_LGB_SHAPE_PARAMS)
    params["lambdarank_truncation_level"] = max(10, int(truncation))
    params["label_gain"] = gain[:n_grades]
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


def _as_model_list(model: Optional[object], n: int) -> Optional[List[object]]:
    if model is None:
        return None
    if isinstance(model, (list, tuple)):
        items = list(model)
        return items if len(items) == int(n) else None
    return [model] if int(n) == 1 else None


def _predict_one(model: object, panel: np.ndarray, mask: np.ndarray, n_bins: int) -> np.ndarray:
    feat = _features_from_panels([np.asarray(panel, dtype=np.float64)], mask, n_bins)
    out = np.full(np.asarray(mask).shape, np.nan, dtype=np.float64)
    ok = np.asarray(mask, dtype=bool) & np.all(np.isfinite(feat), axis=1)
    if int(ok.sum()) == 0:
        return out
    pred = np.asarray(model.predict(feat[ok]), dtype=np.float64)
    out[ok] = pred
    return cross_section_z(out, ok)


def combine_scores_lgbm_shape(
    factor_hist: Sequence[np.ndarray],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    n_bins: int = DEFAULT_BINS,
    min_rows: int = DEFAULT_MIN_ROWS,
    truncation: int = DEFAULT_N,
    model: Optional[object] = None,
) -> Tuple[np.ndarray, Optional[List[object]]]:
    """每个因子单独拟合，再把各树的截面 z 分相加。"""
    need = int(lookback) + int(horizon)
    today_uni = np.asarray(mask_hist[-1], dtype=bool)
    empty = np.full(today_uni.shape, np.nan, dtype=np.float64)
    if not factor_hist:
        return empty, None
    for panel in factor_hist:
        if np.asarray(panel).shape[0] < need:
            return empty, None
    reused = _as_model_list(model, len(factor_hist))
    fitted: List[Optional[object]] = [None] * len(factor_hist)
    scores: List[np.ndarray] = []
    days = None
    lgb = None
    if reused is None:
        lgb = _try_lightgbm()
        if lgb is None:
            logger.warning("未安装 lightgbm，lgbm_shape 跳过本调仓日")
            return empty, None
        days = _realized_days(close_hist, mask_hist, lookback=lookback, horizon=horizon)
    for i, panel in enumerate(factor_hist):
        one = reused[i] if reused is not None else None
        if one is None:
            x, y, groups = _stack_realized_xy(
                [panel],
                close_hist,
                mask_hist,
                lookback=lookback,
                horizon=horizon,
                n_bins=n_bins,
                seed=42 + int(lookback) + int(horizon) + int(n_bins) + i,
                days=days,
            )
            need_rows = max(int(min_rows), 40 * max(1, int(x.shape[1])))
            if groups.size < MIN_GROUPS or x.shape[0] < need_rows:
                continue
            try:
                one = _fit_ranker(lgb, x, y, groups, truncation=max(10, int(truncation) or DEFAULT_N))
            except Exception:
                logger.exception("lgbm_shape 单因子拟合失败 factor=%s", i)
                continue
        today = np.asarray(panel)[-1]
        scores.append(_predict_one(one, today, today_uni, n_bins))
        fitted[i] = one
    if not scores or not any(item is not None for item in fitted):
        return empty, None
    stacked = np.vstack(scores)
    count = np.sum(np.isfinite(stacked), axis=0)
    total = np.nansum(np.where(np.isfinite(stacked), stacked, 0.0), axis=0)
    out = empty.copy()
    ok = count > 0
    out[ok] = total[ok] / count[ok]
    cache = fitted if all(item is not None for item in fitted) else None
    return out, cache


class LgbmShapeTopK(Strategy):
    name = "lgbm_shape"

    @classmethod
    def from_cli(cls, args, allocator):
        if not args.factor:
            raise SystemExit("lgbm_shape 需要 --factor，逗号分隔，如 a,b,c")
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
        smooth = float(
            getattr(args, "smooth", None) if getattr(args, "smooth", None) is not None else DEFAULT_SMOOTH
        )
        bins = int(getattr(args, "bins", None) if getattr(args, "bins", None) is not None else DEFAULT_BINS)
        if lookback < 60:
            raise SystemExit("lgbm_shape --lookback 至少为 60（默认 252）")
        if horizon < 1:
            raise SystemExit("lgbm_shape --horizon 至少为 1（默认 20）")
        if refit < 1:
            raise SystemExit("lgbm_shape --refit 至少为 1（默认 20）")
        if not 0.0 <= smooth < 1.0:
            raise SystemExit("lgbm_shape --smooth 须在 [0, 1)")
        if bins != 0 and bins < 3:
            raise SystemExit("lgbm_shape --bins 为 0（默认，只用原始 z）或 ≥3")
        return cls(
            legs,
            n=n,
            allocator=allocator,
            rebalance=spec,
            lookback=lookback,
            horizon=horizon,
            refit=refit,
            smooth=smooth,
            bins=bins,
        )

    @classmethod
    def cli_fields(cls):
        return [
            cli_field("factor", "因子列表", "str", required=True, placeholder="a,b,-c"),
            n_field(DEFAULT_N, label="持仓上限，0=不截断"),
            rebalance_field("20", placeholder="daily / weekly / 20"),
            cli_field("lookback", "LGBM回看天数", "int", default=DEFAULT_LOOKBACK),
            cli_field("horizon", "LGBM持有期", "int", default=DEFAULT_HORIZON),
            cli_field("refit", "同一模型沿用交易日数", "int", default=DEFAULT_REFIT),
            cli_field("smooth", "上期分数平滑", "float", default=DEFAULT_SMOOTH),
            cli_field("bins", "可选分位档，0=只用原始z", "int", default=DEFAULT_BINS),
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
        tag = "lgbmshape_" + "_".join(parts)
        spec = parse_rebalance(getattr(args, "rebalance", None) or "20")
        tag += spec.tag_suffix()
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        refit = DEFAULT_REFIT if getattr(args, "refit", None) is None else int(args.refit)
        bins = DEFAULT_BINS if getattr(args, "bins", None) is None else int(args.bins)
        n = DEFAULT_N if getattr(args, "n", None) is None else int(args.n)
        tag += f"_lgb{lookback}h{horizon}rf{refit}b{bins}n{n}"
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
        smooth: float = DEFAULT_SMOOTH,
        bins: int = DEFAULT_BINS,
    ):
        legs: List[FactorLeg] = []
        for item in factors:
            if isinstance(item, str):
                legs.extend(parse_factor_list(item))
            else:
                name, sign = item
                legs.append((str(name), float(sign)))
        if not legs:
            raise ValueError("lgbm_shape 至少需要一个因子")
        self.legs = legs
        self.n = max(0, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = (
            parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        )
        self.lookback = max(60, int(lookback))
        self.horizon = max(1, int(horizon))
        self.refit = max(1, int(refit))
        self.min_rows = max(80, int(min_rows))
        self.smooth = float(np.clip(smooth, 0.0, 0.95))
        self.bins = 0 if int(bins) <= 0 else max(3, int(bins))
        self._gate = RebalanceGate(self.rebalance)
        self.last_model: Optional[List[object]] = None
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
        close = ctx.history("close", need)
        t1 = int(ctx._t) + 1
        t0 = max(0, t1 - need)
        mask = np.asarray(ctx._store.mask[t0:t1], dtype=bool)
        if mask.shape[0] < need:
            pad = np.zeros((need - mask.shape[0], mask.shape[1]), dtype=bool)
            mask = np.vstack([pad, mask])
        factor_hist = [ctx.history(name, need) for name, _ in self.legs]
        reuse = None if self._need_refit(ctx._t) else self.last_model
        kwargs = dict(
            lookback=self.lookback,
            horizon=self.horizon,
            n_bins=self.bins,
            min_rows=self.min_rows,
            truncation=self.n,
        )
        score, fitted = combine_scores_lgbm_shape(
            factor_hist, close, mask, model=reuse, **kwargs
        )
        if self._need_refit(ctx._t):
            if fitted is not None:
                self.last_model = fitted
                self._fitted_t = int(ctx._t)
            elif self.last_model is not None:
                score, _ = combine_scores_lgbm_shape(
                    factor_hist, close, mask, model=self.last_model, **kwargs
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
        if isinstance(self.allocator, EqualWeight):
            weights = score_to_weights(values, tradable, self.n)
        else:
            ranked = np.where(tradable, values, -np.inf)
            k = int(tradable.sum()) if self.n <= 0 else int(min(self.n, int(tradable.sum())))
            if k <= 0:
                weights = np.zeros(len(ctx.symbols))
            else:
                pick = np.argpartition(ranked, -k)[-k:]
                pick = pick[tradable[pick]]
                weights = self.allocator.size(ctx, pick)
        return self._gate.remember(
            TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
        )
