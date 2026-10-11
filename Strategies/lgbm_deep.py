#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""更深 LambdaRank：纯树分、适度强化头部、保留分数平滑。

用法：--strategy lgbm_deep --factor a,b,c
因子名前加 '-' 表示取负。

与 lgbm 独立实现，不改原版：
1. 树更深（15 叶 / 深度 6 / 100 棵），不用等权混合，拟合失败当日空仓
2. 收益分 6 档，头部两档增益高于原版五档，但不拉到极端尾部
3. 仍与上期分数 smooth，压换手
4. 每 --refit 个交易日重训一次，同一模型沿用到下次重训
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

logger = logging.getLogger("Strategies.lgbm_deep")

DEFAULT_N = 50
DEFAULT_LOOKBACK = 252
DEFAULT_HORIZON = 20
DEFAULT_REFIT = 20
DEFAULT_MIN_ROWS = 400
DEFAULT_SMOOTH = 0.50
MAX_GROUP = 512
MAX_GROUPS = 80
MIN_GROUPS = 12
N_GRADES = 6
# 原版五档增益到 15；这里多一档，顶部两档约 20 / 40，强化头部但不追彩票尾
HEAD_GAIN = (0.0, 1.0, 3.0, 8.0, 20.0, 40.0)
KEEP_GRADES = (0, 4, 5)

_LGB_DEEP_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.04,
    "num_leaves": 15,
    "max_depth": 6,
    "min_child_samples": 32,
    "min_split_gain": 0.02,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_lambda": 3.0,
    "reg_alpha": 0.1,
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
    prefer = np.where(np.isin(grades, np.asarray(KEEP_GRADES, dtype=np.int32)))[0]
    keep = np.unique(prefer)
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


def _stack_realized_xy(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = np.asarray(close_hist, dtype=np.float64)
    mask = np.asarray(mask_hist, dtype=bool)
    step = _day_stride(horizon)
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
        xx, yy = _cap_group(x[row_ok], y[row_ok], MAX_GROUP, rng)
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
    if len(xs) > MAX_GROUPS:
        xs = xs[:MAX_GROUPS]
        ys = ys[:MAX_GROUPS]
        groups = groups[:MAX_GROUPS]
    return np.vstack(xs), np.concatenate(ys), np.asarray(groups, dtype=np.int32)


def _fit_ranker(lgb, x: np.ndarray, y: np.ndarray, groups: np.ndarray, truncation: int):
    y = np.asarray(y, dtype=np.int32)
    n_grades = int(y.max()) + 1 if y.size else N_GRADES
    n_grades = max(n_grades, N_GRADES)
    gain = list(HEAD_GAIN)
    if len(gain) < n_grades:
        gain.extend([gain[-1]] * (n_grades - len(gain)))
    params = dict(_LGB_DEEP_PARAMS)
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


def combine_scores_lgbm_deep(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    min_rows: int = DEFAULT_MIN_ROWS,
    truncation: int = DEFAULT_N,
    model: Optional[object] = None,
) -> Tuple[np.ndarray, Optional[object]]:
    """只返回树分截面 z。失败返回全 NaN，不回退等权。传入 model 则只打分不重训。"""
    need = int(lookback) + int(horizon)
    today_uni = np.asarray(mask_hist[-1], dtype=bool)
    empty = np.full(today_uni.shape, np.nan, dtype=np.float64)
    if not factor_hist:
        return empty, None
    fitted = model
    if fitted is None:
        for panel in factor_hist:
            if np.asarray(panel).shape[0] < need:
                return empty, None
        lgb = _try_lightgbm()
        if lgb is None:
            logger.warning("未安装 lightgbm，lgbm_deep 跳过本调仓日")
            return empty, None
        x, y, groups = _stack_realized_xy(
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
            return empty, None
        try:
            fitted = _fit_ranker(lgb, x, y, groups, truncation=truncation)
        except Exception:
            logger.exception("lgbm_deep 排序拟合失败")
            return empty, None

    today_panels = [np.asarray(p)[-1] for p in factor_hist]
    z_cols = []
    for panel, sign in zip(today_panels, signs):
        raw = np.asarray(panel, dtype=np.float64) * float(sign)
        z_cols.append(cross_section_z(raw, today_uni))
    z = np.column_stack(z_cols)
    tree = np.full(today_uni.shape, np.nan, dtype=np.float64)
    ok = today_uni & np.all(np.isfinite(z), axis=1)
    if int(ok.sum()) == 0:
        return empty, fitted
    pred = np.asarray(fitted.predict(z[ok]), dtype=np.float64)
    tree[ok] = pred
    return cross_section_z(tree, ok), fitted


class LgbmDeepTopK(Strategy):
    name = "lgbm_deep"

    @classmethod
    def from_cli(cls, args, allocator):
        if not args.factor:
            raise SystemExit("lgbm_deep 需要 --factor，逗号分隔，如 a,b,c")
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
        if lookback < 60:
            raise SystemExit("lgbm_deep --lookback 至少为 60（默认 252）")
        if horizon < 1:
            raise SystemExit("lgbm_deep --horizon 至少为 1（默认 20）")
        if refit < 1:
            raise SystemExit("lgbm_deep --refit 至少为 1（默认 20，同一模型沿用天数）")
        if not 0.0 <= smooth < 1.0:
            raise SystemExit("lgbm_deep --smooth 须在 [0, 1)（上期分数权重）")
        return cls(
            legs,
            n=n,
            allocator=allocator,
            rebalance=spec,
            lookback=lookback,
            horizon=horizon,
            refit=refit,
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
            cli_field("refit", "同一模型沿用交易日数", "int", default=DEFAULT_REFIT),
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
        tag = "lgbmdeep_" + "_".join(parts)
        spec = parse_rebalance(getattr(args, "rebalance", None) or "20")
        tag += spec.tag_suffix()
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        refit = DEFAULT_REFIT if getattr(args, "refit", None) is None else int(args.refit)
        tag += f"_lgb{lookback}h{horizon}rf{refit}d15"
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
    ):
        legs: List[FactorLeg] = []
        for item in factors:
            if isinstance(item, str):
                legs.extend(parse_factor_list(item))
            else:
                name, sign = item
                legs.append((str(name), float(sign)))
        if not legs:
            raise ValueError("lgbm_deep 至少需要一个因子")
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
        self.smooth = float(np.clip(smooth, 0.0, 0.95))
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
        close = ctx.history("close", need)
        t1 = int(ctx._t) + 1
        t0 = max(0, t1 - need)
        mask = np.asarray(ctx._store.mask[t0:t1], dtype=bool)
        if mask.shape[0] < need:
            pad = np.zeros((need - mask.shape[0], mask.shape[1]), dtype=bool)
            mask = np.vstack([pad, mask])
        signs = [sign for _, sign in self.legs]
        factor_hist = [ctx.history(name, need) for name, _ in self.legs]
        reuse = None if self._need_refit(ctx._t) else self.last_model
        score, fitted = combine_scores_lgbm_deep(
            factor_hist,
            signs,
            close,
            mask,
            lookback=self.lookback,
            horizon=self.horizon,
            min_rows=self.min_rows,
            truncation=self.n,
            model=reuse,
        )
        if self._need_refit(ctx._t):
            if fitted is not None:
                self.last_model = fitted
                self._fitted_t = int(ctx._t)
            elif self.last_model is not None:
                score, _ = combine_scores_lgbm_deep(
                    factor_hist,
                    signs,
                    close,
                    mask,
                    lookback=self.lookback,
                    horizon=self.horizon,
                    min_rows=self.min_rows,
                    truncation=self.n,
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