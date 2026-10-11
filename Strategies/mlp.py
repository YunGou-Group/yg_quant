#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""滚动 IC 选题 + 浅层 MLP 预测持有期收益（多头，不中性化）。

用法：--strategy mlp --factor a,b,c
因子名前加 '-' 仅在 --select 0 时生效；select>0 时方向由滚动 RankIC 决定。

每 --refit 个交易日用过去 lookback（默认一年）已实现样本拟合一次，
同一模型沿用到下次重训。调仓日只用冻结模型对当天因子打分。

默认 horizon=5、rebalance=20、refit=20、lookback=252。
未安装 scikit-learn 或样本不足时回退等权 z-score。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
from Strategies.icir import spearman_ic
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli

logger = logging.getLogger("Strategies.mlp")

DEFAULT_N = 50
DEFAULT_LOOKBACK = 252
DEFAULT_HORIZON = 5
DEFAULT_REFIT = 20
DEFAULT_SELECT = 8
DEFAULT_BLEND = 0.80
DEFAULT_SMOOTH = 0.70
DEFAULT_HIDDEN = (32, 16)
DEFAULT_MIN_ROWS = 400
MAX_GROUP = 256
MAX_GROUPS = 80
MIN_IC_OBS = 30
WINSOR = 2.5


@dataclass
class _MlpFit:
    model: object
    keep: List[int]
    signs: List[float]
    names: List[str]


def _try_mlp():
    try:
        from sklearn.neural_network import MLPRegressor
    except ImportError:
        return None
    return MLPRegressor


def _parse_hidden(text: object) -> Tuple[int, ...]:
    if text is None or str(text).strip() == "":
        return DEFAULT_HIDDEN
    parts = [int(p.strip()) for p in str(text).split(",") if p.strip()]
    if not parts or any(p < 1 for p in parts):
        raise ValueError("mlp --hidden 须是正整数，如 32,16")
    return tuple(parts)


def _day_stride(horizon: int) -> int:
    """按持有期抽不重叠截面，一年 lookback、horizon=5 大约 50 组。"""
    return max(1, int(horizon))


def _feature_z(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    uni: np.ndarray,
    i: int,
) -> np.ndarray:
    cols = []
    for panel, sign in zip(factor_hist, signs):
        raw = np.asarray(panel[i], dtype=np.float64) * float(sign)
        cols.append(cross_section_z(raw, uni))
    if not cols:
        return np.empty((int(uni.shape[0]), 0), dtype=np.float64)
    return np.column_stack(cols)


def _mean_rank_ic(
    factor_hist: Sequence[np.ndarray],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
) -> np.ndarray:
    """每个因子 vs 已实现 horizon 日收益的日均 RankIC（因子取原始方向）。"""
    n_f = len(factor_hist)
    close = np.asarray(close_hist, dtype=np.float64)
    mask = np.asarray(mask_hist, dtype=bool)
    ics = np.full((int(lookback), n_f), np.nan, dtype=np.float64)
    for i in range(int(lookback)):
        j = i + int(horizon)
        c0 = close[i]
        c1 = close[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            y = c1 / c0 - 1.0
        y = np.where(np.isfinite(c0) & (c0 > 0) & np.isfinite(c1) & (c1 > 0), y, np.nan)
        uni = mask[i]
        for k, panel in enumerate(factor_hist):
            z = cross_section_z(np.asarray(panel[i], dtype=np.float64), uni)
            ics[i, k] = spearman_ic(z, y, min_obs=MIN_IC_OBS)
    return np.array([np.nanmean(ics[:, k]) for k in range(n_f)], dtype=np.float64)


def _pick_factors(
    names: Sequence[str],
    user_signs: Sequence[float],
    mean_ic: np.ndarray,
    select: int,
) -> Tuple[List[int], List[float]]:
    """select>0：按 |IC| 取前 select，符号取 IC；select=0：全用，符号用用户指定。"""
    n_f = len(names)
    if int(select) <= 0:
        return list(range(n_f)), [float(s) for s in user_signs]
    order = np.argsort(-np.abs(np.where(np.isfinite(mean_ic), mean_ic, 0.0)))
    keep: List[int] = []
    signs: List[float] = []
    for idx in order:
        ic = float(mean_ic[int(idx)])
        if not np.isfinite(ic) or abs(ic) < 1e-8:
            continue
        keep.append(int(idx))
        signs.append(1.0 if ic > 0 else -1.0)
        if len(keep) >= int(select):
            break
    return keep, signs


def stack_mlp_xy(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """已实现样本：X=因子截面 z，y=持有期收益截面 z（去极值）。"""
    close = np.asarray(close_hist, dtype=np.float64)
    mask = np.asarray(mask_hist, dtype=bool)
    step = _day_stride(horizon)
    rng = np.random.default_rng(int(seed) % (2**31))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    last = int(lookback) - 1
    i = last
    n_f = len(factor_hist)
    while i >= 0:
        j = i + int(horizon)
        c0 = close[i]
        c1 = close[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_y = c1 / c0 - 1.0
        raw_y = np.where(
            np.isfinite(c0) & (c0 > 0) & np.isfinite(c1) & (c1 > 0), raw_y, np.nan
        )
        uni = mask[i]
        y = cross_section_z(raw_y, uni)
        x = _feature_z(factor_hist, signs, uni, i)
        row_ok = uni & np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        n = int(row_ok.sum())
        if n < 8:
            i -= step
            continue
        xx = x[row_ok]
        yy = y[row_ok]
        mu = float(yy.mean())
        sd = float(yy.std(ddof=0))
        if sd > 1e-12:
            yy = np.clip(yy, mu - WINSOR * sd, mu + WINSOR * sd)
        if n > MAX_GROUP:
            pick = rng.choice(n, size=MAX_GROUP, replace=False)
            pick.sort()
            xx = xx[pick]
            yy = yy[pick]
        xs.append(xx)
        ys.append(yy)
        i -= step
    if not xs:
        return np.empty((0, n_f), dtype=np.float64), np.empty(0, dtype=np.float64)
    if len(xs) > MAX_GROUPS:
        xs = xs[:MAX_GROUPS]
        ys = ys[:MAX_GROUPS]
    return np.vstack(xs), np.concatenate(ys)


def _apply_fit(
    fit: _MlpFit,
    factor_hist: Sequence[np.ndarray],
    universe: np.ndarray,
    blend: float,
) -> np.ndarray:
    uni = np.asarray(universe, dtype=bool)
    picked_today = [np.asarray(factor_hist[i])[-1] for i in fit.keep]
    eq = equal_weight_scores(picked_today, fit.signs, uni)
    if fit.model is None:
        return eq
    picked_hist = [factor_hist[i] for i in fit.keep]
    z = _feature_z(picked_hist, fit.signs, uni, -1)
    pred = np.full(uni.shape, np.nan, dtype=np.float64)
    ok = uni & np.all(np.isfinite(z), axis=1)
    if int(ok.sum()) == 0:
        return eq
    pred[ok] = np.asarray(fit.model.predict(z[ok]), dtype=np.float64)
    pz = cross_section_z(pred, ok)
    b = float(np.clip(blend, 0.0, 1.0))
    both = np.isfinite(pz) & np.isfinite(eq)
    out = eq.copy()
    out[both] = (1.0 - b) * eq[both] + b * pz[both]
    only = np.isfinite(pz) & ~np.isfinite(eq)
    out[only] = pz[only]
    return out


def fit_mlp(
    factor_hist: Sequence[np.ndarray],
    names: Sequence[str],
    user_signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    select: int,
    min_rows: int,
    hidden: Tuple[int, ...],
) -> Optional[_MlpFit]:
    if not factor_hist:
        return None
    need = int(lookback) + int(horizon)
    if any(np.asarray(p).shape[0] < need for p in factor_hist):
        return None
    mean_ic = _mean_rank_ic(
        factor_hist, close_hist, mask_hist, lookback=lookback, horizon=horizon
    )
    keep, signs = _pick_factors(names, user_signs, mean_ic, select)
    if not keep:
        return None
    picked_hist = [factor_hist[i] for i in keep]
    picked_names = [str(names[i]) for i in keep]
    bits = [
        f"{('+' if s > 0 else '-')}{picked_names[j]}({mean_ic[keep[j]]:.3f})"
        for j, s in enumerate(signs)
    ]
    logger.info("mlp 重训选用 %s", " ".join(bits))

    mlp_cls = _try_mlp()
    if mlp_cls is None:
        logger.warning("未安装 scikit-learn，mlp 回退等权 z-score")
        return _MlpFit(model=None, keep=keep, signs=signs, names=picked_names)

    x, y = stack_mlp_xy(
        picked_hist,
        signs,
        close_hist,
        mask_hist,
        lookback=lookback,
        horizon=horizon,
        seed=42 + int(lookback) + int(horizon),
    )
    if x.shape[0] < max(int(min_rows), 80 * len(picked_hist)):
        return _MlpFit(model=None, keep=keep, signs=signs, names=picked_names)

    kwargs = dict(
        hidden_layer_sizes=tuple(int(h) for h in hidden),
        activation="relu",
        solver="adam",
        alpha=1.0e-3,
        learning_rate_init=0.01,
        max_iter=80,
        random_state=42,
        batch_size=min(256, int(x.shape[0])),
        verbose=False,
    )
    if x.shape[0] >= 200:
        kwargs.update(early_stopping=True, validation_fraction=0.15, n_iter_no_change=8)
    try:
        model = mlp_cls(**kwargs)
        model.fit(x, y)
    except Exception:
        logger.exception("mlp 拟合失败，回退等权")
        return _MlpFit(model=None, keep=keep, signs=signs, names=picked_names)
    return _MlpFit(model=model, keep=keep, signs=signs, names=picked_names)


class MlpTopK(Strategy):
    name = "mlp"

    @classmethod
    def from_cli(cls, args, allocator):
        if not args.factor:
            raise SystemExit("mlp 需要 --factor 作为候选池，逗号分隔")
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
        select = int(
            getattr(args, "select", None)
            if getattr(args, "select", None) is not None
            else DEFAULT_SELECT
        )
        blend = float(
            getattr(args, "blend", None)
            if getattr(args, "blend", None) is not None
            else DEFAULT_BLEND
        )
        smooth = float(
            getattr(args, "smooth", None)
            if getattr(args, "smooth", None) is not None
            else DEFAULT_SMOOTH
        )
        try:
            hidden = _parse_hidden(getattr(args, "hidden", None))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if lookback < 40:
            raise SystemExit("mlp --lookback 至少为 40（默认 252）")
        if horizon < 1:
            raise SystemExit("mlp --horizon 至少为 1（默认 5）")
        if refit < 1:
            raise SystemExit("mlp --refit 至少为 1（默认 20，同一模型沿用天数）")
        if select < 0:
            raise SystemExit("mlp --select 须 >= 0（0=不用 IC 选题，全用 --factor）")
        if not 0.0 <= blend <= 1.0:
            raise SystemExit("mlp --blend 须在 0 与 1 之间")
        if not 0.0 <= smooth < 1.0:
            raise SystemExit("mlp --smooth 须在 [0, 1)")
        return cls(
            legs,
            n=n,
            allocator=allocator,
            rebalance=spec,
            lookback=lookback,
            horizon=horizon,
            refit=refit,
            select=select,
            blend=blend,
            smooth=smooth,
            hidden=hidden,
        )

    @classmethod
    def cli_fields(cls):
        return [
            cli_field("factor", "候选因子池", "str", required=True, placeholder="a,b,c"),
            n_field(DEFAULT_N),
            rebalance_field("20", placeholder="daily / 5 / 20"),
            cli_field("lookback", "训练回看天数，默认一年", "int", default=DEFAULT_LOOKBACK),
            cli_field("horizon", "持有期，5或1", "int", default=DEFAULT_HORIZON),
            cli_field("refit", "同一模型沿用交易日数", "int", default=DEFAULT_REFIT),
            cli_field("select", "按|IC|取前几个，0=全用", "int", default=DEFAULT_SELECT),
            cli_field("blend", "MLP分权重", "float", default=DEFAULT_BLEND),
            cli_field("smooth", "上期分数平滑", "float", default=DEFAULT_SMOOTH),
            cli_field("hidden", "隐层，如32,16", "str", default="32,16"),
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
        tag = "mlp_" + "_".join(parts)
        spec = parse_rebalance(getattr(args, "rebalance", None) or "20")
        tag += spec.tag_suffix()
        lookback = int(getattr(args, "lookback", None) or DEFAULT_LOOKBACK)
        horizon = int(getattr(args, "horizon", None) or DEFAULT_HORIZON)
        refit = DEFAULT_REFIT if getattr(args, "refit", None) is None else int(args.refit)
        select = (
            DEFAULT_SELECT
            if getattr(args, "select", None) is None
            else int(args.select)
        )
        blend = DEFAULT_BLEND if getattr(args, "blend", None) is None else float(args.blend)
        hid = _parse_hidden(getattr(args, "hidden", None))
        hid_s = "x".join(str(h) for h in hid)
        tag += (
            f"_mlp{lookback}h{horizon}rf{refit}s{select}"
            f"rk{int(round(blend * 100))}{hid_s}"
        )
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
        select: int = DEFAULT_SELECT,
        min_rows: int = DEFAULT_MIN_ROWS,
        blend: float = DEFAULT_BLEND,
        smooth: float = DEFAULT_SMOOTH,
        hidden: Sequence[int] = DEFAULT_HIDDEN,
    ):
        legs: List[FactorLeg] = []
        for item in factors:
            if isinstance(item, str):
                legs.extend(parse_factor_list(item))
            else:
                name, sign = item
                legs.append((str(name), float(sign)))
        if not legs:
            raise ValueError("mlp 至少需要一个候选因子")
        self.legs = legs
        self.n = max(1, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = (
            parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        )
        self.lookback = max(40, int(lookback))
        self.horizon = max(1, int(horizon))
        self.refit = max(1, int(refit))
        self.select = max(0, int(select))
        self.min_rows = max(80, int(min_rows))
        self.blend = float(np.clip(blend, 0.0, 1.0))
        self.smooth = float(np.clip(smooth, 0.0, 0.95))
        self.hidden = tuple(int(h) for h in hidden) or DEFAULT_HIDDEN
        self._gate = RebalanceGate(self.rebalance)
        self.last_model: Optional[object] = None
        self.last_picked: List[str] = []
        self._fit: Optional[_MlpFit] = None
        self._fitted_t: Optional[int] = None
        self._prev_score: Optional[np.ndarray] = None

    @property
    def factor_names(self) -> List[str]:
        return [name for name, _ in self.legs]

    def _need_refit(self, t: int) -> bool:
        if self._fit is None or self._fitted_t is None:
            return True
        return int(t) - int(self._fitted_t) >= int(self.refit)

    def _mlp_score(self, ctx: DayContext) -> np.ndarray:
        need = self.lookback + self.horizon
        close = ctx.history("close", need)
        t1 = int(ctx._t) + 1
        t0 = max(0, t1 - need)
        mask = np.asarray(ctx._store.mask[t0:t1], dtype=bool)
        if mask.shape[0] < need:
            pad = np.zeros((need - mask.shape[0], mask.shape[1]), dtype=bool)
            mask = np.vstack([pad, mask])
        names = [name for name, _ in self.legs]
        signs = [sign for _, sign in self.legs]
        factor_hist = [ctx.history(name, need) for name in names]
        today_uni = np.asarray(mask[-1], dtype=bool)
        if self._need_refit(ctx._t):
            fitted = fit_mlp(
                factor_hist,
                names,
                signs,
                close,
                mask,
                lookback=self.lookback,
                horizon=self.horizon,
                select=self.select,
                min_rows=self.min_rows,
                hidden=self.hidden,
            )
            if fitted is not None:
                self._fit = fitted
                self._fitted_t = int(ctx._t)
                self.last_model = fitted.model
                self.last_picked = list(fitted.names)
        if self._fit is None:
            score = equal_weight_scores(
                [np.asarray(p)[-1] for p in factor_hist], signs, today_uni
            )
        else:
            score = _apply_fit(self._fit, factor_hist, today_uni, self.blend)
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

        values = self._mlp_score(ctx)
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
