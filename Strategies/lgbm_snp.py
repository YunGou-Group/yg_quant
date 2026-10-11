#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LGBM 奖励路径：Size+NLSIZE 残差特征 + 残差标签 + 市值分层组内排序。

用法：--strategy lgbm_snp --factor a,b,-c --universe zz1000
不改 lgbm.py / lgbm_sn.py。相对 lgbm_sn 多做三件事：

1. 标签是持有期收益对 Size+NLSIZE 的截面 OLS 残差，再在组内打 0–4 档
2. LambdaRank 的 group 是当日市值五分位，小盘与大盘不在同一组里比
3. 选出 TopN 后把多头平均 Size 暴露压进 [--max-size-abs, +max-size-abs]（默认 0.35）

默认 --blend 1、--smooth 0.8。不要把 basics_market_cap 当交易因子。
股票池用引擎 --universe zz1000（不要 all）。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import cli_field, n_field, rebalance_field
from Strategies._factor_combo import cross_section_z, equal_weight_scores, parse_factor_list
from Strategies._style_neutralize import (
    neutralize_panel_on_expo,
    neutralize_scores,
    size_nlsize_expo_on_store,
)
from Strategies.lgbm import (
    DEFAULT_HORIZON,
    DEFAULT_LOOKBACK,
    DEFAULT_MIN_ROWS,
    DEFAULT_N,
    DEFAULT_REFIT,
    MAX_GROUP,
    MAX_GROUPS,
    MIN_GROUPS,
    _cap_group,
    _day_stride,
    _fit_ranker,
    _try_lightgbm,
)
from Strategies.lgbm_sn import DEFAULT_BLEND, DEFAULT_SMOOTH, LgbmSizeNlTopK

DEFAULT_BUCKETS = 5
DEFAULT_MAX_SIZE_ABS = 0.35
MIN_BUCKET = 8


def _size_buckets(size: np.ndarray, uni: np.ndarray, n_buckets: int) -> np.ndarray:
    """当日可交易股票按 Size 分成 1..n 档；0 表示不入组。"""
    size = np.asarray(size, dtype=np.float64)
    n = int(size.shape[0])
    out = np.zeros(n, dtype=np.int32)
    ok = np.asarray(uni, dtype=bool) & np.isfinite(size)
    m = int(ok.sum())
    n_b = max(2, int(n_buckets))
    if m < max(MIN_BUCKET, n_b * 4):
        return out
    vals = size[ok]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(m, dtype=np.float64)
    ranks[order] = np.arange(m, dtype=np.float64)
    b = np.floor(ranks * n_b / m).astype(np.int32)
    b = np.clip(b, 0, n_b - 1) + 1
    out[np.flatnonzero(ok)] = b
    return out


def stack_snp_xy(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    expo_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    n_buckets: int = DEFAULT_BUCKETS,
    stride: Optional[int] = None,
    group_cap: int = MAX_GROUP,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """已实现样本：y 为 Size+NLSIZE 残差收益；每个交易日按市值分位拆成多个 LambdaRank 组。"""
    close = np.asarray(close_hist, dtype=np.float64)
    mask = np.asarray(mask_hist, dtype=bool)
    expo = np.asarray(expo_hist, dtype=np.float64)
    if expo.ndim != 3 or expo.shape[2] < 1:
        n_f = len(factor_hist)
        return (
            np.empty((0, n_f), dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    step = int(stride) if stride is not None else _day_stride(horizon)
    rng = np.random.default_rng(int(seed) % (2**31))
    n_b = max(2, int(n_buckets))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    groups: List[int] = []
    last = int(lookback) - 1
    i = last
    n_days = 0
    while i >= 0:
        j = i + int(horizon)
        c0 = close[i]
        c1 = close[j]
        with np.errstate(invalid="ignore", divide="ignore"):
            y_raw = c1 / c0 - 1.0
        y_raw = np.where(np.isfinite(c0) & (c0 > 0) & np.isfinite(c1) & (c1 > 0), y_raw, np.nan)
        uni = mask[i]
        y = neutralize_scores(y_raw, expo[i], uni)
        buckets = _size_buckets(expo[i, :, 0], uni, n_b)
        cols = []
        for panel, sign in zip(factor_hist, signs):
            raw = np.asarray(panel[i], dtype=np.float64) * float(sign)
            cols.append(cross_section_z(raw, uni))
        x = np.column_stack(cols)
        row_ok = uni & np.isfinite(y) & np.all(np.isfinite(x), axis=1) & (buckets > 0)
        day_added = False
        for b in range(1, n_b + 1):
            sel = row_ok & (buckets == b)
            if int(sel.sum()) < MIN_BUCKET:
                continue
            xx, yy = _cap_group(x[sel], y[sel], int(group_cap), rng)
            xs.append(xx)
            ys.append(yy)
            groups.append(int(yy.shape[0]))
            day_added = True
        if day_added:
            n_days += 1
            if n_days >= MAX_GROUPS:
                break
        i -= step
    if not xs:
        n_f = len(factor_hist)
        return (
            np.empty((0, n_f), dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    return np.vstack(xs), np.concatenate(ys), np.asarray(groups, dtype=np.int32)


def combine_scores_lgbm_snp(
    factor_hist: Sequence[np.ndarray],
    signs: Sequence[float],
    close_hist: np.ndarray,
    mask_hist: np.ndarray,
    expo_hist: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    n_buckets: int = DEFAULT_BUCKETS,
    min_rows: int = DEFAULT_MIN_ROWS,
    blend: float = DEFAULT_BLEND,
    model: Optional[object] = None,
) -> Tuple[np.ndarray, Optional[object]]:
    """残差标签 + 市值分层 LambdaRank；失败则等权回退。传入 model 则只打分不重训。"""
    need = int(lookback) + int(horizon)
    today_uni = np.asarray(mask_hist[-1], dtype=bool)
    today_panels = [np.asarray(p)[-1] for p in factor_hist]
    eq = equal_weight_scores(today_panels, signs, today_uni)
    if not factor_hist:
        return eq, None
    fitted = model
    if fitted is None:
        for panel in factor_hist:
            if np.asarray(panel).shape[0] < need:
                return eq, None
        lgb = _try_lightgbm()
        if lgb is None:
            return eq, None
        x, y, groups = stack_snp_xy(
            factor_hist,
            signs,
            close_hist,
            mask_hist,
            expo_hist,
            lookback=lookback,
            horizon=horizon,
            n_buckets=n_buckets,
            seed=42 + int(lookback) + int(horizon) + int(n_buckets),
        )
        need_rows = max(int(min_rows), 80 * len(factor_hist))
        min_g = max(int(MIN_GROUPS), int(n_buckets) * 4)
        if groups.size < min_g or x.shape[0] < need_rows:
            return eq, None
        try:
            fitted = _fit_ranker(lgb, x, y, groups)
        except Exception:
            return eq, None

    z_cols = []
    for panel, sign in zip(today_panels, signs):
        raw = np.asarray(panel, dtype=np.float64) * float(sign)
        z_cols.append(cross_section_z(raw, today_uni))
    z = np.column_stack(z_cols)
    tree = np.full(today_uni.shape, np.nan, dtype=np.float64)
    ok = today_uni & np.all(np.isfinite(z), axis=1)
    if int(ok.sum()) == 0:
        return eq, fitted
    pred = np.asarray(fitted.predict(z[ok]), dtype=np.float64)
    tree[ok] = pred
    tz = cross_section_z(tree, ok)
    b = float(np.clip(blend, 0.0, 1.0))
    both = np.isfinite(tz) & np.isfinite(eq)
    out = eq.copy()
    out[both] = (1.0 - b) * eq[both] + b * tz[both]
    only_tree = np.isfinite(tz) & ~np.isfinite(eq)
    out[only_tree] = tz[only_tree]
    return out, fitted


def _constrain_long_size(
    pick: np.ndarray,
    score: np.ndarray,
    size: np.ndarray,
    tradable: np.ndarray,
    cap: float,
) -> np.ndarray:
    """多头 TopN：用次优替换把平均 Size 暴露压进 ±cap。不做空头对冲。"""
    cap = float(cap)
    if cap <= 0 or pick.size == 0:
        return pick
    ids = [int(i) for i in pick]
    used = set(ids)
    order = np.argsort(-np.where(tradable, score, -np.inf), kind="mergesort")

    def mean_sz(cur: Sequence[int]) -> float:
        s = np.asarray([size[i] for i in cur], dtype=np.float64)
        s = s[np.isfinite(s)]
        return float(np.mean(s)) if s.size else 0.0

    k = len(ids)
    for _ in range(k * 3):
        m = mean_sz(ids)
        if abs(m) <= cap:
            break
        if m < -cap:
            drop = min(ids, key=lambda i: size[i] if np.isfinite(size[i]) else -np.inf)
            add = None
            for j in order:
                j = int(j)
                if j in used or not tradable[j]:
                    continue
                if np.isfinite(size[j]) and (
                    not np.isfinite(size[drop]) or size[j] > size[drop]
                ):
                    add = j
                    break
        else:
            drop = max(ids, key=lambda i: size[i] if np.isfinite(size[i]) else np.inf)
            add = None
            for j in order:
                j = int(j)
                if j in used or not tradable[j]:
                    continue
                if np.isfinite(size[j]) and (
                    not np.isfinite(size[drop]) or size[j] < size[drop]
                ):
                    add = j
                    break
        if add is None:
            break
        ids.remove(drop)
        used.remove(drop)
        ids.append(add)
        used.add(add)
    return np.asarray(ids, dtype=np.int64)


class LgbmSnPathTopK(LgbmSizeNlTopK):
    name = "lgbm_snp"

    @classmethod
    def from_cli(cls, args, allocator):
        inst = super().from_cli(args, allocator)
        n_buckets = getattr(args, "n_buckets", None)
        inst.n_buckets = max(
            2, int(n_buckets if n_buckets is not None else DEFAULT_BUCKETS)
        )
        mx = getattr(args, "max_size_abs", None)
        inst.max_size_abs = max(
            0.0, float(mx if mx is not None else DEFAULT_MAX_SIZE_ABS)
        )
        if inst.n_buckets > 20:
            raise SystemExit("lgbm_snp --n-buckets 须在 2 与 20 之间")
        return inst

    @classmethod
    def cli_fields(cls):
        return [
            cli_field("factor", "因子列表", "str", required=True, placeholder="a,b,-c"),
            n_field(DEFAULT_N),
            rebalance_field("20", placeholder="daily / weekly / 20"),
            cli_field("lookback", "LGBM回看天数", "int", default=DEFAULT_LOOKBACK),
            cli_field("horizon", "LGBM持有期", "int", default=DEFAULT_HORIZON),
            cli_field("refit", "同一模型沿用交易日数", "int", default=DEFAULT_REFIT),
            cli_field("blend", "树分权重", "float", default=DEFAULT_BLEND),
            cli_field("smooth", "上期分数平滑", "float", default=DEFAULT_SMOOTH),
            cli_field("n_buckets", "市值分层组数", "int", default=DEFAULT_BUCKETS),
            cli_field("max_size_abs", "多头平均Size暴露上限，0关闭", "float", default=DEFAULT_MAX_SIZE_ABS),
        ]

    @classmethod
    def run_tag(cls, args) -> str:
        blend = getattr(args, "blend", None)
        if blend is None:
            args.blend = DEFAULT_BLEND
        smooth = getattr(args, "smooth", None)
        if smooth is None:
            args.smooth = DEFAULT_SMOOTH
        tag = LgbmSizeNlTopK.run_tag(args)
        if tag.startswith("lgbmsn_"):
            tag = "lgbmsnp_" + tag[len("lgbmsn_") :]
        else:
            tag = "lgbmsnp_" + tag
        n_b = getattr(args, "n_buckets", None)
        n_b = DEFAULT_BUCKETS if n_b is None else int(n_b)
        mx = getattr(args, "max_size_abs", None)
        mx = DEFAULT_MAX_SIZE_ABS if mx is None else float(mx)
        return tag + f"b{n_b}sz{int(round(mx * 100))}"

    def __init__(
        self,
        *args,
        n_buckets: int = DEFAULT_BUCKETS,
        max_size_abs: float = DEFAULT_MAX_SIZE_ABS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_buckets = max(2, int(n_buckets))
        self.max_size_abs = max(0.0, float(max_size_abs))

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
        expo = size_nlsize_expo_on_store(ctx._store)[t0:t1]
        if expo.shape[0] < need:
            pad = np.full(
                (need - expo.shape[0], expo.shape[1], expo.shape[2]),
                np.nan,
                dtype=np.float64,
            )
            expo = np.concatenate([pad, expo], axis=0)
        factor_hist = [
            neutralize_panel_on_expo(panel, expo, mask) for panel in factor_hist
        ]
        reuse = None if self._need_refit(ctx._t) else self.last_model
        score, fitted = combine_scores_lgbm_snp(
            factor_hist,
            signs,
            close,
            mask,
            expo,
            lookback=self.lookback,
            horizon=self.horizon,
            n_buckets=self.n_buckets,
            min_rows=self.min_rows,
            blend=self.blend,
            model=reuse,
        )
        if self._need_refit(ctx._t):
            if fitted is not None:
                self.last_model = fitted
                self._fitted_t = int(ctx._t)
            elif self.last_model is not None:
                score, _ = combine_scores_lgbm_snp(
                    factor_hist,
                    signs,
                    close,
                    mask,
                    expo,
                    lookback=self.lookback,
                    horizon=self.horizon,
                    n_buckets=self.n_buckets,
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
        if self.max_size_abs > 0:
            size = np.asarray(
                size_nlsize_expo_on_store(ctx._store)[int(ctx._t), :, 0],
                dtype=np.float64,
            )
            pick = _constrain_long_size(pick, values, size, tradable, self.max_size_abs)
        weights = self.allocator.size(ctx, pick)
        return self._gate.remember(
            TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
        )
