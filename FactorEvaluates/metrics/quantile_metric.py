#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分层收益 / 组内 IC / 命中率 / 换手。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric
from ..context import BatchEvalContext, EvalContext
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec
from ..matrix_utils import assign_quantiles, pearson_pairwise, spearman_pairwise

N_QUANTILES_PARAM = ParamSpec(
    name="n_quantiles",
    type="int",
    label="分层组数",
    default=5,
    min=2,
    max=20,
    scope="metric",
)

MIN_PER_BIN_PARAM = ParamSpec(
    name="min_stock_per_bin",
    type="int",
    label="每组最少股票数",
    default=5,
    min=1,
    max=200,
    scope="metric",
)


def _ensure_labels(batch_ctx: BatchEvalContext, n_q: int) -> np.ndarray:
    labels = batch_ctx.intermediates.get("quantile_labels")
    if labels is not None:
        return labels
    labels = assign_quantiles(
        batch_ctx.masked_factors(), n_q
    )
    batch_ctx.intermediates["quantile_labels"] = labels
    return labels


def _group_mean(values: np.ndarray, labels: np.ndarray, qi: int, min_count: int) -> np.ndarray:
    p = labels.shape[1] if labels.ndim == 2 else 1
    if labels.ndim == 1:
        labels = labels[:, None]
        values = values[:, None] if values.ndim == 1 else values
    out = np.full(p, np.nan, dtype=np.float64)
    if values.ndim == 1:
        values = np.broadcast_to(values[:, None], labels.shape)
    for j in range(p):
        sel = (labels[:, j] == qi) & np.isfinite(values[:, j])
        if int(sel.sum()) < min_count:
            continue
        out[j] = float(np.mean(values[sel, j]))
    return out


class QuantileMetric(BaseMetric):
    name = "quantile"
    dimension = "预测力"
    description = "按因子截面分层等权，看高分组是否比低分组更能赚钱，以及收益是否随因子值单调"
    cost = "panel"
    produces = ("quantile_labels",)
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM, MIN_PER_BIN_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (
            FieldDoc("spread", "预测力", "分层价差", "最高组与最低组的日均 N 日收益差"),
            FieldDoc("monotonicity", "预测力", "单调性", "分组序号与组收益的 Spearman 相关"),
            FieldDoc("n_quantiles", "参数", "分层组数", "截面分位的组数"),
            FieldDoc("n_days", "样本", "有效天数", "各组收益都齐全的交易日数"),
            FieldDoc("horizon", "参数", "持有期 N", "远期收益的持有交易日数"),
        )

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        min_per = int(params.get("min_stock_per_bin", 5))
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        p = f.shape[1]
        empty = {f"quantile_returns_Q{i}": np.full(p, np.nan) for i in range(1, n_q + 1)}
        empty["quantile_spread"] = np.full(p, np.nan)
        if r is None:
            return empty
        labels = _ensure_labels(batch_ctx, n_q)
        r_mat = np.broadcast_to(r[:, None], f.shape)
        out = {}
        cols = []
        for qi in range(1, n_q + 1):
            vals = _group_mean(r_mat, labels, qi, min_per)
            out[f"quantile_returns_Q{qi}"] = vals
            cols.append(vals)
        stacked = np.vstack(cols)
        complete = np.isfinite(stacked).all(axis=0)
        spread = stacked[-1] - stacked[0]
        out["quantile_spread"] = np.where(complete, spread, np.nan)
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        n_quantiles = int(params.get("n_quantiles", 5))
        min_per_bin = int(params.get("min_stock_per_bin", 5))
        horizon = int(ctx.horizon)
        factor = ctx.masked_factor()
        fwd = ctx.masked_fwd_ret()
        daily = pd.DataFrame(
            np.nan,
            index=factor.index,
            columns=[f"Q{i + 1}" for i in range(n_quantiles)],
        )
        label_rows = []
        for date in factor.index:
            row_f = factor.loc[date]
            row_r = fwd.loc[date]
            valid = row_f.notna() & row_r.notna()
            labels_s = pd.Series(np.nan, index=factor.columns)
            if int(valid.sum()) >= n_quantiles * min_per_bin:
                f_valid = row_f[valid].to_numpy(dtype=np.float64)
                labels = assign_quantiles(f_valid, n_quantiles)
                labels_s.loc[row_f.index[valid]] = labels.astype(float)
                r_valid = row_r[valid].to_numpy(dtype=np.float64)
                for qi in range(1, n_quantiles + 1):
                    sel = labels == qi
                    if int(sel.sum()) < min_per_bin:
                        continue
                    daily.loc[date, f"Q{qi}"] = float(np.mean(r_valid[sel]))
            label_rows.append(labels_s)
        ctx.intermediates["quantile_labels"] = pd.DataFrame(label_rows, index=factor.index)

        complete = daily.dropna(how="any")
        horizon_ret = complete.clip(lower=-0.999, upper=10.0)
        daily = daily.clip(lower=-0.999, upper=10.0)
        one_day = (1.0 + horizon_ret) ** (1.0 / horizon) - 1.0
        nav = (1.0 + one_day).cumprod()
        spread = complete.iloc[:, -1] - complete.iloc[:, 0] if not complete.empty else pd.Series(dtype=float)

        mono_scores = []
        ranks = np.arange(1, n_quantiles + 1, dtype=np.float64)
        for date in complete.index:
            vals = complete.loc[date].to_numpy(dtype=np.float64)
            if float(np.std(vals)) == 0:
                continue
            score = self._spearman(ranks, vals)
            if np.isfinite(score):
                mono_scores.append(score)

        series: Dict[str, pd.Series] = {}
        for i, col in enumerate(daily.columns):
            series[f"quantile_returns_{col}"] = daily.iloc[:, i].astype("float64")
            if not nav.empty:
                series[str(col)] = pd.Series(nav.iloc[:, i].to_numpy(dtype=np.float64), index=nav.index, name=str(col))
        series["spread"] = spread.astype("float64") if len(spread) else pd.Series(dtype="float64")
        spread_vals = spread.to_numpy(dtype=np.float64) if len(spread) else np.array([])
        return MetricResult(
            scalars={
                "spread": float(np.mean(spread_vals)) if spread_vals.size else float("nan"),
                "monotonicity": (
                    float(np.mean(np.asarray(mono_scores, dtype=np.float64)))
                    if mono_scores
                    else float("nan")
                ),
                "n_quantiles": n_quantiles,
                "n_days": int(len(complete)),
                "horizon": horizon,
            },
            series=series,
        )

    @staticmethod
    def _spearman(left: np.ndarray, right: np.ndarray) -> float:
        rx = QuantileMetric._rank(left)
        ry = QuantileMetric._rank(right)
        lx = rx - rx.mean()
        ly = ry - ry.mean()
        den = float(np.sqrt(np.sum(lx * lx) * np.sum(ly * ly)))
        if den == 0:
            return float("nan")
        return float(np.sum(lx * ly) / den)

    @staticmethod
    def _rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.shape[0], dtype=np.float64)
        ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.float64)
        return ranks


class QuantileICMetric(BaseMetric):
    name = "quantile_ic"
    dimension = "预测力"
    description = "各分位组内因子与收益的 Pearson IC"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "组内IC均值", "最高组 quantile_ic 的区间均值"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        p = f.shape[1]
        out = {f"quantile_ic_Q{qi}": np.full(p, np.nan) for qi in range(1, n_q + 1)}
        if r is None:
            return out
        labels = _ensure_labels(batch_ctx, n_q)
        min_obs = max(10, int(params.get("min_obs", 10)))
        for qi in range(1, n_q + 1):
            vals = np.full(p, np.nan)
            for j in range(p):
                sel = labels[:, j] == qi
                if int(sel.sum()) < min_obs:
                    continue
                vals[j] = pearson_pairwise(f[sel, j : j + 1], r[sel], min_obs=min_obs)[0]
            out[f"quantile_ic_Q{qi}"] = vals
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        from ..matrix_utils import apply_daily_matrix, result_from_daily

        daily = apply_daily_matrix(self, ctx, params)
        n_q = int(params.get("n_quantiles", 5))
        return result_from_daily(daily, primary=f"quantile_ic_Q{n_q}", extra_scalars={"horizon": ctx.horizon})


class QuantileRankICMetric(BaseMetric):
    name = "quantile_rank_ic"
    dimension = "预测力"
    description = "各分位组内因子与收益的 Spearman RankIC"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "组内RankIC均值", "最高组 quantile_rank_ic 的区间均值"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        p = f.shape[1]
        out = {f"quantile_rank_ic_Q{qi}": np.full(p, np.nan) for qi in range(1, n_q + 1)}
        if r is None:
            return out
        labels = _ensure_labels(batch_ctx, n_q)
        min_obs = max(10, int(params.get("min_obs", 10)))
        for qi in range(1, n_q + 1):
            vals = np.full(p, np.nan)
            for j in range(p):
                sel = labels[:, j] == qi
                if int(sel.sum()) < min_obs:
                    continue
                vals[j] = spearman_pairwise(f[sel, j : j + 1], r[sel], min_obs=min_obs)[0]
            out[f"quantile_rank_ic_Q{qi}"] = vals
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        from ..matrix_utils import apply_daily_matrix, result_from_daily

        daily = apply_daily_matrix(self, ctx, params)
        n_q = int(params.get("n_quantiles", 5))
        return result_from_daily(daily, primary=f"quantile_rank_ic_Q{n_q}", extra_scalars={"horizon": ctx.horizon})


class QuantileHitMetric(BaseMetric):
    name = "quantile_return_hit"
    dimension = "预测力"
    description = "各因子分位组里，收益处于截面顶/底分位的股票占比"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "顶组命中", "最高因子组命中收益顶组的平均比例"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        p = f.shape[1]
        out = {}
        for qi in range(1, n_q + 1):
            out[f"quantile_top_hit_Q{qi}"] = np.full(p, np.nan)
            out[f"quantile_bot_hit_Q{qi}"] = np.full(p, np.nan)
        if r is None:
            return out
        fq = _ensure_labels(batch_ctx, n_q)
        rq = assign_quantiles(r[:, None], n_q)[:, 0]
        top_r = rq == n_q
        bot_r = rq == 1
        for qi in range(1, n_q + 1):
            in_q = fq == qi
            cnt = in_q.sum(axis=0)
            top = (in_q & top_r[:, None]).sum(axis=0)
            bot = (in_q & bot_r[:, None]).sum(axis=0)
            out[f"quantile_top_hit_Q{qi}"] = np.where(cnt > 0, top / np.maximum(cnt, 1), np.nan)
            out[f"quantile_bot_hit_Q{qi}"] = np.where(cnt > 0, bot / np.maximum(cnt, 1), np.nan)
        out["best_in_best_ratio"] = out[f"quantile_top_hit_Q{n_q}"]
        out["worst_in_best_ratio"] = out[f"quantile_bot_hit_Q{n_q}"]
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        from ..matrix_utils import apply_daily_matrix, result_from_daily

        daily = apply_daily_matrix(self, ctx, params)
        n_q = int(params.get("n_quantiles", 5))
        return result_from_daily(daily, primary=f"quantile_top_hit_Q{n_q}", extra_scalars={"horizon": ctx.horizon})


class QuantileTurnoverMetric(BaseMetric):
    name = "quantile_turnover"
    dimension = "有效期"
    description = "相邻日分位组成员变动率 1 − overlap/total"
    cost = "derived"
    produces = ("quantile_turnover",)
    requires = ("quantile_labels",)

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "有效期", "顶组换手均值", "最高分位组日均换手"),)

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        labels = ctx.intermediates.get("quantile_labels")
        if not isinstance(labels, pd.DataFrame):
            raise ValueError("quantile_turnover 需要中间量 quantile_labels，请先运行 quantile")
        n_q = int(params.get("n_quantiles", 5))
        series: Dict[str, pd.Series] = {}
        for qi in range(1, n_q + 1):
            prev = None
            values = []
            for date in labels.index:
                members = set(labels.columns[labels.loc[date] == qi])
                if prev is None:
                    values.append(np.nan)
                else:
                    total = len(members | prev)
                    overlap = len(members & prev)
                    values.append(1.0 - overlap / total if total else np.nan)
                prev = members
            series[f"quantile_turnover_Q{qi}"] = pd.Series(values, index=labels.index, dtype="float64")
        ctx.intermediates["quantile_turnover"] = series
        top = series[f"quantile_turnover_Q{n_q}"]
        clean = top.dropna()
        return MetricResult(
            scalars={
                "mean": float(clean.mean()) if len(clean) else float("nan"),
                "n_days": int(len(clean)),
                "horizon": ctx.horizon,
            },
            series=series,
        )

    def compute_from_labels(
        self, labels: np.ndarray, params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", 5))
        n_dates, _, n_f = labels.shape
        out = {
            f"quantile_turnover_Q{q}": np.full((n_dates, n_f), np.nan, dtype=np.float32)
            for q in range(1, n_q + 1)
        }
        for t in range(1, n_dates):
            prev = labels[t - 1]
            cur = labels[t]
            for q in range(1, n_q + 1):
                pm = prev == q
                cm = cur == q
                inter = np.sum(pm & cm, axis=0)
                union = np.sum(pm | cm, axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[f"quantile_turnover_Q{q}"][t] = np.where(
                        union > 0, 1.0 - inter / union, np.nan
                    )
        return out


class QuantileFactorMeanMetric(BaseMetric):
    name = "quantile_factor_mean"
    dimension = "暴露度"
    description = "各分位组内因子值的截面均值"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "暴露度", "顶组因子均值", "最高分位组日均因子值"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f = batch_ctx.masked_factors()
        p = f.shape[1]
        labels = _ensure_labels(batch_ctx, n_q)
        out = {}
        for qi in range(1, n_q + 1):
            out[f"quantile_factor_mean_Q{qi}"] = _group_mean(f, labels, qi, min_count=1)
        if not out:
            for qi in range(1, n_q + 1):
                out[f"quantile_factor_mean_Q{qi}"] = np.full(p, np.nan)
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        from ..matrix_utils import apply_daily_matrix, result_from_daily

        daily = apply_daily_matrix(self, ctx, params)
        n_q = int(params.get("n_quantiles", 5))
        return result_from_daily(
            daily, primary=f"quantile_factor_mean_Q{n_q}", extra_scalars={"horizon": ctx.horizon}
        )


# A 股默认完整买卖约 11.2bp；首日只收买入约 3.1bp
DEFAULT_ROUND_TRIP_BPS = 11.2
DEFAULT_BUY_RATE = 0.00031
DEFAULT_SELL_RATE = 0.00081

COST_BPS_PARAM = ParamSpec(
    name="transaction_cost_bps",
    type="float",
    label="完整买卖费率 (bp)",
    default=DEFAULT_ROUND_TRIP_BPS,
    min=0.0,
    max=100.0,
    scope="metric",
)
BUY_RATE_PARAM = ParamSpec(
    name="transaction_buy_rate",
    type="float",
    label="买入费率",
    default=DEFAULT_BUY_RATE,
    min=0.0,
    max=0.01,
    scope="metric",
)
SELL_RATE_PARAM = ParamSpec(
    name="transaction_sell_rate",
    type="float",
    label="卖出费率",
    default=DEFAULT_SELL_RATE,
    min=0.0,
    max=0.01,
    scope="metric",
)
LIQUIDATE_PARAM = ParamSpec(
    name="liquidate_at_end",
    type="bool",
    label="末日清仓收费",
    default=False,
    scope="metric",
)


def _net_from_turnover(gross: np.ndarray, turnover: np.ndarray, cost_bps: float) -> np.ndarray:
    return np.asarray(gross, dtype=np.float64) - np.asarray(turnover, dtype=np.float64) * (
        float(cost_bps) / 10_000.0
    )


class QuantileReturnsAfterCostMetric(BaseMetric):
    name = "quantile_returns_after_cost"
    dimension = "预测力"
    description = "分位组收益按成员换手扣费：中间日 turnover×完整买卖费率，首日仅收买入费"
    cost = "derived"
    produces = ()
    requires = ("quantile_labels", "quantile_turnover")

    def params(self) -> Sequence[ParamSpec]:
        return (
            HORIZON_PARAM,
            UNIVERSE_PARAM,
            N_QUANTILES_PARAM,
            MIN_PER_BIN_PARAM,
            COST_BPS_PARAM,
            BUY_RATE_PARAM,
            SELL_RATE_PARAM,
            LIQUIDATE_PARAM,
        )

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "顶组费后收益均值", "最高分位组扣费后日均收益"),)

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        labels = ctx.intermediates.get("quantile_labels")
        if not isinstance(labels, pd.DataFrame):
            raise ValueError("quantile_returns_after_cost 需要 quantile_labels，请先运行 quantile")
        n_q = int(params.get("n_quantiles", 5))
        min_per = int(params.get("min_stock_per_bin", 5))
        cost_bps = float(params.get("transaction_cost_bps", DEFAULT_ROUND_TRIP_BPS))
        buy_rate = float(params.get("transaction_buy_rate", DEFAULT_BUY_RATE))
        sell_rate = float(params.get("transaction_sell_rate", DEFAULT_SELL_RATE))
        liquidate = bool(params.get("liquidate_at_end", False))

        turnover = ctx.intermediates.get("quantile_turnover")
        if not isinstance(turnover, dict):
            raise ValueError("quantile_returns_after_cost 需要 quantile_turnover，请先运行 quantile_turnover")

        factor = ctx.masked_factor()
        fwd = ctx.masked_fwd_ret()
        series: Dict[str, pd.Series] = {}
        for qi in range(1, n_q + 1):
            gross_vals = []
            for date in factor.index:
                row_f = factor.loc[date]
                row_r = fwd.loc[date]
                lab = labels.loc[date]
                sel = (lab == qi) & row_f.notna() & row_r.notna()
                if int(sel.sum()) < min_per:
                    gross_vals.append(np.nan)
                else:
                    gross_vals.append(float(row_r[sel].mean()))
            gross = pd.Series(gross_vals, index=factor.index, dtype="float64")
            to = turnover.get(f"quantile_turnover_Q{qi}")
            if to is None:
                to = pd.Series(np.nan, index=factor.index)
            net = gross.copy()
            for i, date in enumerate(gross.index):
                g = gross.iloc[i]
                if not np.isfinite(g):
                    net.iloc[i] = np.nan
                    continue
                if i == 0:
                    net.iloc[i] = g - buy_rate
                else:
                    t = to.iloc[i] if date in to.index else np.nan
                    if not np.isfinite(t):
                        net.iloc[i] = np.nan
                    else:
                        net.iloc[i] = float(_net_from_turnover(g, t, cost_bps))
                if liquidate and i == len(gross) - 1 and np.isfinite(net.iloc[i]):
                    net.iloc[i] = float(net.iloc[i] - (1.0 + g) * sell_rate)
            series[f"quantile_returns_after_cost_Q{qi}"] = net.astype("float64")

        top = series[f"quantile_returns_after_cost_Q{n_q}"]
        clean = top.dropna()
        return MetricResult(
            scalars={
                "mean": float(clean.mean()) if len(clean) else float("nan"),
                "n_days": int(len(clean)),
                "horizon": ctx.horizon,
                "transaction_cost_bps": cost_bps,
            },
            series=series,
        )

    def compute_crossday(
        self, arrays: Mapping[str, Any], params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", 5))
        cost_bps = float(params.get("transaction_cost_bps", DEFAULT_ROUND_TRIP_BPS))
        buy_rate = float(params.get("transaction_buy_rate", DEFAULT_BUY_RATE))
        sell_rate = float(params.get("transaction_sell_rate", DEFAULT_SELL_RATE))
        liquidate = bool(params.get("liquidate_at_end", False))
        out: Dict[str, Any] = {}
        for qi in range(1, n_q + 1):
            ret = arrays.get(f"quantile_returns_Q{qi}")
            to = arrays.get(f"quantile_turnover_Q{qi}")
            if ret is None or to is None:
                continue
            ret = np.asarray(ret, dtype=np.float64)
            to = np.asarray(to, dtype=np.float64)
            net = _net_from_turnover(ret, to, cost_bps)
            if net.shape[0] > 0:
                net = net.copy()
                net[0] = ret[0] - buy_rate
                if liquidate:
                    net[-1] = net[-1] - (1.0 + ret[-1]) * sell_rate
            out[f"quantile_returns_after_cost_Q{qi}"] = net.astype(np.float32, copy=False)
        return out