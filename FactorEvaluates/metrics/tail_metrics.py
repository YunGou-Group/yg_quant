#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""尾部命中、加权 IC、子样本 IC、配对正确率、非线性尾部。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ..base_metric import BaseMetric
from ..context import BatchEvalContext, EvalContext
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec
from ..matrix_utils import apply_daily_matrix, assign_quantiles, pearson_pairwise, rank_cols, result_from_daily, spearman_pairwise

_MIN = 10
N_QUANTILES_PARAM = ParamSpec(
    name="n_quantiles",
    type="int",
    label="分层组数",
    default=5,
    min=2,
    max=20,
    scope="metric",
)


def _fq_rq(batch_ctx: BatchEvalContext, n_q: int):
    f = batch_ctx.masked_factors()
    r = batch_ctx.masked_returns()
    fq = batch_ctx.intermediates.get("quantile_labels")
    if fq is None:
        fq = assign_quantiles(f, n_q)
        batch_ctx.intermediates["quantile_labels"] = fq
    rq = None
    if r is not None:
        rq = assign_quantiles(r[:, None], n_q)[:, 0]
    return f, r, fq, rq


class TailHitRateMetric(BaseMetric):
    name = "tail_hit_rate"
    dimension = "预测力"
    description = "收益 Top/Bottom 股票落在各因子分位组的占比"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "顶组命中均值", "收益顶组落在最高因子分位的比例"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f, r, fq, rq = _fq_rq(batch_ctx, n_q)
        p = f.shape[1]
        out: Dict[str, np.ndarray] = {}
        if rq is None:
            for qi in range(1, n_q + 1):
                out[f"tail_hit_rate_top_Q{qi}"] = np.full(p, np.nan)
                out[f"tail_hit_rate_bottom_Q{qi}"] = np.full(p, np.nan)
            return out
        top_r, bot_r = rq == n_q, rq == 1
        n_top, n_bot = int(top_r.sum()), int(bot_r.sum())
        for qi in range(1, n_q + 1):
            hit_top = np.full(p, np.nan)
            hit_bot = np.full(p, np.nan)
            if n_top >= _MIN:
                hit_top = (fq[top_r] == qi).sum(axis=0).astype(np.float64) / n_top
            if n_bot >= _MIN:
                hit_bot = (fq[bot_r] == qi).sum(axis=0).astype(np.float64) / n_bot
            out[f"tail_hit_rate_top_Q{qi}"] = hit_top
            out[f"tail_hit_rate_bottom_Q{qi}"] = hit_bot
        out["tail_hit_rate_top"] = out[f"tail_hit_rate_top_Q{n_q}"]
        out["tail_hit_rate_bottom"] = out["tail_hit_rate_bottom_Q1"]
        return out

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        n_q = int(params.get("n_quantiles", 5))
        return result_from_daily(daily, primary=f"tail_hit_rate_top_Q{n_q}", extra_scalars={"horizon": ctx.horizon})


class TailWeightedICMetric(BaseMetric):
    name = "tail_weighted_ic"
    dimension = "预测力"
    description = "V 型权重强调收益两端的加权 Pearson IC"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "尾部加权IC均值", "V 型加权 IC 的区间平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        f = batch_ctx.masked_factors()
        r = batch_ctx.masked_returns()
        p = f.shape[1]
        out = np.full(p, np.nan)
        if r is None:
            return {"tail_weighted_ic": out}
        valid_r = np.isfinite(r)
        nv = int(valid_r.sum())
        if nv < _MIN:
            return {"tail_weighted_ic": out}
        rv = r[valid_r]
        fv = f[valid_r]
        order = np.argsort(rv)
        ranks = np.empty(nv, dtype=np.float64)
        ranks[order] = np.arange(1, nv + 1, dtype=np.float64)
        w = 2.0 * np.abs(ranks / nv - 0.5)
        for j in range(p):
            col_ok = np.isfinite(fv[:, j])
            if int(col_ok.sum()) < _MIN:
                continue
            x, y, wv = fv[col_ok, j], rv[col_ok], w[col_ok]
            ws = float(wv.sum())
            if ws < 1e-15:
                continue
            xm = float(np.dot(wv, x) / ws)
            ym = float(np.dot(wv, y) / ws)
            xc, yc = x - xm, y - ym
            cov = float(np.dot(wv, xc * yc) / ws)
            den = np.sqrt(float(np.dot(wv, xc * xc) / ws) * float(np.dot(wv, yc * yc) / ws))
            out[j] = cov / den if den > 1e-15 else np.nan
        return {"tail_weighted_ic": out}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="tail_weighted_ic", extra_scalars={"horizon": ctx.horizon})


class TailSubsampleICMetric(BaseMetric):
    name = "tail_subsample_ic"
    dimension = "预测力"
    description = "收益 Top/Bottom 子集上的 RankIC"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "顶部分样本IC", "收益最高组内 RankIC 均值"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f, r, fq, rq = _fq_rq(batch_ctx, n_q)
        p = f.shape[1]
        ic_top = np.full(p, np.nan)
        ic_bot = np.full(p, np.nan)
        if r is None or rq is None:
            return {"tail_subsample_ic_top": ic_top, "tail_subsample_ic_bottom": ic_bot}
        for label, dest in ((n_q, ic_top), (1, ic_bot)):
            sel = rq == label
            if int(sel.sum()) < _MIN:
                continue
            dest[:] = spearman_pairwise(f[sel], r[sel], min_obs=_MIN)
        return {"tail_subsample_ic_top": ic_top, "tail_subsample_ic_bottom": ic_bot}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="tail_subsample_ic_top", extra_scalars={"horizon": ctx.horizon})


class TailPairAccuracyMetric(BaseMetric):
    name = "tail_pair_accuracy"
    dimension = "预测力"
    description = "收益 Top vs Bottom 股票对中因子排序正确的比例（Mann-Whitney）"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "尾部配对正确率", "极端对排序正确比例的平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f, r, fq, rq = _fq_rq(batch_ctx, n_q)
        p = f.shape[1]
        result = np.full(p, np.nan)
        if r is None or rq is None:
            return {"tail_pair_accuracy": result}
        top_mask, bot_mask = rq == n_q, rq == 1
        n_a, n_b = int(top_mask.sum()), int(bot_mask.sum())
        if n_a < _MIN or n_b < _MIN:
            return {"tail_pair_accuracy": result}
        combined = np.vstack([f[top_mask], f[bot_mask]])
        ranks = rank_cols(combined)
        sum_top = np.nansum(ranks[:n_a], axis=0)
        na_eff = np.isfinite(ranks[:n_a]).sum(axis=0)
        nb_eff = np.isfinite(ranks[n_a:]).sum(axis=0)
        enough = (na_eff >= _MIN) & (nb_eff >= _MIN)
        u = sum_top - na_eff * (na_eff + 1) / 2
        denom = np.maximum(na_eff * nb_eff, 1.0)
        result = np.where(enough, u / denom, np.nan)
        return {"tail_pair_accuracy": result}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="tail_pair_accuracy", extra_scalars={"horizon": ctx.horizon})


class TailQuantileSeparationMetric(BaseMetric):
    name = "tail_quantile_separation"
    dimension = "预测力"
    description = "收益 Top/Bottom 在因子分位上的总变差距离，方向无关"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "尾部分离度", "TVD 的区间平均"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f, r, fq, rq = _fq_rq(batch_ctx, n_q)
        p = f.shape[1]
        tvd = np.full(p, np.nan)
        if rq is None:
            return {"tail_quantile_separation": tvd}
        top_r, bot_r = rq == n_q, rq == 1
        n_top, n_bot = int(top_r.sum()), int(bot_r.sum())
        if n_top < _MIN or n_bot < _MIN:
            return {"tail_quantile_separation": tvd}
        p_top = np.stack([(fq[top_r] == qi).mean(axis=0) for qi in range(1, n_q + 1)])
        p_bot = np.stack([(fq[bot_r] == qi).mean(axis=0) for qi in range(1, n_q + 1)])
        tvd = 0.5 * np.sum(np.abs(p_top - p_bot), axis=0)
        return {"tail_quantile_separation": tvd}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="tail_quantile_separation", extra_scalars={"horizon": ctx.horizon})


class TailConcentrationMetric(BaseMetric):
    name = "tail_concentration"
    dimension = "预测力"
    description = "收益 Top/Bottom 在因子分位上的归一化条件熵集中度"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "顶部集中度", "收益顶组在因子分位上的集中度"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f, r, fq, rq = _fq_rq(batch_ctx, n_q)
        p = f.shape[1]
        nan = np.full(p, np.nan)
        if rq is None:
            return {"tail_concentration_top": nan, "tail_concentration_bottom": nan}
        h_max = np.log(n_q)

        def _conc(mask):
            n = int(mask.sum())
            if n < _MIN:
                return nan.copy()
            probs = np.stack([(fq[mask] == qi).mean(axis=0) for qi in range(1, n_q + 1)])
            with np.errstate(divide="ignore", invalid="ignore"):
                log_p = np.where(probs > 0, np.log(probs), 0.0)
            h = -np.sum(probs * log_p, axis=0)
            return 1.0 - h / h_max

        return {
            "tail_concentration_top": _conc(rq == n_q),
            "tail_concentration_bottom": _conc(rq == 1),
        }

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="tail_concentration_top", extra_scalars={"horizon": ctx.horizon})


class TailSubsampleEta2Metric(BaseMetric):
    name = "tail_subsample_eta2"
    dimension = "预测力"
    description = "收益 Top/Bottom 子集内因子分组对收益的 η²"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM, UNIVERSE_PARAM, N_QUANTILES_PARAM)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean", "预测力", "顶部η²", "收益顶组内因子分位解释收益方差的比例"),)

    def compute_matrix(self, batch_ctx: BatchEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        n_q = int(params.get("n_quantiles", batch_ctx.n_quantiles))
        f, r, fq, rq = _fq_rq(batch_ctx, n_q)
        p = f.shape[1]
        eta_top = np.full(p, np.nan)
        eta_bot = np.full(p, np.nan)
        if r is None or rq is None:
            return {"tail_subsample_eta2_top": eta_top, "tail_subsample_eta2_bottom": eta_bot}
        for label, dest in ((n_q, eta_top), (1, eta_bot)):
            sel = rq == label
            if int(sel.sum()) < _MIN:
                continue
            r_sub, fq_sub = r[sel], fq[sel]
            for j in range(p):
                valid = np.isfinite(r_sub) & (fq_sub[:, j] > 0)
                if int(valid.sum()) < _MIN:
                    continue
                rv, gj = r_sub[valid], fq_sub[valid, j]
                grand = rv.mean()
                ss_tot = float(((rv - grand) ** 2).sum())
                if ss_tot < 1e-30:
                    dest[j] = 0.0
                    continue
                ss_b = 0.0
                for k in range(1, n_q + 1):
                    in_k = gj == k
                    n_k = int(in_k.sum())
                    if n_k:
                        ss_b += n_k * (rv[in_k].mean() - grand) ** 2
                dest[j] = ss_b / ss_tot
        return {"tail_subsample_eta2_top": eta_top, "tail_subsample_eta2_bottom": eta_bot}

    def compute(self, ctx: EvalContext, params: Mapping[str, Any]) -> MetricResult:
        daily = apply_daily_matrix(self, ctx, params)
        return result_from_daily(daily, primary="tail_subsample_eta2_top", extra_scalars={"horizon": ctx.horizon})