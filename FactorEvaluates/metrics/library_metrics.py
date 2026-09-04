#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""库级指标：因子值相关、IC 序列相关、家族冗余。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from ..base_metric import BaseMetric
from ..context import LibraryEvalContext
from ..field_doc import FieldDoc
from ..metric_result import MetricResult
from ..param_spec import HORIZON_PARAM, ParamSpec
from ..batch.family_key import family_map


class FactorCorrMetric(BaseMetric):
    name = "factor_corr"
    dimension = "家族"
    description = "区间内日度截面 Spearman（因子×因子）的均值矩阵"
    eval_scope = "library"
    cost = "panel"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM,)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean_abs", "家族", "平均|相关|", "非对角元素绝对值的平均"),)

    def compute(self, ctx: Any, params: Mapping[str, Any]) -> MetricResult:
        raise ValueError("库级指标不能用于单因子评估")

    def compute_library(self, library_ctx: LibraryEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        corr = library_ctx.mean_factor_corr
        names = list(library_ctx.factor_names)
        if corr is None:
            return {"matrix": None, "scalars": {}}
        off = corr.copy()
        np.fill_diagonal(off, np.nan)
        return {
            "matrix": pd.DataFrame(corr, index=names, columns=names),
            "scalars": {
                "mean_abs": float(np.nanmean(np.abs(off))),
                "max_abs": float(np.nanmax(np.abs(off))) if np.isfinite(off).any() else float("nan"),
                "n_factors": len(names),
            },
        }


class ICCorrMetric(BaseMetric):
    name = "ic_corr"
    dimension = "家族"
    description = "各因子 daily_rank_ic 时间序列之间的相关"
    eval_scope = "library"
    cost = "derived"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM,)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean_abs", "家族", "IC相关|均值|", "RankIC 序列相关非对角绝对值平均"),)

    def compute(self, ctx: Any, params: Mapping[str, Any]) -> MetricResult:
        raise ValueError("库级指标不能用于单因子评估")

    def compute_library(self, library_ctx: LibraryEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        ic = library_ctx.daily_rank_ic
        names = list(library_ctx.factor_names)
        if ic is None or ic.shape[1] < 2:
            return {"matrix": None, "scalars": {}}
        with np.errstate(invalid="ignore"):
            corr = pd.DataFrame(ic, columns=names).corr(method="pearson").to_numpy(dtype=np.float64)
        off = corr.copy()
        np.fill_diagonal(off, np.nan)
        return {
            "matrix": pd.DataFrame(corr, index=names, columns=names),
            "scalars": {
                "mean_abs": float(np.nanmean(np.abs(off))),
                "max_abs": float(np.nanmax(np.abs(off))) if np.isfinite(off).any() else float("nan"),
            },
        }


class FamilyRedundancyMetric(BaseMetric):
    name = "family_redundancy"
    dimension = "家族"
    description = "家族内 |因子相关| 均值与最大配对相关"
    eval_scope = "library"
    cost = "derived"
    produces = ()
    requires = ()

    def params(self) -> Sequence[ParamSpec]:
        return (HORIZON_PARAM,)

    def fields(self) -> Sequence[FieldDoc]:
        return (FieldDoc("mean_abs", "家族", "家族内冗余", "所有家族内 |corr| 的平均"),)

    def compute(self, ctx: Any, params: Mapping[str, Any]) -> MetricResult:
        raise ValueError("库级指标不能用于单因子评估")

    def compute_library(self, library_ctx: LibraryEvalContext, params: Mapping[str, Any]) -> Dict[str, Any]:
        corr = library_ctx.mean_factor_corr
        names = list(library_ctx.factor_names)
        families = library_ctx.family_of or family_map(names)
        rows = []
        if corr is None:
            return {"families": [], "scalars": {}}
        by_fam: Dict[str, list] = {}
        for i, name in enumerate(names):
            by_fam.setdefault(families.get(name, "other"), []).append(i)
        all_abs = []
        for fam, idxs in sorted(by_fam.items()):
            if len(idxs) < 2:
                rows.append({"family": fam, "n": len(idxs), "mean_abs": float("nan"), "max_abs": float("nan")})
                continue
            sub = corr[np.ix_(idxs, idxs)].copy()
            np.fill_diagonal(sub, np.nan)
            mean_abs = float(np.nanmean(np.abs(sub)))
            max_abs = float(np.nanmax(np.abs(sub))) if np.isfinite(sub).any() else float("nan")
            all_abs.append(mean_abs)
            rows.append({"family": fam, "n": len(idxs), "mean_abs": mean_abs, "max_abs": max_abs})
        return {
            "families": rows,
            "scalars": {
                "mean_abs": float(np.nanmean(all_abs)) if all_abs else float("nan"),
                "n_families": len(by_fam),
            },
        }