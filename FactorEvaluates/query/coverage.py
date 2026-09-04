#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全库 run 覆盖：缺图/缺列原因，以及两次 run 对比。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..batch.batch_result_writer import read_table
from .datasets import read_meta, run_dir
from .factors import available_metrics

REASON_TEXT = {
    "ok": "有数",
    "not_in_run": "这次 run 没有跑对应指标",
    "no_file": "指标应产出该列，但 daily 目录没有文件",
    "no_factor": "文件在，但没有这个因子列",
    "all_nan": "有列但这个因子全是空值",
    "missing_input": "缺输入（例如没有风格暴露 X_T）",
    "not_applicable": "全库评估不产出这张图",
}

# 与 ChartGallery / 各 ChartPanel 标题对齐：any 命中即可渲染
CHART_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "market",
        "title": "市场（评估区间）",
        "keys": ("market_daily", "market_dd", "market_cum"),
        "metrics": (),
        "batch_note": "市场收益序列只在单因子交互评估里写，全库评估不产出",
    },
    {
        "id": "ic_series",
        "title": "IC 序列",
        "keys": (
            "rank_ic",
            "ic",
            "pure_ic",
            "rolling_ic",
            "weighted_ic",
            "nonlinear_ic",
            "mi_ic",
        ),
        "metrics": ("rank_ic", "ic", "pure_ic", "rolling_ic", "weighted_ic", "nonlinear_ic", "mi_ic"),
    },
    {
        "id": "ic_hist",
        "title": "IC 直方图",
        "keys": ("rank_ic", "ic"),
        "metrics": ("rank_ic", "ic"),
    },
    {
        "id": "ic_decay",
        "title": "IC 衰减",
        "prefixes": ("ic_decay_",),
        "metrics": ("ic_decay",),
    },
    {
        "id": "quantile_returns",
        "title": "分位净值",
        "keys": ("quantile_spread", "spread", "quantile_ic", "quantile_rank_ic", "quantile_return_hit"),
        "prefixes": ("quantile_returns_", "quantile_ic_", "quantile_rank_ic_", "quantile_top_hit_", "quantile_bot_hit_"),
        "metrics": ("quantile", "quantile_ic", "quantile_rank_ic", "quantile_return_hit"),
    },
    {
        "id": "turnover",
        "title": "分位换手",
        "prefixes": ("quantile_turnover_",),
        "metrics": ("quantile_turnover",),
    },
    {
        "id": "monotonicity",
        "title": "单调性（分组均收益）",
        "prefixes": ("quantile_returns_",),
        "metrics": ("quantile",),
    },
    {
        "id": "consistency",
        "title": "长短窗一致性",
        "keys": ("long_short_term_consistency", "consistency", "ma_fast", "ma_slow"),
        "metrics": ("long_short_term_consistency",),
    },
    {
        "id": "pnl",
        "title": "PnL / 多头",
        "keys": (
            "weighted_pnl",
            "weighted_long_pnl",
            "weighted_short_pnl",
            "long_only_return",
            "short_only_return",
            "factor_returns",
        ),
        "metrics": ("weighted_pnl", "long_only_return", "factor_returns"),
    },
    {
        "id": "size_strat",
        "title": "市值分层",
        "keys": ("size_ls", "size_long", "size_short", "size_rank_ls", "size_rank_long", "size_rank_short"),
        "prefixes": ("size_ls_", "size_rank_ls_"),
        "metrics": ("size_stratified_long_short", "size_rank_cut_long_short"),
    },
    {
        "id": "crowding",
        "title": "拥挤度",
        "prefixes": ("crowding_",),
        "metrics": ("factor_crowding",),
    },
    {
        "id": "tail",
        "title": "尾部",
        "prefixes": ("tail_",),
        "metrics": (
            "tail_hit_rate",
            "tail_weighted_ic",
            "tail_subsample_ic",
            "tail_pair_accuracy",
            "tail_quantile_separation",
            "tail_concentration",
            "tail_subsample_eta2",
        ),
    },
    {
        "id": "quality",
        "title": "质量",
        "keys": ("coverage_rate", "missing_rate", "unique_rate", "extreme_value_ratio", "iqr"),
        "metrics": ("coverage_rate",),
    },
    {
        "id": "factor_stats",
        "title": "截面分布",
        "keys": ("factor_mean", "factor_std", "factor_median", "factor_skewness", "factor_kurtosis"),
        "metrics": ("factor_stats",),
    },
    {
        "id": "factor_autocorr",
        "title": "因子自相关",
        "prefixes": ("factor_autocorr_",),
        "metrics": ("factor_autocorr",),
    },
    {
        "id": "style_corr",
        "title": "风格秩相关",
        "prefixes": ("factor_style_correlation_",),
        "metrics": ("factor_style_correlation",),
    },
    {
        "id": "barra_exposure",
        "title": "风格暴露 β",
        "prefixes": ("beta_", "ma_beta_"),
        "metrics": ("exposure",),
        "needs_xt": True,
    },
    {
        "id": "attribution",
        "title": "风格归因",
        "keys": ("cum_factor", "cum_explained", "cum_attr_industry", "cum_residual", "factor_return", "attr_industry"),
        "prefixes": (
            "f_style_",
            "f_nlsize",
            "ma_f_",
            "attr_style_",
            "attr_nlsize",
            "cum_attr_style_",
            "cum_attr_nlsize",
        ),
        "metrics": ("attribution",),
        "needs_xt": True,
    },
)


CHART_DIMENSIONS = {
    "market": "市场",
    "ic_series": "预测力",
    "ic_hist": "预测力",
    "ic_decay": "有效期",
    "quantile_returns": "预测力",
    "turnover": "有效期",
    "monotonicity": "预测力",
    "consistency": "预测力",
    "pnl": "预测力",
    "size_strat": "预测力",
    "crowding": "暴露度",
    "tail": "尾部",
    "quality": "暴露度",
    "factor_stats": "暴露度",
    "factor_autocorr": "有效期",
    "style_corr": "风格暴露",
    "barra_exposure": "风格暴露",
    "attribution": "风格归因",
}


def chart_catalog() -> List[Dict[str, Any]]:
    rows = []
    for spec in CHART_SPECS:
        rows.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "dimension": CHART_DIMENSIONS.get(spec["id"], "其他"),
                "keys": list(spec.get("keys") or ()),
                "prefixes": list(spec.get("prefixes") or ()),
                "metrics": list(spec.get("metrics") or ()),
            }
        )
    return rows


def run_metric_names(meta: Optional[Dict[str, Any]] = None) -> List[str]:
    payload = meta or {}
    names: List[str] = []
    raw = payload.get("metrics")
    if isinstance(raw, list):
        names.extend(str(x) for x in raw)
    elif isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, list):
                names.extend(str(x) for x in value)
    for key in ("daily_metrics", "crossday_metrics", "label_metrics", "library_metrics"):
        extra = payload.get(key)
        if isinstance(extra, list):
            names.extend(str(x) for x in extra)
    seen = set()
    out = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def factor_coverage(run_id: Optional[str] = None, factor: Optional[str] = None) -> Dict[str, Any]:
    meta = read_meta(run_id)
    rid = str(meta.get("run_id") or run_id or "")
    present = set(available_metrics(rid))
    run_metrics = set(run_metric_names(meta))
    has_xt = bool(meta.get("has_xt"))
    if "has_xt" not in meta:
        has_xt = any(name.startswith("beta_") or name in {"factor_return", "attr_industry"} for name in present)
    daily = run_dir(rid) / "daily"
    charts = []
    missing = 0
    for spec in CHART_SPECS:
        matched = _matching_keys(spec, present)
        statuses = {key: _column_status(daily, key, factor) for key in matched} if factor else {
            key: "ok" for key in matched
        }
        reason, detail = _diagnose(spec, matched, statuses, run_metrics, has_xt)
        row = {
            "id": spec["id"],
            "title": spec["title"],
            "dimension": CHART_DIMENSIONS.get(spec["id"], "其他"),
            "status": "ok" if reason == "ok" else "missing",
            "reason": reason,
            "reason_text": REASON_TEXT.get(reason, reason),
            "detail": detail,
            "metrics": list(spec.get("metrics") or ()),
            "matched_keys": matched,
        }
        if row["status"] != "ok":
            missing += 1
        charts.append(row)
    return {
        "run_id": rid,
        "factor": factor,
        "n_charts": len(charts),
        "n_missing": missing,
        "metrics": sorted(run_metrics),
        "daily_files": sorted(present),
        "has_xt": has_xt,
        "charts": charts,
    }


def compare_runs(left_id: str, right_id: str) -> Dict[str, Any]:
    left = _snapshot(left_id)
    right = _snapshot(right_id)
    fields = (
        "start",
        "end",
        "universe",
        "horizon",
        "n_factors",
        "n_dates",
        "batch_size",
        "n_workers",
        "elapsed_sec",
        "has_xt",
        "has_style",
    )
    meta_diff = []
    for field in fields:
        lv = left["meta"].get(field)
        rv = right["meta"].get(field)
        if lv != rv:
            meta_diff.append({"field": field, "left": lv, "right": rv})
    left_metrics = set(run_metric_names(left["meta"]))
    right_metrics = set(run_metric_names(right["meta"]))
    left_files = set(left["daily_files"])
    right_files = set(right["daily_files"])
    left_factors = set(left["factors"])
    right_factors = set(right["factors"])
    both_factors = sorted(left_factors & right_factors)
    rank_ic = _rank_ic_delta(left["summary"], right["summary"], both_factors)
    return {
        "left": _public_meta(left["meta"]),
        "right": _public_meta(right["meta"]),
        "meta_diff": meta_diff,
        "metrics": _set_diff(left_metrics, right_metrics),
        "daily_files": _set_diff(left_files, right_files),
        "factors": _set_diff(left_factors, right_factors, cap=40),
        "summary_columns": _set_diff(set(left["summary_columns"]), set(right["summary_columns"])),
        "rank_ic": rank_ic,
    }


def _snapshot(run_id: str) -> Dict[str, Any]:
    meta = read_meta(run_id)
    folder = run_dir(run_id)
    daily = folder / "daily"
    files = sorted({p.stem for p in daily.glob("*.parquet")})
    summary = pd.DataFrame()
    columns: List[str] = []
    factors: List[str] = []
    summary_path = folder / "summary" / "summary.parquet"
    if summary_path.is_file():
        try:
            summary = read_table(summary_path)
            columns = [str(c) for c in summary.columns]
            if "factor_name" in summary.columns:
                factors = [str(v) for v in summary["factor_name"].tolist()]
        except Exception:
            summary = pd.DataFrame()
    return {
        "meta": meta,
        "daily_files": files,
        "summary": summary,
        "summary_columns": columns,
        "factors": factors,
    }


def _public_meta(meta: Mapping[str, Any]) -> Dict[str, Any]:
    keep = (
        "run_id",
        "label",
        "display_label",
        "start",
        "end",
        "universe",
        "horizon",
        "n_factors",
        "n_dates",
        "batch_size",
        "n_workers",
        "elapsed_sec",
        "has_xt",
        "has_style",
        "metrics",
        "daily_metrics",
        "crossday_metrics",
        "label_metrics",
        "library_metrics",
    )
    return {key: meta.get(key) for key in keep if key in meta}


def _set_diff(left: set, right: set, cap: int = 80) -> Dict[str, Any]:
    only_left = sorted(left - right)
    only_right = sorted(right - left)
    return {
        "both": len(left & right),
        "only_left": only_left[:cap],
        "only_right": only_right[:cap],
        "only_left_n": len(only_left),
        "only_right_n": len(only_right),
    }


def _rank_ic_delta(
    left: pd.DataFrame, right: pd.DataFrame, both: Sequence[str]
) -> Dict[str, Any]:
    empty = {"n_both": len(both), "mean_delta": None, "top_improved": [], "top_worsened": []}
    if not both or left.empty or right.empty:
        return empty
    if "factor_name" not in left.columns or "factor_name" not in right.columns:
        return empty
    col = "rank_ic_mean" if "rank_ic_mean" in left.columns and "rank_ic_mean" in right.columns else None
    if col is None:
        return empty
    ls = left.set_index(left["factor_name"].astype(str))[col]
    rs = right.set_index(right["factor_name"].astype(str))[col]
    delta = pd.to_numeric(rs.reindex(both), errors="coerce") - pd.to_numeric(
        ls.reindex(both), errors="coerce"
    )
    delta = delta.replace([np.inf, -np.inf], np.nan).dropna()
    if delta.empty:
        return empty
    improved = delta.sort_values(ascending=False).head(8)
    worsened = delta.sort_values(ascending=True).head(8)
    return {
        "n_both": int(delta.size),
        "mean_delta": float(delta.mean()),
        "top_improved": [{"factor": str(i), "delta": float(v)} for i, v in improved.items()],
        "top_worsened": [{"factor": str(i), "delta": float(v)} for i, v in worsened.items()],
    }


def _matching_keys(spec: Mapping[str, Any], present: set) -> List[str]:
    keys = []
    for key in spec.get("keys") or ():
        if key in present:
            keys.append(key)
    for prefix in spec.get("prefixes") or ():
        keys.extend(sorted(name for name in present if name.startswith(prefix) and name not in keys))
    return keys


def _diagnose(
    spec: Mapping[str, Any],
    matched: Sequence[str],
    statuses: Mapping[str, str],
    run_metrics: set,
    has_xt: bool,
) -> Tuple[str, str]:
    if any(statuses.get(key) == "ok" for key in matched):
        hits = [key for key in matched if statuses.get(key) == "ok"]
        return "ok", f"有 {', '.join(hits[:6])}"
    note = spec.get("batch_note")
    if note:
        return "not_applicable", str(note)
    wanted = list(spec.get("metrics") or ())
    have_metric = (not wanted) or any(name in run_metrics for name in wanted)
    if wanted and not have_metric and not matched:
        return "not_in_run", "未跑 " + " / ".join(wanted)
    if spec.get("needs_xt") and not has_xt:
        return "missing_input", "这次 run 没有风格暴露 X_T，暴露/归因图需要它"
    if "all_nan" in statuses.values():
        return "all_nan", "对应列存在但这个因子全是空值"
    if "no_factor" in statuses.values():
        return "no_factor", "对应 parquet 里没有这个因子列"
    if have_metric and not matched:
        return "no_file", f"指标已列入 run，但 daily 没有「{spec['title']}」所需列"
    if not matched:
        return "no_file", "daily 目录没有对应 parquet"
    return "no_file", "没有可用序列"


def _column_status(daily: Path, key: str, factor: Optional[str]) -> str:
    if not factor:
        return "ok"
    path = daily / f"{key}.parquet"
    if not path.is_file():
        return "no_file"
    try:
        frame = pd.read_parquet(path, columns=[factor])
    except (ValueError, KeyError, OSError):
        return "no_factor"
    except Exception:
        return "no_file"
    values = pd.to_numeric(frame[factor], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).any():
        return "all_nan"
    return "ok"
