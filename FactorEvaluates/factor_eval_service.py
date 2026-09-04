#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子评估服务：缓存 open / 当前因子，组股票池，跑指标并落盘标量。"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from .exposure_engine import RAW_STYLE_NAMES, ExposureEngine
from .industry_panel import load_l1_code_panel
from .factor_evaluator import FactorEvaluator
from .factor_panel_loader import FactorPanelLoader
from .market_panel_loader import HS300_SYMBOL, MarketPanelLoader
from .metric_discoverer import MetricDiscoverer
from Universes.catalog import build_mask as build_universe_mask

from .param_spec import HORIZON_PARAM, UNIVERSE_PARAM, ParamSpec, universe_param
from .return_calculator import ReturnCalculator
from .summary_writer import SummaryWriter

logger = logging.getLogger("FactorEvaluates")

DEFAULT_METRICS = (
    "rank_ic",
    "icir",
    "rolling_ic",
    "long_short_term_consistency",
    "quantile",
    "coverage_rate",
)


class FactorEvalService:
    def __init__(
        self,
        factor_loader: Optional[FactorPanelLoader] = None,
        market_loader: Optional[MarketPanelLoader] = None,
        discoverer: Optional[MetricDiscoverer] = None,
        writer: Optional[SummaryWriter] = None,
        evaluator: Optional[FactorEvaluator] = None,
    ):
        self.factor_loader = factor_loader or FactorPanelLoader()
        self.market_loader = market_loader or MarketPanelLoader()
        self.discoverer = discoverer or MetricDiscoverer()
        self.writer = writer or SummaryWriter()
        self.evaluator = evaluator or FactorEvaluator(discoverer=self.discoverer)
        self._open: Optional[pd.DataFrame] = None
        self._calculator: Optional[ReturnCalculator] = None
        self._factor_name: Optional[str] = None
        self._factor_panel: Optional[pd.DataFrame] = None
        self._open_lock = threading.Lock()

    def warmup(self) -> None:
        self._ensure_open()

    def meta(self) -> Dict[str, Any]:
        calendar = self.factor_loader.calendar()
        return {
            "factors": self.factor_loader.list_factors(),
            "calendar": {
                "start": calendar[0] if calendar else None,
                "end": calendar[-1] if calendar else None,
            },
            "contract": {
                "asof": ReturnCalculator.ASOF,
                "label_template": ReturnCalculator.LABEL_TEMPLATE,
            },
            "schema": self.discoverer.schema(eval_scope="single"),
            "defaults": {"metrics": list(DEFAULT_METRICS)},
        }

    def evaluate(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        factor_name = str(request.get("factor") or "").strip()
        if not factor_name:
            raise ValueError("缺少 factor")
        selected = request.get("metrics")
        if not selected:
            selected = list(DEFAULT_METRICS)
        selected = [str(name) for name in selected]
        self.discoverer.reject_library_metrics(selected)
        logger.info("开始评估 %s 指标=%s", factor_name, selected)
        print(f"[eval] 开始 {factor_name} 指标={selected}", flush=True)
        ordered = self.discoverer.resolve(selected, eval_scope="single")
        params = self._collect_params(request, ordered)

        start = self._date_or_none(request.get("start"))
        end = self._date_or_none(request.get("end"))
        t0 = time.perf_counter()
        print(f"[eval] 读取因子面板 {factor_name} …", flush=True)
        factor = self._slice_factor(self._load_factor(factor_name), start, end)
        print(
            f"[eval] 因子面板 {factor.shape[0]} 日 × {factor.shape[1]} 股 "
            f"({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
        t0 = time.perf_counter()
        print("[eval] 准备 open / 远期收益 …", flush=True)
        open_panel = self._ensure_open()
        if open_panel.empty:
            raise ValueError("open 面板为空，无法评估")
        print(
            f"[eval] open {open_panel.shape[0]} 日 × {open_panel.shape[1]} 股 "
            f"({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
        t0 = time.perf_counter()
        print(f"[eval] 股票池 mask universe={params.get('universe', 'all')} …", flush=True)
        mask = build_universe_mask(
            str(params.get("universe", "all")),
            factor.index,
            factor.columns,
            open_panel,
            self.market_loader.storage,
        )
        print(f"[eval] mask 完成 ({time.perf_counter() - t0:.1f}s)", flush=True)
        extras = {}
        needs_xt = any(
            "style_exposures" in getattr(metric, "engine_requires", ())
            for metric in ordered
        )
        needs_raw = any(
            "style_raw" in getattr(metric, "engine_requires", ())
            for metric in ordered
        )
        if needs_xt or needs_raw:
            t0 = time.perf_counter()
            print("[eval] 组装风格暴露 X_T（含 NLSIZE + 申万一级）…", flush=True)
            exposures, raw = self._build_exposures(mask, factor, build_xt=needs_xt)
            if needs_xt:
                extras["style_exposures"] = exposures
            if needs_raw:
                extras["style_raw"] = raw
            print(f"[eval] X_T 完成 ({time.perf_counter() - t0:.1f}s)", flush=True)
        if any(metric.get_name() == "ic_decay" for metric in ordered) and self._calculator is not None:
            from .metrics.extra_ic_metrics import DECAY_HORIZONS

            extras["fwd_ret_by_horizon"] = {}
            for n in DECAY_HORIZONS:
                fwd = self._calculator.get(int(n))
                extras["fwd_ret_by_horizon"][int(n)] = fwd.reindex(
                    index=factor.index, columns=factor.columns
                )
        payload = self.evaluator.evaluate(
            factor,
            open_panel,
            horizon=int(params["horizon"]),
            metric_names=selected,
            params=params,
            universe_mask=mask,
            calculator=self._calculator,
            start=start,
            end=end,
            factor_name=factor_name,
            universe=str(params.get("universe", "all")),
            intermediates=extras,
        )
        scalars = {
            name: item.get("scalars") or {}
            for name, item in payload["metrics"].items()
        }
        persist_params = {
            "horizon": params["horizon"],
            "universe": params.get("universe", "all"),
            "metrics": sorted(payload["metrics"]),
        }
        for metric in ordered:
            for spec in metric.params():
                if spec.scope == "metric" and spec.name in params:
                    persist_params[spec.name] = params[spec.name]
        snapshot = self.writer.upsert(
            factor_name,
            params=persist_params,
            label=payload["label"],
            asof=payload["asof"],
            start=start,
            end=end,
            scalars=scalars,
        )
        market = self._build_market_view(factor.index)
        print("[eval] 摘要已写入 JSON", flush=True)
        return {
            "factor": factor_name,
            "horizon": payload["horizon"],
            "label": payload["label"],
            "asof": payload["asof"],
            "start": start,
            "end": end,
            "universe": payload.get("universe"),
            "snapshot": snapshot,
            "market": market,
            "metrics": {
                name: {
                    "scalars": item["scalars"],
                    "series": {
                        key: self._series_payload(series)
                        for key, series in item["series"].items()
                    },
                }
                for name, item in payload["metrics"].items()
            },
        }

    def _build_market_view(self, dates: pd.Index) -> Optional[Dict[str, Any]]:
        symbol = HS300_SYMBOL
        try:
            close = self.market_loader.load_index_close(symbol)
        except Exception:
            logger.exception("读取大盘指数失败: %s", symbol)
            return None
        if close is None or close.empty:
            logger.warning("大盘指数为空: %s", symbol)
            return None
        aligned = close.copy()
        aligned.index = pd.to_datetime(aligned.index).strftime("%Y-%m-%d")
        window = pd.Index(pd.to_datetime(dates).strftime("%Y-%m-%d"))
        view = MarketPanelLoader.summarize_index(aligned.reindex(window), symbol)
        if view is None:
            return None
        return {
            "symbol": view["symbol"],
            "label": view["label"],
            "scalars": {
                key: FactorEvaluator._json_number(val)
                for key, val in view["scalars"].items()
            },
            "series": {
                key: self._series_payload(series)
                for key, series in view["series"].items()
            },
        }

    def list_results(self) -> List[Dict[str, Any]]:
        return self.writer.list_all()

    def get_results(self, factor_name: str) -> Dict[str, Any]:
        return self.writer.list_factor(factor_name)

    def _ensure_open(self) -> pd.DataFrame:
        if self._open is not None:
            return self._open
        with self._open_lock:
            if self._open is None:
                print("[eval] 正在从 SQLite 透视 open 宽表（可能较久）…", flush=True)
                self._open = self.market_loader.load_open()
                if self._open is not None and not self._open.empty:
                    self._calculator = ReturnCalculator(self._open)
                else:
                    self._calculator = None
        return self._open if self._open is not None else pd.DataFrame()

    def _load_factor(self, factor_name: str) -> pd.DataFrame:
        if self._factor_name != factor_name or self._factor_panel is None:
            self._factor_panel = None
            self._factor_name = factor_name
            panel = self.factor_loader.load(factor_name)
            if panel.empty:
                raise ValueError(f"因子 {factor_name} 没有数据")
            panel = panel.copy()
            panel.index = pd.to_datetime(panel.index).strftime("%Y-%m-%d")
            panel.columns = [str(col) for col in panel.columns]
            self._factor_panel = panel.sort_index()
        return self._factor_panel

    @staticmethod
    def _slice_factor(
        panel: pd.DataFrame,
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DataFrame:
        out = panel
        if start:
            out = out.loc[out.index >= start]
        if end:
            out = out.loc[out.index <= end]
        if out.empty:
            raise ValueError("日期切片后因子面板为空")
        return out

    def _collect_params(
        self,
        request: Mapping[str, Any],
        metrics: Sequence,
    ) -> Dict[str, Any]:
        incoming = dict(request.get("params") or {})
        if request.get("horizon") is not None:
            incoming["horizon"] = request["horizon"]
        if request.get("universe") is not None:
            incoming["universe"] = request["universe"]
        specs: Dict[str, ParamSpec] = {
            HORIZON_PARAM.name: HORIZON_PARAM,
            UNIVERSE_PARAM.name: universe_param(),
        }
        for metric in metrics:
            for spec in metric.params():
                specs[spec.name] = spec
        unknown = set(incoming) - set(specs)
        if unknown:
            raise ValueError(f"未知参数: {sorted(unknown)}")
        merged: Dict[str, Any] = {}
        for name, spec in specs.items():
            value = incoming[name] if name in incoming else spec.default
            merged[name] = spec.coerce(value)
        return merged

    def _build_exposures(
        self, mask: pd.DataFrame, factor: pd.DataFrame, build_xt: bool = True
    ):
        start = str(factor.index.min())
        end = str(factor.index.max())
        active = mask.any(axis=0)
        symbols = [str(col) for col, keep in active.items() if bool(keep)]
        print(
            f"[eval] 一次读取 {len(RAW_STYLE_NAMES)} 个风格 Bin "
            f"({len(symbols)} 只有效股票) …",
            flush=True,
        )
        started = time.perf_counter()
        loaded = self.factor_loader.load_many(
            RAW_STYLE_NAMES,
            start_date=start,
            end_date=end,
            symbols=symbols,
        )
        print(
            f"[eval] 风格 Bin 读取完成 ({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
        raw = {}
        for name, panel in loaded.items():
            if panel is None or panel.empty:
                logger.warning("风格 %s 为空，跳过", name)
                continue
            panel = panel.copy()
            panel.index = pd.to_datetime(panel.index).strftime("%Y-%m-%d")
            panel.columns = [str(col) for col in panel.columns]
            raw[name] = panel.reindex(index=mask.index, columns=mask.columns)
        if "style_size" not in raw and build_xt:
            raise ValueError("纯化/暴露需要 Bin 中的 style_size，请先跑 FactorUpdater")
        if not build_xt:
            return None, raw
        t_ind = time.perf_counter()
        codes, labels = load_l1_code_panel(
            self.market_loader.storage, mask.index, mask.columns
        )
        print(
            f"[eval] 行业面板 {codes.shape[0]} 日 × {codes.shape[1]} 股 "
            f"({time.perf_counter() - t_ind:.1f}s)",
            flush=True,
        )
        t_xt = time.perf_counter()
        exposures = ExposureEngine().build(
            raw, mask, industry_codes=codes, industry_labels=labels
        )
        print(
            f"[eval] X_T 列={len(exposures.names)} "
            f"行业哑变量={len(exposures.industry_names)} "
            f"参照={exposures.industry_labels.get(exposures.industry_ref, exposures.industry_ref)} "
            f"({time.perf_counter() - t_xt:.1f}s)",
            flush=True,
        )
        return exposures, raw

    @staticmethod
    def _date_or_none(value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    @staticmethod
    def _series_payload(series: pd.Series) -> Dict[str, List]:
        values = []
        for value in series.astype("float64"):
            if value != value:
                values.append(None)
            else:
                values.append(float(value))
        return {
            "dates": [str(idx) for idx in series.index],
            "values": values,
        }
