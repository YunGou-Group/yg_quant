#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全库评估编排。"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..factor_panel_loader import FactorPanelLoader
from yg_quant_repo import default_eval_dir
from ..metric_discoverer import MetricDiscoverer
from .batch_result_writer import BatchResultWriter
from .cross_day_engine import CrossDayEngine
from .daily_matrix_engine import DailyMatrixEngine
from .factor_batch_loader import FactorBatchLoader
from .library_metric_engine import LibraryMetricEngine
from .shared_panel_store import SharedPanelStore

ProgressFn = Callable[[str, str, float, str], None]


class BatchEvalRunner:
    def __init__(
        self,
        discoverer: Optional[MetricDiscoverer] = None,
        factor_loader: Optional[FactorPanelLoader] = None,
        batch_size: int = 32,
        n_workers: int = 0,
    ):
        self.discoverer = discoverer or MetricDiscoverer()
        self.factor_loader = factor_loader or FactorPanelLoader()
        self.batch_size = batch_size
        self.n_workers = n_workers

    def run(self, request: Mapping[str, Any], progress: Optional[ProgressFn] = None) -> Dict[str, Any]:
        say = progress or (lambda *_a, **_k: None)
        t0 = time.perf_counter()
        start = request.get("start") or "2010-01-01"
        end = request.get("end")
        universe = str(request.get("universe") or "all")
        horizon = int(request.get("horizon") or 5)
        params = dict(request.get("params") or {})
        params.setdefault("horizon", horizon)
        params.setdefault("universe", universe)
        params.setdefault("n_quantiles", int(request.get("n_quantiles") or 5))
        params.setdefault("min_obs", 20)
        names = list(request.get("factors") or self.factor_loader.list_factors())
        names = [n for n in names if n and not str(n).startswith("style_")]
        if not names:
            raise ValueError("没有可评估的因子")
        # 目录检查放在跑批之前：run_id 非法或已存在时，别让用户等几小时才报错
        run_id = str(request.get("run_id") or uuid.uuid4().hex[:12])
        self._resolve_out_dir(
            run_id,
            request.get("output_dir"),
            overwrite=bool(request.get("overwrite")),
        )
        batch_size = max(1, int(request.get("batch_size") or self.batch_size or 32))
        n_workers = int(request.get("n_workers") if request.get("n_workers") is not None else self.n_workers)
        planned = self.plan_metrics(self.discoverer, request.get("metrics"))
        daily_metrics = planned["daily"]
        label_metrics = planned["label"]
        crossday_metrics = planned["crossday"]
        library_metrics = planned["library"]
        need_xt = planned["need_xt"]
        need_style = planned["need_style"]
        store = SharedPanelStore(
            start=start,
            end=end,
            universe=universe,
            horizon=horizon,
            load_style=need_style,
            load_xt=need_xt,
            factor_loader=self.factor_loader,
            progress=say,
        )
        n_dates = len(store.dates)
        n_factors = len(names)
        name_index = {name: i for i, name in enumerate(names)}
        dummy = self._probe_keys(
            list(daily_metrics) + list(label_metrics),
            params,
            n_q=int(params["n_quantiles"]),
            x_names=list(store.x_names),
        )
        arrays: Dict[str, np.ndarray] = {
            key: np.full((n_dates, n_factors), np.nan, dtype=np.float32) for key in dummy
        }
        acc_corr = np.zeros((n_factors, n_factors), dtype=np.float64)
        acc_corr_n = np.zeros((n_factors, n_factors), dtype=np.float64)
        engine = DailyMatrixEngine(
            store,
            daily_metrics,
            params,
            label_metrics=label_metrics,
            need_daily_corr=planned["need_daily_corr"],
        )
        loader = FactorBatchLoader(store, self.factor_loader, batch_size=batch_size)
        prev_labels: Dict[str, np.ndarray] = {}
        n_batches = max(1, (len(names) + batch_size - 1) // batch_size)
        n_workers = self._fit_workers(store, batch_size, n_workers)
        say(
            "setup",
            "",
            0.24,
            f"{n_factors} 个因子, {n_dates} 日, 每批最多 {batch_size} 个, {n_workers} 进程; "
            f"预估本批立方 {self._cube_gb(n_dates, len(store.symbols), batch_size):.1f}GB",
        )
        try:
            for b, (chunk, cube) in enumerate(loader.iter_batches(names)):
                say("load_factors", chunk[0], 0.25 + 0.52 * b / n_batches, f"读取因子批次 {b + 1}/{n_batches}")
                engine.run(
                    chunk,
                    cube,
                    acc_corr,
                    acc_corr_n,
                    arrays,
                    name_index,
                    prev_labels,
                    progress=say,
                    n_workers=n_workers,
                    batch_i=b,
                    n_batches=n_batches,
                )
        finally:
            engine.close()
        if engine.metric_errors:
            out_dir = self._resolve_out_dir(
                run_id,
                request.get("output_dir"),
                overwrite=True,
            )
            fail_meta = {
                "run_id": run_id,
                "success": False,
                "metric_errors": engine.metric_errors,
                "n_metric_errors": len(engine.metric_errors),
            }
            BatchResultWriter(out_dir).write_meta(fail_meta)
            sample = engine.metric_errors[0]
            raise RuntimeError(
                f"已选指标计算失败 {len(engine.metric_errors)} 次，"
                f"例如 {sample.get('metric')} @ {sample.get('date')}: {sample.get('error')}"
            )
        if planned["need_daily_corr"] and n_batches > 1:
            self._accumulate_cross_batch_corr(
                loader,
                names,
                store,
                acc_corr,
                acc_corr_n,
                name_index,
                min_obs=int(params["min_obs"]),
                stride=max(1, int(request.get("corr_date_stride") or 1)),
                progress=say,
            )
        say("crossday", "", 0.78, "跨日派生")
        arrays = CrossDayEngine().run(arrays, params, crossday_metrics)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_corr = np.where(acc_corr_n > 0, acc_corr / acc_corr_n, np.nan)
        np.fill_diagonal(mean_corr, 1.0)
        say("library", "", 0.86, "库级指标")
        library = LibraryMetricEngine().run(
            library_metrics,
            names,
            store.dates,
            arrays.get("rank_ic", np.full((n_dates, n_factors), np.nan)),
            mean_corr,
            params,
        )
        out_dir = self._resolve_out_dir(
            run_id,
            request.get("output_dir"),
            overwrite=bool(request.get("overwrite")),
        )
        writer = BatchResultWriter(out_dir)
        say("write", "", 0.9, "写入 parquet")
        for key, arr in arrays.items():
            if arr.ndim == 2 and arr.shape[0] == n_dates:
                writer.write_daily(key, arr, store.dates, names)
        summary = writer.write_summary(arrays, names)
        writer.write_library(library)
        elapsed = time.perf_counter() - t0
        daily_names = [m.get_name() for m in daily_metrics]
        crossday_names = [m.get_name() for m in crossday_metrics]
        label_names = [m.get_name() for m in label_metrics]
        library_names = [m.get_name() for m in library_metrics]
        meta = {
            "run_id": run_id,
            "start": store.dates[0],
            "end": store.dates[-1],
            "universe": universe,
            "horizon": horizon,
            "n_factors": n_factors,
            "n_dates": n_dates,
            "batch_size": batch_size,
            "n_workers": n_workers,
            "dtype_daily": "float32",
            "precision_note": "日度输出 float32，与单因子 float64 允许约 1e-6 量级差",
            "metrics": sorted(set(daily_names + crossday_names + label_names + library_names)),
            "daily_metrics": daily_names,
            "crossday_metrics": crossday_names,
            "label_metrics": label_names,
            "library_metrics": library_names,
            "daily_keys": sorted(arrays.keys()),
            "has_xt": store.x_t is not None,
            "has_style": bool(store.style),
            "elapsed_sec": elapsed,
            "output_dir": str(out_dir),
        }
        label = str(request.get("label") or "").strip()[:80]
        if label:
            meta["label"] = label
        writer.write_meta(meta)
        say("done", run_id, 1.0, f"完成 {elapsed / 60:.1f} min")
        return {"run_id": run_id, "output_dir": str(out_dir), "meta": meta, "n_rows": int(len(summary))}

    @staticmethod
    def _accumulate_cross_batch_corr(
        loader: FactorBatchLoader,
        names: Sequence[str],
        store: SharedPanelStore,
        acc_corr: np.ndarray,
        acc_corr_n: np.ndarray,
        name_index: Mapping[str, int],
        min_obs: int,
        stride: int,
        progress: ProgressFn,
    ) -> None:
        """补齐跨批因子对的相关块。

        主循环只在每批内部算相关，非对角块会全是 NaN，导致 factor_corr /
        family_redundancy 只能看到同批因子。这里按批对 (i<j) 重载两批立方，
        用同样的口径累加非对角块。日期可用 corr_date_stride 抽样降成本。
        """
        from ..matrix_utils import spearman_cross_corr

        chunks = loader.chunks(list(names))
        if len(chunks) < 2:
            return
        n_pairs = len(chunks) * (len(chunks) - 1) // 2
        cols = [np.array([name_index[n] for n in chunk], dtype=int) for chunk in chunks]
        days = range(0, store.mask.shape[0], max(1, int(stride)))
        done = 0
        for i in range(len(chunks)):
            cube_i = loader.load_cube(chunks[i])
            for j in range(i + 1, len(chunks)):
                progress(
                    "cross_corr",
                    "",
                    0.78,
                    f"跨批因子相关 {done + 1}/{n_pairs}",
                )
                cube_j = loader.load_cube(chunks[j])
                block = np.zeros((len(chunks[i]), len(chunks[j])), dtype=np.float64)
                block_n = np.zeros_like(block)
                for t in days:
                    row = store.mask[t]
                    if int(row.sum()) < min_obs:
                        continue
                    corr = spearman_cross_corr(
                        cube_i[t, row, :].astype(np.float64, copy=False),
                        cube_j[t, row, :].astype(np.float64, copy=False),
                        min_obs=min_obs,
                    )
                    finite = np.isfinite(corr)
                    block = np.where(finite, block + corr, block)
                    block_n += finite
                del cube_j
                acc_corr[np.ix_(cols[i], cols[j])] += block
                acc_corr_n[np.ix_(cols[i], cols[j])] += block_n
                acc_corr[np.ix_(cols[j], cols[i])] += block.T
                acc_corr_n[np.ix_(cols[j], cols[i])] += block_n.T
                done += 1
            del cube_i

    @staticmethod
    def plan_metrics(discoverer: MetricDiscoverer, selected=None) -> Dict[str, Any]:
        wanted = None
        if selected:
            resolved = discoverer.resolve(
                [str(name) for name in selected], eval_scope="single"
            )
            wanted = {m.get_name() for m in resolved}
        single_metrics = discoverer.metrics_for("single")
        matrix_metrics = [
            m
            for m in single_metrics
            if m.has_matrix() and (wanted is None or m.get_name() in wanted)
        ]
        label_metrics = [
            m
            for m in single_metrics
            if m.has_label_history() and (wanted is None or m.get_name() in wanted)
        ]
        crossday_metrics = [
            m
            for m in single_metrics
            if m.has_crossday() and (wanted is None or m.get_name() in wanted)
        ]
        if matrix_metrics:
            ordered = discoverer.resolve(
                [m.get_name() for m in matrix_metrics], eval_scope="single"
            )
            daily_metrics = [m for m in ordered if m.has_matrix()]
        else:
            daily_metrics = []
        library_metrics = discoverer.metrics_for("library")
        if wanted is not None:
            library_metrics = [m for m in library_metrics if m.get_name() in wanted]
        need_daily_corr = any(
            m.get_name() in {"factor_corr", "family_redundancy"} for m in library_metrics
        )
        need_xt = any(m.get_name() in {"exposure", "pure_ic", "attribution"} for m in daily_metrics)
        need_style = need_xt or any(
            "style_raw" in getattr(m, "engine_requires", ()) for m in daily_metrics
        )
        return {
            "wanted": wanted,
            "daily": daily_metrics,
            "label": label_metrics,
            "crossday": crossday_metrics,
            "library": library_metrics,
            "need_xt": need_xt,
            "need_style": need_style,
            "need_daily_corr": need_daily_corr,
        }

    @staticmethod
    def _cube_gb(n_dates: int, n_stocks: int, n_factors: int) -> float:
        return n_dates * n_stocks * n_factors * 4 / (1024 ** 3)

    @staticmethod
    def _fit_workers(store: SharedPanelStore, batch_size: int, requested: int) -> int:
        import os

        cpu = os.cpu_count() or 4
        want = int(requested)
        if want <= 0:
            want = max(1, cpu - 1)
        want = max(1, min(want, cpu))
        n_dates = len(store.dates)
        n_stocks = len(store.symbols)
        panel = n_dates * n_stocks * 4
        panel *= 2 + len(store.fwd) + (len(store.style) if store.style else 0)
        if store.x_t is not None:
            panel += int(np.prod(store.x_t.shape) * 4)
        cube = n_dates * n_stocks * batch_size * 4
        labels = n_dates * n_stocks * batch_size
        used = panel + cube * 2 + labels + 2.5 * (1024 ** 3)
        while want > 1 and used + want * 0.18 * (1024 ** 3) > 24 * (1024 ** 3):
            want -= 1
        return want

    @staticmethod
    def _probe_keys(metrics: Sequence, params: dict, n_q: int, x_names: Optional[Sequence[str]] = None) -> set:
        from ..context import BatchEvalContext
        from ..metrics.extra_ic_metrics import DECAY_HORIZONS
        from ..metrics.statistics_metrics import AUTOCORR_LAGS

        keys = set()
        dummy = BatchEvalContext(
            factors=np.zeros((40, 1), dtype=np.float64),
            returns=np.zeros(40, dtype=np.float64),
            mask=np.ones(40, dtype=bool),
            n_quantiles=n_q,
        )
        dummy.intermediates["quantile_labels"] = np.ones((40, 1), dtype=np.int32)
        dummy.intermediates["fwd_decay"] = np.zeros((len(DECAY_HORIZONS), 40), dtype=np.float64)
        dummy.intermediates["decay_horizons"] = tuple(int(n) for n in DECAY_HORIZONS)
        names = [str(n) for n in (x_names or ())]
        if not names:
            names = ["intercept", "style_size", "style_beta", "ind_801010"]
        dummy.intermediates["x_t"] = np.zeros((40, len(names)), dtype=np.float64)
        dummy.intermediates["x_names"] = names
        dummy.intermediates["factor_ranks_full"] = np.zeros((40, 1), dtype=np.float64)
        dummy.intermediates["rank_ring"] = np.full((max(AUTOCORR_LAGS), 40, 1), np.nan, dtype=np.float64)
        dummy_labels = np.ones((8, 40, 1), dtype=np.int8)
        for metric in metrics:
            if metric.has_matrix():
                try:
                    keys.update(metric.compute_matrix(dummy, params).keys())
                except Exception:
                    keys.add(metric.get_name())
            if metric.has_label_history():
                try:
                    keys.update(metric.compute_from_labels(dummy_labels, params).keys())
                except Exception:
                    keys.add(metric.get_name())
        return keys

    @staticmethod
    def _resolve_out_dir(
        run_id: str, output_dir: Optional[str], overwrite: bool = False
    ) -> Path:
        """run 目录必须落在 runs 根目录下，且默认不覆盖已有结果。"""
        from ..query.datasets import resolve_output_dir

        out_dir = resolve_output_dir(run_id, output_dir)
        if out_dir.exists() and not overwrite:
            raise FileExistsError(
                f"数据集 {run_id} 已存在，换个 run_id 或显式传 overwrite=true"
            )
        return out_dir

    @staticmethod
    def _default_runs() -> Path:
        path = default_eval_dir() / "runs"
        path.mkdir(parents=True, exist_ok=True)
        return path
