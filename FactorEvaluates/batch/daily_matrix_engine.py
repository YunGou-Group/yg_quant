#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日度矩阵引擎：立方进 SharedMemory 后按日期切段多进程。"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..base_metric import BaseMetric
from ..exposure_engine import RAW_STYLE_NAMES
from ..metrics.extra_ic_metrics import DECAY_HORIZONS
from .shared_panel_store import SharedPanelStore
from .day_worker import evaluate_date_range, init_worker_panel, run_chunk_batch
from .shared_mem import ShmArray, ShmPack


class DailyMatrixEngine:
    def __init__(
        self,
        store: SharedPanelStore,
        metrics: Sequence[BaseMetric],
        params: Mapping,
        label_metrics: Optional[Sequence[BaseMetric]] = None,
        *,
        need_daily_corr: bool = True,
    ):
        self.store = store
        self.metrics = [m for m in metrics if m.has_matrix()]
        self.label_metrics = [m for m in (label_metrics or []) if m.has_label_history()]
        self.params = dict(params)
        self.need_daily_corr = bool(need_daily_corr)
        self.n_quantiles = int(self.params.get("n_quantiles", 5))
        self.min_obs = int(self.params.get("min_obs", 20))
        self._panel_pack: Optional[ShmPack] = None
        self._style = self._style_stack()
        self._fwd_decay, self._decay_h = self._decay_stack()
        self._pool = None
        self._pool_workers = 0
        self._panel_worker_spec: Optional[Dict[str, Any]] = None
        self.metric_errors: List[Dict[str, Any]] = []

    def run(
        self,
        names: List[str],
        cube: np.ndarray,
        acc_corr: np.ndarray,
        acc_corr_n: np.ndarray,
        arrays: Dict[str, np.ndarray],
        name_index: Mapping[str, int],
        prev_labels: Dict[str, np.ndarray],
        progress=None,
        n_workers: int = 1,
        batch_i: int = 0,
        n_batches: int = 1,
    ) -> None:
        n_dates, n_stocks, n_f = cube.shape
        key_list = [key for key, arr in arrays.items() if arr.ndim == 2 and arr.shape[0] == n_dates]
        key_index = {key: i for i, key in enumerate(key_list)}
        need_labels = bool(self.label_metrics)
        say = progress or (lambda *_a, **_k: None)
        workers = self._n_workers(n_workers, n_dates)
        cols = np.array([name_index[name] for name in names], dtype=np.int32)

        if workers <= 1:
            out = np.full((len(key_list), n_dates, n_f), np.nan, dtype=np.float32)
            labels = np.zeros((n_dates, n_stocks, n_f), dtype=np.int8) if need_labels else None
            corr_sum, corr_n, errors = evaluate_date_range(
                cube=cube,
                mask=self.store.mask,
                fwd=self.store.fwd[int(self.store.horizon)],
                fwd_decay=self._fwd_decay,
                decay_horizons=self._decay_h,
                style=self._style,
                x_t=self.store.x_t,
                x_names=self.store.x_names,
                out=out,
                labels=labels,
                key_index=key_index,
                metrics=self.metrics,
                params=self.params,
                t0=0,
                t1=n_dates,
                min_obs=self.min_obs,
                n_quantiles=self.n_quantiles,
                size_col=self._size_col(),
                need_daily_corr=self.need_daily_corr,
            )
            self._record_errors(errors)
            self._commit(
                out, labels, key_list, cols, arrays, acc_corr, acc_corr_n, corr_sum, corr_n, need_labels
            )
            return

        batch_pack = ShmPack()
        cube_shm = ShmArray.create(np.ascontiguousarray(cube))
        del cube
        batch_pack.add("cube", cube_shm)
        out_shm = ShmArray.create_empty((len(key_list), n_dates, n_f), np.float32)
        batch_pack.add("out", out_shm)
        has_labels = False
        if need_labels:
            batch_pack.add("labels", ShmArray.create_empty((n_dates, n_stocks, n_f), np.int8))
            has_labels = True

        panel = self._ensure_panel_pack()
        pool = self._ensure_pool(workers, panel)
        task_spec = self._task_spec(key_index, has_labels)
        chunk = max(80, int(np.ceil(n_dates / (workers * 4))))
        bounds = [(i, min(i + chunk, n_dates)) for i in range(0, n_dates, chunk)]
        batch_meta = batch_pack.meta()
        tasks = [(bound, batch_meta, task_spec) for bound in bounds]

        corr_sum = np.zeros((n_f, n_f), dtype=np.float64)
        corr_n = np.zeros((n_f, n_f), dtype=np.float64)
        try:
            done = 0
            for item in pool.imap_unordered(run_chunk_batch, tasks):
                corr_sum += item["corr_sum"]
                corr_n += item["corr_n"]
                self._record_errors(item.get("errors") or [])
                done += 1
                frac = (batch_i + done / max(len(bounds), 1)) / max(n_batches, 1)
                hi = item["t1"]
                date = self.store.dates[min(hi - 1, n_dates - 1)]
                say(
                    "daily",
                    date,
                    0.25 + 0.52 * frac,
                    f"日度 {date}  批次 {batch_i + 1}/{n_batches}  {done}/{len(bounds)}",
                )
            self._commit(
                out_shm.array,
                batch_pack.items["labels"].array if has_labels else None,
                key_list,
                cols,
                arrays,
                acc_corr,
                acc_corr_n,
                corr_sum,
                corr_n,
                need_labels,
            )
        finally:
            batch_pack.close_and_unlink()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
            self._pool_workers = 0
            self._panel_worker_spec = None
        if self._panel_pack is not None:
            # 日度阶段把 store.mask/fwd/... 改挂到 shm 视图；unlink 前必须拷回，
            # 否则跨批相关等后续步骤读 store 会踩已释放内存（Windows 常无 traceback、exit 1）。
            self._detach_store_from_shm()
            self._panel_pack.close_and_unlink()
            self._panel_pack = None

    def _detach_store_from_shm(self) -> None:
        """把仍需要的轴从 shm 视图拷回私有数组，并丢掉日度专用大面板。"""
        pack = self._panel_pack
        if pack is None:
            return
        mask_item = pack.items.get("mask")
        if mask_item is not None:
            self.store.mask = np.array(mask_item.array, copy=True, dtype=bool)
        # 跨批相关 / 写盘不再需要收益与风格面板
        self.store.fwd = {}
        self.store.style.clear()
        self.store.x_t = None
        self.store.x_names = []
        self._style = None
        self._fwd_decay = None
        self._decay_h = ()
        self.store.open = np.empty((0, 0), dtype=np.float32)

    def _ensure_pool(self, workers: int, panel: ShmPack):
        spec = self._panel_worker_spec
        if self._pool is not None and self._pool_workers == workers and spec is not None:
            return self._pool
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        panel_spec = {
            "panel": panel.meta(),
            "metric_names": [m.get_name() for m in self.metrics],
            "params": dict(self.params),
            "min_obs": self.min_obs,
            "n_quantiles": self.n_quantiles,
            "x_names": list(self.store.x_names),
            "decay_horizons": list(self._decay_h),
            "has_decay": "fwd_decay" in panel.items,
            "has_style": "style" in panel.items,
            "has_xt": "x_t" in panel.items,
            "size_col": self._size_col(),
        }
        self._panel_worker_spec = panel_spec
        self._pool = ctx.Pool(workers, initializer=init_worker_panel, initargs=(panel_spec,))
        self._pool_workers = workers
        return self._pool

    def _task_spec(self, key_index: Mapping[str, int], has_labels: bool) -> Dict[str, Any]:
        panel = self._panel_pack
        assert panel is not None
        return {
            "params": dict(self.params),
            "key_index": dict(key_index),
            "min_obs": self.min_obs,
            "n_quantiles": self.n_quantiles,
            "x_names": list(self.store.x_names),
            "decay_horizons": list(self._decay_h),
            "has_decay": "fwd_decay" in panel.items,
            "has_style": "style" in panel.items,
            "has_xt": "x_t" in panel.items,
            "has_labels": has_labels,
            "size_col": self._size_col(),
            "need_daily_corr": self.need_daily_corr,
        }

    def _record_errors(self, errors: Sequence[Mapping[str, Any]]) -> None:
        if not errors or len(self.metric_errors) >= 200:
            return
        dates = self.store.dates
        n = len(dates)
        for item in errors:
            if len(self.metric_errors) >= 200:
                break
            t = int(item.get("t", -1))
            date = dates[t] if 0 <= t < n else str(t)
            self.metric_errors.append(
                {
                    "metric": str(item.get("metric") or ""),
                    "date": date,
                    "error": str(item.get("error") or ""),
                }
            )

    def _ensure_panel_pack(self) -> ShmPack:
        if self._panel_pack is not None:
            return self._panel_pack
        pack = ShmPack()
        pack.add("mask", ShmArray.create(self.store.mask))
        self.store.mask = pack.items["mask"].array
        pack.add("fwd", ShmArray.create(self.store.fwd[int(self.store.horizon)]))
        # 主进程改挂 shm 视图，丢掉私有副本；worker 只读这份
        keep_fwd = {int(self.store.horizon): pack.items["fwd"].array}
        if self._fwd_decay is not None:
            pack.add("fwd_decay", ShmArray.create(self._fwd_decay))
            self._fwd_decay = pack.items["fwd_decay"].array
            for i, n in enumerate(self._decay_h):
                keep_fwd[int(n)] = self._fwd_decay[i]
        self.store.fwd = keep_fwd
        if self._style is not None:
            pack.add("style", ShmArray.create(self._style))
            self._style = pack.items["style"].array
            # 日度路径只用 style 栈；释放按名拆开的重复面板
            self.store.style.clear()
        if self.store.x_t is not None:
            pack.add("x_t", ShmArray.create(self.store.x_t))
            self.store.x_t = pack.items["x_t"].array
        # open 面板评估日度不再需要
        self.store.open = np.empty((0, 0), dtype=np.float32)
        self._panel_pack = pack
        return pack

    def _commit(
        self,
        out: np.ndarray,
        labels: Optional[np.ndarray],
        key_list: List[str],
        cols: np.ndarray,
        arrays: Dict[str, np.ndarray],
        acc_corr: np.ndarray,
        acc_corr_n: np.ndarray,
        corr_sum: np.ndarray,
        corr_n: np.ndarray,
        need_labels: bool,
    ) -> None:
        for i, key in enumerate(key_list):
            dest = arrays.get(key)
            if dest is None or dest.ndim != 2:
                continue
            dest[:, cols] = out[i]
        if need_labels and labels is not None:
            for metric in self.label_metrics:
                turned = metric.compute_from_labels(labels, self.params)
                for key, arr in turned.items():
                    dest = arrays.get(key)
                    if dest is not None:
                        dest[:, cols] = arr
        if corr_sum.size:
            acc_corr[np.ix_(cols, cols)] += corr_sum
            acc_corr_n[np.ix_(cols, cols)] += corr_n

    def _style_stack(self) -> Optional[np.ndarray]:
        if not self.store.style:
            return None
        n_dates, n_stocks = self.store.mask.shape
        stack = np.full((n_dates, n_stocks, len(RAW_STYLE_NAMES)), np.nan, dtype=np.float32)
        for i, name in enumerate(RAW_STYLE_NAMES):
            arr = self.store.style.get(name)
            if arr is not None:
                stack[:, :, i] = arr
        return stack

    def _decay_stack(self) -> Tuple[Optional[np.ndarray], Tuple[int, ...]]:
        if not any(m.get_name() == "ic_decay" for m in self.metrics):
            # 只要主 horizon，丢掉其余远期面板
            h = int(self.store.horizon)
            if h in self.store.fwd:
                self.store.fwd = {h: self.store.fwd[h]}
            return None, ()
        hs = tuple(int(n) for n in DECAY_HORIZONS if int(n) in self.store.fwd)
        if not hs:
            return None, ()
        stacked = np.stack([self.store.fwd[n] for n in hs], axis=0)
        return stacked, hs

    def _size_col(self) -> int:
        try:
            return list(RAW_STYLE_NAMES).index("style_size")
        except ValueError:
            return 0

    @staticmethod
    def _n_workers(requested: int, n_dates: int) -> int:
        cpu = os.cpu_count() or 4
        want = int(requested)
        if want <= 0:
            want = max(1, cpu - 1)
        want = max(1, min(want, cpu))
        if n_dates < 80:
            return 1
        return want
