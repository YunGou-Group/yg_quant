#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网页进程共享状态：任务表、评估服务、子进程队列。"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from FactorEvaluates.factor_eval_service import FactorEvalService

# 已完成任务保留时长与条数：只够前端轮询取回结果，不做历史归档
JOB_TTL_SECONDS = float(os.getenv("YG_QUANT_JOB_TTL_SECONDS", "3600"))
MAX_FINISHED_JOBS = int(os.getenv("YG_QUANT_JOB_KEEP_FINISHED", "20"))
# 同时排队 / 运行的任务上限。评估本身有全局锁，排太多只会堆内存
MAX_ACTIVE_JOBS = int(os.getenv("YG_QUANT_JOB_MAX_ACTIVE", "8"))
_ACTIVE_STATES = {"queued", "running"}


class JobQueueFull(RuntimeError):
    pass


@dataclass
class WebState:
    service: Optional[FactorEvalService] = None
    injected: bool = False
    worker_proc: Any = None
    req_q: Any = None
    resp_q: Any = None
    eval_lock: threading.Lock = field(default_factory=threading.Lock)
    backtest_lock: threading.Lock = field(default_factory=threading.Lock)
    jobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    jobs_guard: threading.Lock = field(default_factory=threading.Lock)

    def engine(self) -> FactorEvalService:
        if self.service is None:
            self.service = FactorEvalService()
        return self.service

    def set_job(self, job_id: str, **fields: Any) -> None:
        with self.jobs_guard:
            current = dict(self.jobs.get(job_id) or {})
            current.update(fields)
            current["updated_at"] = time.time()
            current.setdefault("created_at", current["updated_at"])
            self.jobs[job_id] = current
            self._evict(keep=job_id)

    def open_job(self, job_id: str, **fields: Any) -> None:
        """登记一个新任务，活跃任务超上限时拒绝，避免无界排队。"""
        with self.jobs_guard:
            self._evict()
            active = sum(
                1
                for item in self.jobs.values()
                if item.get("status") in _ACTIVE_STATES
            )
            if active >= MAX_ACTIVE_JOBS:
                raise JobQueueFull(
                    f"排队任务已达上限 {MAX_ACTIVE_JOBS}，请等当前任务结束"
                )
            now = time.time()
            self.jobs[job_id] = {
                "status": "queued",
                "error": None,
                "result_path": None,
                "created_at": now,
                "updated_at": now,
                **fields,
            }

    def _evict(self, keep: Optional[str] = None) -> None:
        """调用方必须已持有 jobs_guard。"""
        now = time.time()
        finished = []
        for key, item in self.jobs.items():
            if key == keep or item.get("status") in _ACTIVE_STATES:
                continue
            finished.append(key)
        for key in finished:
            age = now - float(self.jobs[key].get("updated_at") or 0.0)
            if age > JOB_TTL_SECONDS:
                self.jobs.pop(key, None)
        remaining = [key for key in finished if key in self.jobs]
        for key in remaining[: max(0, len(remaining) - MAX_FINISHED_JOBS)]:
            self.jobs.pop(key, None)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.jobs_guard:
            item = self.jobs.get(job_id)
            return dict(item) if item is not None else None
