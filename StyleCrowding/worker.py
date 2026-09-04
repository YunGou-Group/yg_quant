#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StyleCrowding 后台任务（App HTTP 触发）。"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Callable, Dict, Optional

from .config import StyleCrowdingSettings, default_settings
from .runner import run_style_crowding


def start_crowding_job(
    jobs: Dict[str, Dict[str, Any]],
    jobs_guard: threading.Lock,
    *,
    payload: Optional[Dict[str, Any]] = None,
    set_job: Optional[Callable[..., None]] = None,
    open_job: Optional[Callable[..., None]] = None,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    if open_job:
        # 走 WebState 的排队上限与 TTL 清理
        open_job(job_id, kind="style_crowding")
    elif set_job:
        set_job(job_id, status="queued", kind="style_crowding")
    else:
        with jobs_guard:
            jobs[job_id] = {"status": "queued", "kind": "style_crowding"}

    def _run() -> None:
        def progress(info: Dict[str, Any]) -> None:
            if set_job:
                set_job(job_id, status="running", progress=info)
            else:
                with jobs_guard:
                    jobs[job_id]["status"] = "running"
                    jobs[job_id]["progress"] = info

        try:
            p = payload or {}
            incremental = bool(p.get("incremental"))
            start = p.get("start")
            if not start:
                start = "" if incremental else "2018-01-01"
            skip = p.get("skip")
            if skip is None and incremental:
                skip = ("pairwise",)
            else:
                skip = tuple(skip or ())
            settings = default_settings(
                start_date=start,
                end_date=p.get("end") or "",
                universe=p.get("universe") or "all",
                skip_indicators=skip,
            )
            result = run_style_crowding(
                settings,
                incremental=incremental,
                progress=progress,
            )
            if set_job:
                set_job(job_id, status="done", result=result)
            else:
                with jobs_guard:
                    jobs[job_id].update(status="done", result=result)
        except Exception as exc:
            if set_job:
                set_job(job_id, status="error", error=str(exc))
            else:
                with jobs_guard:
                    jobs[job_id].update(status="error", error=str(exc))

    threading.Thread(target=_run, daemon=True, name=f"crowding-{job_id}").start()
    return job_id
