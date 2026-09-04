#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""策略回测 HTTP：给网页可视化用。命令行仍走 python -m StrategyEngine。"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from StrategyEngine.attribution import attribute_for_web, load_attribution_view
from StrategyEngine.run import catalog_meta, list_snapshots, load_snapshot, run_backtest

from ..log import say
from ..state import JobQueueFull, WebState

logger = logging.getLogger("App")


class BacktestRequest(BaseModel):
    strategy: str
    start: Optional[str] = None
    end: Optional[str] = None
    universe: Optional[str] = "all"
    factor: Optional[str] = None
    n: Optional[int] = None
    hold: Optional[int] = None
    anti_tail: bool = False
    allocator: str = "equal"
    allocator_lookback: int = 252
    max_weight: float = 1.0
    risk_free_rate: float = 0.02
    risk_aversion: float = 1.0
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    l2_gamma: float = 0.1
    tc_rate: float = 0.001
    tail_confidence: float = 0.95
    mu_model: str = "geometric"
    cash: float = 1_000_000.0
    commission: float = 0.0003
    stamp: float = 0.0005
    benchmark: Optional[str] = None
    no_benchmark: bool = False
    save: bool = True


def register(app: FastAPI, state: WebState) -> None:
    @app.get("/api/backtest/meta")
    def meta() -> Dict[str, Any]:
        payload = catalog_meta()
        say(f"GET /api/backtest/meta 策略 {len(payload.get('strategies') or [])} 个")
        return payload

    @app.get("/api/backtest/runs")
    def runs(limit: int = 40) -> Dict[str, Any]:
        return {"items": list_snapshots(limit=limit)}

    @app.get("/api/backtest/runs/{run_id}")
    def run_one(run_id: str) -> Dict[str, Any]:
        try:
            return load_snapshot(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"未知回测 {run_id}") from exc

    @app.post("/api/backtest/run")
    def start_run(body: BacktestRequest) -> Dict[str, Any]:
        payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
        job_id = uuid.uuid4().hex[:12]
        try:
            state.open_job(
                job_id,
                kind="backtest",
                result=None,
                progress={"phase": "queued"},
                started=time.time(),
            )
        except JobQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        say(f"排队回测 job={job_id} strategy={body.strategy} {body.start} → {body.end}")
        threading.Thread(
            target=_run_job,
            args=(state, job_id, payload),
            daemon=True,
            name=f"backtest-{job_id}",
        ).start()
        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/backtest/runs/{run_id}/attribution")
    def get_attribution(run_id: str) -> Dict[str, Any]:
        try:
            return load_attribution_view(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc) or f"回测 {run_id} 还没有归因") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/backtest/runs/{run_id}/attribution")
    def start_attribution(run_id: str) -> Dict[str, Any]:
        try:
            load_snapshot(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"未知回测 {run_id}") from exc
        job_id = uuid.uuid4().hex[:12]
        try:
            state.open_job(
                job_id,
                kind="attribution",
                result=None,
                progress={"phase": "queued"},
                started=time.time(),
            )
        except JobQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        say(f"排队归因 job={job_id} run={run_id}")
        threading.Thread(
            target=_run_attribution_job,
            args=(state, job_id, run_id),
            daemon=True,
            name=f"attr-{job_id}",
        ).start()
        return {"job_id": job_id, "status": "queued"}


def _run_attribution_job(state: WebState, job_id: str, run_id: str) -> None:
    state.set_job(job_id, status="running", progress={"phase": "attributing"})
    started = time.perf_counter()
    try:
        with state.backtest_lock:
            result = attribute_for_web(run_id)
        elapsed = time.perf_counter() - started
        say(f"归因完成 job={job_id} run={run_id} 用时 {elapsed:.1f}s")
        state.set_job(
            job_id,
            status="done",
            result=result,
            error=None,
            progress={"phase": "done"},
        )
    except Exception as exc:
        logger.exception("归因失败 job=%s run=%s", job_id, run_id)
        state.set_job(job_id, status="error", result=None, error=str(exc))


def _run_job(state: WebState, job_id: str, payload: Dict[str, Any]) -> None:
    state.set_job(job_id, status="running", progress={"phase": "loading_panel"})
    started = time.perf_counter()
    try:
        def progress(phase: str) -> None:
            state.set_job(job_id, status="running", progress={"phase": phase})

        with state.backtest_lock:
            result = run_backtest(payload, save=bool(payload.get("save", True)), progress=progress)
        elapsed = time.perf_counter() - started
        say(f"回测完成 job={job_id} 用时 {elapsed:.1f}s")
        state.set_job(
            job_id,
            status="done",
            result=result,
            error=None,
            progress={"phase": "done"},
        )
    except Exception as exc:
        logger.exception("回测失败 job=%s", job_id)
        state.set_job(job_id, status="error", result=None, error=str(exc))
