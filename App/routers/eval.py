#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估相关 HTTP：调用 FactorEvaluates，不实现评估本身。"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from FactorEvaluates.eval_worker import dump_result, load_result
from FactorEvaluates.factor_eval_service import DEFAULT_METRICS
from FactorEvaluates.query.datasets import valid_run_id

from ..log import say
from ..state import JobQueueFull, WebState

logger = logging.getLogger("App")


class EvalRequest(BaseModel):
    factor: str
    start: Optional[str] = None
    end: Optional[str] = None
    horizon: Optional[int] = None
    universe: Optional[str] = None
    metrics: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None


class BatchEvalRequest(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    horizon: Optional[int] = None
    universe: Optional[str] = None
    metrics: Optional[List[str]] = None
    factors: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None
    label: Optional[str] = None
    n_quantiles: Optional[int] = None
    batch_size: Optional[int] = None
    n_workers: Optional[int] = None
    overwrite: bool = False
    corr_date_stride: Optional[int] = None


def listen_worker(state: WebState) -> None:
    assert state.resp_q is not None
    while True:
        try:
            kind, job_id, status, path, error = state.resp_q.get()
        except (EOFError, OSError, KeyboardInterrupt):
            break
        if kind == "event":
            if status == "ready":
                say("评估子进程就绪")
            elif error:
                say(str(error))
            continue
        if kind == "progress":
            progress = {}
            if error:
                try:
                    progress = json.loads(error)
                except Exception:
                    progress = {"message": str(error)}
            state.set_job(job_id, status="running", progress=progress)
            continue
        if not job_id:
            continue
        fields: Dict[str, Any] = {"status": status, "error": error}
        if path:
            fields["result_path"] = path
        state.set_job(job_id, **fields)
        if status == "done":
            say(f"评估完成 job={job_id}")
        elif status == "error":
            say(f"评估失败 job={job_id}: {error}")


def run_local(state: WebState, job_id: str, payload: Dict[str, Any]) -> None:
    state.set_job(job_id, status="running")
    started = time.perf_counter()
    try:
        with state.eval_lock:
            if payload.get("mode") == "batch":
                from FactorEvaluates.batch.batch_eval_runner import BatchEvalRunner

                result = BatchEvalRunner().run(payload)
            else:
                result = state.engine().evaluate(payload)
        path = dump_result(job_id, result)
        elapsed = time.perf_counter() - started
        say(f"评估完成 job={job_id} 用时 {elapsed:.1f}s")
        state.set_job(job_id, status="done", result_path=path, error=None)
    except Exception as exc:
        logger.exception("评估失败 job=%s", job_id)
        state.set_job(job_id, status="error", result_path=None, error=str(exc))


def _enqueue(state: WebState, job_id: str, payload: Dict[str, Any], thread_name: str) -> None:
    alive = (
        state.req_q is not None
        and state.worker_proc is not None
        and state.worker_proc.is_alive()
    )
    if alive:
        state.req_q.put((job_id, payload))
        return
    say("评估子进程不可用，回退到本进程线程")
    threading.Thread(
        target=run_local,
        args=(state, job_id, payload),
        daemon=True,
        name=thread_name,
    ).start()


def register(app: FastAPI, state: WebState) -> None:
    @app.get("/api/meta")
    def meta() -> Dict[str, Any]:
        say("GET /api/meta")
        payload = state.engine().meta()
        say(f"meta 返回 {len(payload.get('factors') or [])} 个因子")
        return payload

    @app.post("/api/eval")
    def eval_factor(body: EvalRequest) -> Dict[str, Any]:
        payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
        job_id = uuid.uuid4().hex[:12]
        try:
            state.open_job(job_id, started=time.time())
        except JobQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        say(
            f"排队 job={job_id} factor={body.factor!r} "
            f"horizon={body.horizon} {body.start} → {body.end}"
        )
        _enqueue(state, job_id, payload, f"eval-{job_id}")
        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> Dict[str, Any]:
        item = state.get_job(job_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"未知任务 {job_id}")
        status = item.get("status")
        result = item.get("result")
        if result is None and status == "done":
            result = load_result(job_id)
        return {
            "job_id": job_id,
            "status": status,
            "error": item.get("error"),
            "progress": item.get("progress"),
            "result": result,
        }

    @app.get("/api/results")
    def results() -> Dict[str, Any]:
        return {"items": state.engine().list_results()}

    @app.post("/api/batch/eval")
    def batch_eval(body: BatchEvalRequest) -> Dict[str, Any]:
        payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
        payload["mode"] = "batch"
        run_id = str(body.run_id or "").strip()
        if run_id and not valid_run_id(run_id):
            raise HTTPException(
                status_code=400,
                detail="run_id 只允许字母数字和 . _ -，且不超过 64 个字符",
            )
        job_id = uuid.uuid4().hex[:12]
        try:
            state.open_job(job_id, started=time.time())
        except JobQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        say(f"排队全库 job={job_id} {body.start} → {body.end}")
        _enqueue(state, job_id, payload, f"batch-{job_id}")
        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/batch/jobs/{job_id}")
    def batch_job(job_id: str) -> Dict[str, Any]:
        return job_status(job_id)

    @app.get("/api/datasets")
    def datasets() -> Dict[str, Any]:
        from FactorEvaluates.query.datasets import list_datasets

        return {"items": list_datasets()}

    @app.get("/api/datasets/compare")
    def datasets_compare(left: str, right: str) -> Dict[str, Any]:
        from FactorEvaluates.query.coverage import compare_runs

        try:
            return compare_runs(left, right)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/datasets/{run_id}/coverage")
    def dataset_coverage(run_id: str, factor: Optional[str] = None) -> Dict[str, Any]:
        from FactorEvaluates.query.coverage import factor_coverage

        try:
            return factor_coverage(run_id, factor=factor)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.patch("/api/datasets/{run_id}")
    def dataset_patch(run_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        from FactorEvaluates.query.datasets import update_run

        try:
            return update_run(run_id, label=body.get("label"))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/datasets/{run_id}")
    def dataset_delete(run_id: str) -> Dict[str, Any]:
        from FactorEvaluates.query.datasets import delete_run

        try:
            delete_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "run_id": run_id}

    @app.get("/api/factors")
    def api_factors(run_id: Optional[str] = None, q: Optional[str] = None) -> Dict[str, Any]:
        from FactorEvaluates.query.factors import list_factors

        try:
            return list_factors(run_id, q=q)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/factors/{factor_name}")
    def api_factor(factor_name: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        from FactorEvaluates.query.factors import available_metrics, factor_summary

        try:
            return {
                "summary": factor_summary(factor_name, run_id),
                "metrics": available_metrics(run_id),
            }
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/factors/{factor_name}/series")
    def api_factor_series(
        factor_name: str, metric: str, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        from FactorEvaluates.query.factors import factor_series

        try:
            return factor_series(factor_name, metric, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/factors/{factor_name}/series-bundle")
    def api_factor_series_bundle(
        factor_name: str,
        run_id: Optional[str] = None,
        metrics: Optional[str] = None,
    ) -> Dict[str, Any]:
        from FactorEvaluates.query.factors import available_metrics, factor_series

        names = [item.strip() for item in str(metrics or "").split(",") if item.strip()]
        if not names:
            names = available_metrics(run_id)
        series = {}
        for name in names:
            try:
                series[name] = factor_series(factor_name, name, run_id)
            except FileNotFoundError:
                continue
        return {"factor": factor_name, "series": series}

    @app.get("/api/families")
    def api_families(run_id: Optional[str] = None) -> Dict[str, Any]:
        from FactorEvaluates.query.families import list_families

        try:
            return list_families(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/families-eval")
    def api_families_eval(
        run_id: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        from FactorEvaluates.query.family_eval import family_eval

        try:
            return family_eval(run_id, start=start, end=end)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/families-eval")
    def api_families_eval_post(body: Dict[str, Any]) -> Dict[str, Any]:
        from FactorEvaluates.query.family_eval import family_eval

        try:
            return family_eval(
                body.get("run_id"),
                start=body.get("start"),
                end=body.get("end"),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/select/preview")
    def api_select(body: Dict[str, Any]) -> Dict[str, Any]:
        from FactorEvaluates.query.select import preview

        try:
            return preview(body, run_id=body.get("run_id"))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/docs")
    def api_docs() -> Dict[str, Any]:
        from FactorEvaluates.query.docs import list_docs

        return {"items": list_docs(state.engine().discoverer)}

    @app.get("/api/docs/{slug}")
    def api_doc(slug: str) -> Dict[str, Any]:
        from FactorEvaluates.query.docs import get_doc

        try:
            return get_doc(slug, state.engine().discoverer)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/results/{factor_name}")
    def results_factor(factor_name: str) -> Dict[str, Any]:
        return state.engine().get_results(factor_name)

    app.state.defaults = list(DEFAULT_METRICS)
