#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立评估进程：占 GIL 的计算不堵 HTTP 轮询。"""

from __future__ import annotations

import json
import math
import multiprocessing
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .repo import ensure_repo_root
from yg_quant_repo import default_eval_dir

ensure_repo_root()


def job_dir() -> Path:
    root = default_eval_dir() / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def result_path(job_id: str) -> Path:
    return job_dir() / f"{job_id}.json"


def dump_result(job_id: str, payload: Dict[str, Any]) -> str:
    path = result_path(job_id)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    files = sorted(job_dir().glob("*.json"), key=lambda item: item.stat().st_mtime)
    for old in files[:-8]:
        try:
            old.unlink()
        except OSError:
            pass
    return str(path)


def load_result(job_id: str) -> Optional[Dict[str, Any]]:
    path = result_path(job_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except (ValueError, AttributeError):
            pass
    return str(value)


def worker_loop(req_q, resp_q) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from .factor_eval_service import FactorEvalService

    def say(message: str) -> None:
        print(f"[eval] {message}", flush=True)

    say("评估子进程启动，预加载 open 面板…")
    service = FactorEvalService()
    try:
        service.warmup()
        say("open 面板预加载完成")
        resp_q.put(("event", None, "ready", None, None))
    except Exception as exc:
        traceback.print_exc()
        resp_q.put(("event", None, "warmup_error", None, f"预加载失败: {exc}"))

    while True:
        item = req_q.get()
        if item is None:
            break
        job_id, payload = item
        resp_q.put(("job", job_id, "running", None, None))
        factor = (payload or {}).get("factor")
        mode = (payload or {}).get("mode")
        say(f"子进程开始 job={job_id} mode={mode or 'eval'} factor={factor!r}")
        try:
            if mode == "batch":
                from FactorEvaluates.batch.batch_eval_runner import BatchEvalRunner

                def progress(phase, detail, pct, message):
                    resp_q.put(
                        (
                            "progress",
                            job_id,
                            "running",
                            None,
                            json.dumps(
                                {
                                    "phase": phase,
                                    "detail": detail,
                                    "pct": pct,
                                    "message": message,
                                },
                                ensure_ascii=False,
                            ),
                        )
                    )

                result = BatchEvalRunner().run(payload or {}, progress=progress)
            else:
                result = service.evaluate(payload)
            path = dump_result(job_id, result)
            say(f"子进程完成 job={job_id} -> {path}")
            resp_q.put(("job", job_id, "done", path, None))
        except Exception as exc:
            traceback.print_exc()
            say(f"子进程失败 job={job_id}: {exc}")
            resp_q.put(("job", job_id, "error", None, str(exc)))


def start_eval_worker() -> Tuple[Any, Any, Any]:
    ctx = multiprocessing.get_context("spawn")
    req_q = ctx.Queue()
    resp_q = ctx.Queue()
    proc = ctx.Process(
        target=worker_loop,
        args=(req_q, resp_q),
        daemon=False,
        name="qmt-eval-worker",
    )
    proc.start()
    return proc, req_q, resp_q
