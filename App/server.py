#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网页进程：托管 frontend，挂评估等 API。"""

from __future__ import annotations

import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from FactorEvaluates.eval_worker import start_eval_worker
from FactorEvaluates.factor_eval_service import FactorEvalService

from .log import logger, say
from .routers.backtest import register as register_backtest
from .routers.crowding import register as register_crowding
from .routers.eval import listen_worker, register as register_eval
from .state import WebState

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


def create_app(service: Optional[FactorEvalService] = None) -> FastAPI:
    state = WebState(service=service, injected=service is not None)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if not state.injected:
            say("启动评估子进程…")
            state.worker_proc, state.req_q, state.resp_q = start_eval_worker()
            threading.Thread(
                target=listen_worker,
                args=(state,),
                daemon=True,
                name="eval-job-listener",
            ).start()
        else:
            threading.Thread(target=_warmup, args=(state,), daemon=True).start()
        try:
            yield
        finally:
            if state.req_q is not None:
                try:
                    state.req_q.put(None)
                except Exception:
                    pass
            if state.worker_proc is not None and state.worker_proc.is_alive():
                state.worker_proc.join(timeout=3)
                if state.worker_proc.is_alive():
                    state.worker_proc.terminate()

    app = FastAPI(title="yg_quant", lifespan=lifespan)
    app.state.web = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_eval(app, state)
    register_backtest(app, state)
    register_crowding(app, state)
    _mount_frontend(app)
    return app


def _warmup(state: WebState) -> None:
    say("后台预加载 open 面板…")
    try:
        state.engine().warmup()
        say("open 面板预加载完成")
    except Exception:
        logger.exception("预加载 open 面板失败，评估请求时会再试")
        print("[app] open 面板预加载失败，见上方 traceback", flush=True)


def _mount_frontend(app: FastAPI) -> None:
    index_path = FRONTEND_DIST / "index.html"
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    def index():
        if index_path.is_file():
            return FileResponse(index_path)
        return HTMLResponse(
            "<p>前端尚未构建。开发请先启动 API，再在 frontend/ 执行 "
            "<code>npm install</code> 与 <code>npm run dev</code>，"
            "浏览器打开 <a href='http://127.0.0.1:5173'>http://127.0.0.1:5173</a>。"
            "或执行 <code>npm run build</code> 后刷新本页。</p>",
            status_code=503,
        )

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="not found")
        if index_path.is_file():
            return FileResponse(index_path)
        return index()


app = create_app()


def main() -> None:
    import multiprocessing

    multiprocessing.freeze_support()
    import uvicorn

    uvicorn.run(
        "App.server:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    if str(Path(__file__).resolve().parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
