#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Style 拥挤 HTTP API。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from StyleCrowding.query.bootstrap import bootstrap, fetch_series, load_manifest
from StyleCrowding.worker import start_crowding_job

from ..state import JobQueueFull, WebState


class CrowdingRunRequest(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    universe: Optional[str] = None
    incremental: bool = False
    skip: Optional[list[str]] = None


def register(app: FastAPI, state: WebState) -> None:
    @app.get("/api/style-crowding/bootstrap")
    def crowding_bootstrap() -> Dict[str, Any]:
        data = bootstrap()
        if not data.get("manifest"):
            raise HTTPException(status_code=404, detail="尚未发布 Style 拥挤结果，请先运行 StyleCrowding")
        return data

    @app.get("/api/style-crowding/manifest")
    def crowding_manifest() -> Dict[str, Any]:
        manifest = load_manifest()
        if not manifest:
            raise HTTPException(status_code=404, detail="manifest 不存在")
        return manifest

    @app.get("/api/style-crowding/series")
    def crowding_series(
        style: str,
        indicator: str,
        group_count: int = 10,
        orientation: str = "positive",
        transform: str = "raw",
    ) -> Dict[str, Any]:
        try:
            return fetch_series(
                style,
                indicator,
                group_count=group_count,
                orientation=orientation,
                transform=transform,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/style-crowding/run")
    def crowding_run(body: CrowdingRunRequest) -> Dict[str, Any]:
        payload = body.model_dump(exclude_none=True)
        try:
            job_id = start_crowding_job(
                state.jobs,
                state.jobs_guard,
                payload=payload,
                set_job=state.set_job,
                open_job=state.open_job,
            )
        except JobQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/style-crowding/jobs/{job_id}")
    def crowding_job(job_id: str) -> Dict[str, Any]:
        item = state.get_job(job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return item
