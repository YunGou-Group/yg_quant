#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""列出全库评估 runs。"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from yg_quant_repo import default_eval_dir


def runs_root() -> Path:
    path = default_eval_dir() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_label(meta: Dict[str, Any]) -> str:
    custom = str(meta.get("label") or "").strip()
    if custom:
        return custom
    bits = []
    start, end = meta.get("start"), meta.get("end")
    if start and end:
        bits.append(f"{start}→{end}")
    if meta.get("universe"):
        bits.append(str(meta["universe"]))
    if meta.get("horizon") is not None:
        bits.append(f"h{meta['horizon']}")
    if meta.get("n_factors"):
        bits.append(f"{meta['n_factors']}因子")
    return " ".join(bits) or str(meta.get("run_id") or "")


def list_datasets() -> List[Dict[str, Any]]:
    rows = []
    for folder in sorted(runs_root().iterdir(), reverse=True):
        meta_path = folder / "meta" / "run.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta["run_id"] = meta.get("run_id") or folder.name
        meta["path"] = str(folder)
        meta["display_label"] = display_label(meta)
        rows.append(meta)
    return rows


def latest_run_id() -> Optional[str]:
    items = list_datasets()
    return items[0]["run_id"] if items else None


def run_dir(run_id: Optional[str] = None) -> Path:
    if run_id is None:
        rid = latest_run_id()
        if not rid:
            raise FileNotFoundError("还没有全库评估结果，请先运行全量评估")
    else:
        rid = str(run_id)
    _assert_run_id(rid)
    path = runs_root() / rid
    if not path.is_dir():
        raise FileNotFoundError(f"未知数据集 {rid}")
    return path


def read_meta(run_id: Optional[str] = None) -> Dict[str, Any]:
    path = run_dir(run_id) / "meta" / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["display_label"] = display_label(meta)
    return meta


def update_run(run_id: str, *, label: Optional[str] = None) -> Dict[str, Any]:
    folder = run_dir(run_id)
    path = folder / "meta" / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    if label is not None:
        text = str(label).strip()[:80]
        if text:
            meta["label"] = text
        else:
            meta.pop("label", None)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["run_id"] = meta.get("run_id") or run_id
    meta["display_label"] = display_label(meta)
    return meta


def delete_run(run_id: str) -> None:
    folder = run_dir(run_id).resolve()
    root = runs_root().resolve()
    if folder.parent != root:
        raise ValueError(f"拒绝删除 {run_id}")
    shutil.rmtree(folder)


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def valid_run_id(run_id: str) -> bool:
    text = str(run_id or "")
    return bool(RUN_ID_RE.fullmatch(text)) and text not in {".", ".."}


def _assert_run_id(run_id: str) -> None:
    if not valid_run_id(run_id):
        raise FileNotFoundError(f"未知数据集 {run_id}")


def resolve_output_dir(run_id: str, output_dir: Optional[str] = None) -> Path:
    """把 run 目录钉在 runs 根目录下，拒绝任何越界的 output_dir。"""
    _assert_run_id(run_id)
    root = runs_root().resolve()
    base = root if not output_dir else Path(output_dir).expanduser().resolve()
    target = (base / run_id).resolve()
    if target.parent != root:
        raise ValueError(f"输出目录必须落在 {root} 下，收到 {target}")
    return target
