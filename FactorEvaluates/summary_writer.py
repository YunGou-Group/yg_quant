#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估摘要写入者：按因子 upsert JSON 标量快照，不存时序。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from yg_quant_repo import default_eval_dir


class SummaryWriter:
    def __init__(self, root_dir: Optional[str] = None):
        if root_dir:
            self.root = Path(root_dir).expanduser().resolve()
        else:
            self.root = default_eval_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def fingerprint(payload: Mapping[str, Any]) -> str:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def path_for(self, factor_name: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in factor_name)
        return self.root / f"{safe}.json"

    def upsert(
        self,
        factor_name: str,
        *,
        params: Mapping[str, Any],
        label: str,
        asof: str,
        start: Optional[str],
        end: Optional[str],
        scalars: Mapping[str, Any],
    ) -> Dict[str, Any]:
        key_payload = {
            "factor": factor_name,
            "params": dict(params),
            "label": label,
            "asof": asof,
            "start": start,
            "end": end,
        }
        snapshot = {
            "fingerprint": self.fingerprint(key_payload),
            "params": dict(params),
            "label": label,
            "asof": asof,
            "start": start,
            "end": end,
            "computed_at": datetime.now().isoformat(timespec="seconds"),
            "scalars": dict(scalars),
        }
        path = self.path_for(factor_name)
        document = {"factor": factor_name, "snapshots": []}
        if path.exists():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                document = {"factor": factor_name, "snapshots": []}
        snapshots: List[Dict[str, Any]] = list(document.get("snapshots") or [])
        snapshots = [
            item
            for item in snapshots
            if item.get("fingerprint") != snapshot["fingerprint"]
        ]
        snapshots.append(snapshot)
        document = {"factor": factor_name, "snapshots": snapshots}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
        return snapshot

    def list_all(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            factor = document.get("factor") or path.stem
            for item in document.get("snapshots") or []:
                row = dict(item)
                row["factor"] = factor
                rows.append(row)
        rows.sort(key=lambda item: str(item.get("computed_at") or ""), reverse=True)
        return rows

    def list_factor(self, factor_name: str) -> Dict[str, Any]:
        path = self.path_for(factor_name)
        if not path.exists():
            return {"factor": factor_name, "snapshots": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"factor": factor_name, "snapshots": []}
