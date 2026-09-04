#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全库结果：daily parquet + summary + meta + 库级 JSON。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd


class BatchResultWriter:
    def __init__(self, run_dir: Path):
        self.root = Path(run_dir)
        self.daily = self.root / "daily"
        self.summary = self.root / "summary"
        self.meta = self.root / "meta"
        self.library = self.root / "library"
        for folder in (self.daily, self.summary, self.meta, self.library):
            folder.mkdir(parents=True, exist_ok=True)

    def write_daily(self, key: str, values: np.ndarray, dates: List[str], names: List[str]) -> None:
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[0] == 1 and len(dates) > 1:
            # scalar-per-factor (IR)：写成单行 summary 列，不占 daily
            return
        frame = pd.DataFrame(values, index=pd.Index(dates, name="trade_date"), columns=names)
        self._to_table(self.daily / f"{key}.parquet", frame)

    def write_summary(self, arrays: Mapping[str, np.ndarray], names: List[str]) -> pd.DataFrame:
        rows = []
        for i, name in enumerate(names):
            row: Dict[str, Any] = {"factor_name": name}
            for key, arr in arrays.items():
                if arr.ndim == 1:
                    if arr.size > i:
                        row[f"{key}"] = _num(arr[i])
                    continue
                if arr.shape[0] == 1:
                    row[f"{key}"] = _num(arr[0, i])
                    continue
                col = arr[:, i]
                finite = col[np.isfinite(col)]
                if finite.size == 0:
                    row[f"{key}_mean"] = None
                    row[f"{key}_std"] = None
                    row[f"{key}_ir"] = None
                    row[f"{key}_pct_positive"] = None
                    row[f"{key}_n"] = 0
                    continue
                mean = float(np.mean(finite))
                std = float(np.std(finite, ddof=1)) if finite.size > 1 else None
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = std
                row[f"{key}_ir"] = (mean / std) if std and std > 0 else None
                row[f"{key}_pct_positive"] = float(np.mean(finite > 0))
                row[f"{key}_n"] = int(finite.size)
            rows.append(row)
        frame = pd.DataFrame(rows)
        self._to_table(self.summary / "summary.parquet", frame)
        return frame

    def write_library(self, payload: Mapping[str, Any]) -> None:
        for name, item in payload.items():
            data = _json_ready(item)
            (self.library / f"{name}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def write_meta(self, payload: Mapping[str, Any]) -> None:
        body = dict(payload)
        body["computed_at"] = datetime.now().isoformat(timespec="seconds")
        (self.meta / "run.json").write_text(
            json.dumps(_json_ready(body), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _to_table(path: Path, frame: pd.DataFrame) -> None:
        try:
            frame.to_parquet(path, compression="zstd")
        except Exception:
            frame.to_parquet(path)


def read_table(path: Path) -> pd.DataFrame:
    parquet = path if path.suffix == ".parquet" else path.with_suffix(".parquet")
    if parquet.is_file():
        return pd.read_parquet(parquet)
    raise FileNotFoundError(path)


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, pd.DataFrame):
        return {
            "index": [str(i) for i in value.index],
            "columns": [str(c) for c in value.columns],
            "data": _json_ready(value.to_numpy().tolist()),
        }
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except (ValueError, AttributeError):
            pass
    return str(value)
