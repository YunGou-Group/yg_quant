#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict reader for the versioned stock daily market projection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from DailyUpdates.storage.bin_storage import (
    Column,
    _parse_layout,
    _serialize_lines,
)
from DailyUpdates.storage.market_projection import (
    PAYLOAD_DTYPE,
    ProjectionError,
    ProjectionPointer,
    load_manifest,
    load_pointer,
)
from DailyUpdates.storage.price_adjust import (
    ADJUSTABLE_FIELDS,
    ADJUST_MODES,
    adjusted_panel,
)


class ProjectionNotFoundError(ProjectionError):
    pass


class ProjectionStaleError(ProjectionError):
    pass


class ProjectionSchemaError(ProjectionError):
    pass


class MarketProjectionReader:
    def __init__(self, root_dir: str, *, sidecar_storage=None, strict: bool = True):
        self.root = Path(root_dir).expanduser().resolve()
        self.sidecar_storage = sidecar_storage
        self.strict = bool(strict)
        self.pointer: ProjectionPointer
        self.manifest: Dict
        self.epoch_dir: Path
        self.calendar: List[str]
        self.columns: List[Column]
        self.delist: Dict[int, str]
        self._load_snapshot()

    @property
    def generation_id(self) -> str:
        return f"{self.pointer.epoch}/{self.pointer.commit}"

    def _load_snapshot(self) -> None:
        try:
            self.pointer = load_pointer(self.root)
            self.manifest = load_manifest(self.root, self.pointer)
        except ProjectionError as exc:
            raise ProjectionNotFoundError(str(exc)) from exc
        self.epoch_dir = self.root / "epochs" / self.pointer.epoch
        axes = self.epoch_dir / "axes"
        n_days = int(self.manifest["n_days"])
        n_columns = int(self.manifest["n_columns"])
        n_delist = int(self.manifest["n_delist"])
        self.calendar = self._read_lines(axes / "calendar.txt", n_days)
        self.columns = []
        for line in self._read_lines(axes / "columns.idx", n_columns):
            try:
                slot, symbol, list_date = line.split("\t", 2)
                self.columns.append(Column(int(slot), symbol, list_date))
            except Exception as exc:
                raise ProjectionSchemaError(f"股票轴损坏: {line!r}") from exc
        self.delist = {}
        for line in self._read_lines(axes / "delist.log", n_delist):
            try:
                slot, day = line.split("\t", 1)
                self.delist[int(slot)] = day
            except Exception as exc:
                raise ProjectionSchemaError(f"退市轴损坏: {line!r}") from exc
        if len(self.calendar) != n_days or len(self.columns) != n_columns:
            raise ProjectionSchemaError("投影轴短于 manifest")
        self._list_dates = np.asarray(
            [column.list_date for column in self.columns],
            dtype="U10",
        )
        self._delist_dates = np.full(len(self.columns), "9999-12-31", dtype="U10")
        for slot, day in self.delist.items():
            if 0 <= slot < len(self.columns):
                self._delist_dates[slot] = day
        self._verify_axis_hashes()
        if self.strict:
            self.assert_current()

    @staticmethod
    def _read_lines(path: Path, count: int) -> List[str]:
        if not path.is_file():
            raise ProjectionSchemaError(f"投影轴文件不存在: {path}")
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) < count:
            raise ProjectionSchemaError(f"投影轴 {path} 只有 {len(lines)} 行，期望 {count}")
        return lines[:count]

    def _verify_axis_hashes(self) -> None:
        calendar_blob = _serialize_lines(self.calendar)
        columns_blob = _serialize_lines(
            [f"{col.slot}\t{col.symbol}\t{col.list_date}" for col in self.columns]
        )
        delist_blob = _serialize_lines(
            [f"{slot}\t{day}" for slot, day in sorted(self.delist.items())]
        )
        got = (
            hashlib.sha256(calendar_blob).hexdigest(),
            hashlib.sha256(columns_blob).hexdigest(),
            hashlib.sha256(delist_blob).hexdigest(),
        )
        want = (
            self.manifest["calendar_sha256"],
            self.manifest["columns_sha256"],
            self.manifest["delist_sha256"],
        )
        if got != want:
            raise ProjectionSchemaError("投影轴哈希与 manifest 不一致")

    def assert_current(self) -> None:
        if self.sidecar_storage is None:
            return
        source_db = Path(str(self.manifest.get("source_db") or "")).resolve()
        actual_db = Path(self.sidecar_storage.db_path).resolve()
        if source_db != actual_db:
            raise ProjectionStaleError(
                f"行情投影属于 {source_db}，当前 SQLite 是 {actual_db}"
            )
        latest = self.sidecar_storage.get_latest_stock_market_date()
        published = self.manifest.get("source_stock_max_date")
        if latest != published:
            raise ProjectionStaleError(
                f"股票行情投影落后: projection={published}, sqlite={latest}"
            )
        pending = self.sidecar_storage.list_projection_changes(
            after_seq=int(self.manifest.get("source_change_seq", 0))
        )
        if pending:
            raise ProjectionStaleError(
                f"股票行情投影有 {len(pending)} 条未发布变更，"
                f"首条 seq={pending[0]['seq']}"
            )

    def refresh(self) -> bool:
        pointer = load_pointer(self.root)
        if pointer == self.pointer:
            if self.strict:
                self.assert_current()
            return False
        self._load_snapshot()
        return True

    def available_fields(self) -> set[str]:
        return set(self.manifest.get("fields") or ())

    def latest_date(self) -> Optional[str]:
        return self.manifest.get("max_date")

    def _date_slice(
        self, start_date: Optional[str], end_date: Optional[str]
    ) -> Tuple[int, int]:
        if not self.calendar:
            return 0, -1
        start = (
            pd.Timestamp(start_date).strftime("%Y-%m-%d")
            if start_date
            else self.calendar[0]
        )
        end = (
            pd.Timestamp(end_date).strftime("%Y-%m-%d")
            if end_date
            else self.calendar[-1]
        )
        start_i = int(np.searchsorted(self.calendar, start, side="left"))
        end_i = int(np.searchsorted(self.calendar, end, side="right")) - 1
        if end_i < start_i or start_i >= len(self.calendar) or end_i < 0:
            return 0, -1
        return max(0, start_i), min(len(self.calendar) - 1, end_i)

    def _layout_for(self, field: str):
        if field not in self.available_fields():
            raise ProjectionSchemaError(f"行情投影没有字段 {field!r}")
        meta = self.manifest["field_meta"][field]
        path = self.epoch_dir / str(meta["layout"])
        if not path.is_file():
            raise ProjectionSchemaError(f"字段 {field} layout 不存在")
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != meta["layout_sha256"]:
            raise ProjectionSchemaError(f"字段 {field} layout 哈希错误")
        layout = _parse_layout(blob)
        if layout.n_days != len(self.calendar) or layout.n_columns != len(self.columns):
            raise ProjectionSchemaError(f"字段 {field} layout 轴长度错误")
        if (
            layout.calendar_hash.hex() != self.manifest["calendar_sha256"]
            or layout.columns_hash.hex() != self.manifest["columns_sha256"]
            or layout.delist_hash.hex() != self.manifest["delist_sha256"]
        ):
            raise ProjectionSchemaError(f"字段 {field} layout 轴哈希错误")
        return layout, meta

    def _read_raw_panel(
        self,
        field: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        layout, meta = self._layout_for(field)
        start_i, end_i = self._date_slice(start_date, end_date)
        wanted = {str(symbol) for symbol in symbols} if symbols else None
        all_symbols = [col.symbol for col in self.columns]
        col_keep = [
            i for i, symbol in enumerate(all_symbols)
            if wanted is None or symbol in wanted
        ]
        out_symbols = [all_symbols[i] for i in col_keep]
        if end_i < start_i:
            return pd.DataFrame(
                index=pd.Index([], name="trade_date"), columns=out_symbols, dtype=float
            )
        n = end_i - start_i + 1
        matrix = np.full((n, len(out_symbols)), np.nan, dtype=PAYLOAD_DTYPE)
        payload_path = self.epoch_dir / str(meta["payload"])
        expected_bytes = int(meta["payload_bytes"])
        if not payload_path.is_file() or payload_path.stat().st_size < expected_bytes:
            raise ProjectionSchemaError(f"字段 {field} payload 被截断")
        payload_start = int(layout.indptr[start_i])
        payload_stop = int(layout.indptr[end_i + 1])
        payload = np.fromfile(
            payload_path,
            dtype=PAYLOAD_DTYPE,
            count=payload_stop - payload_start,
            offset=payload_start * PAYLOAD_DTYPE.itemsize,
        )
        if payload.size != payload_stop - payload_start:
            raise ProjectionSchemaError(f"字段 {field} payload 读取不足")
        slot_to_out = np.full(len(self.columns), -1, dtype=np.int64)
        slot_to_out[np.asarray(col_keep, dtype=np.int64)] = np.arange(
            len(col_keep), dtype=np.int64
        )
        for day_i in range(start_i, end_i + 1):
            day = self.calendar[day_i]
            mask = (self._list_dates <= day) & (day < self._delist_dates)
            width = int(layout.indptr[day_i + 1] - layout.indptr[day_i])
            if width != int(mask.sum()):
                raise ProjectionSchemaError(
                    f"字段 {field} 在 {day} 行宽错误"
                )
            listed_slots = np.flatnonzero(mask)
            dest = slot_to_out[listed_slots]
            keep = dest >= 0
            if np.any(keep):
                offset = int(layout.indptr[day_i]) - payload_start
                matrix[day_i - start_i, dest[keep]] = payload[
                    offset : offset + width
                ][keep]
        return pd.DataFrame(
            matrix,
            index=pd.Index(
                self.calendar[start_i : end_i + 1], name="trade_date"
            ),
            columns=out_symbols,
        )

    def read_panels(
        self,
        fields: Sequence[str],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
        adjust: str = "none",
    ) -> Dict[str, pd.DataFrame]:
        self.refresh()
        mode = str(adjust or "none").lower()
        if mode not in ADJUST_MODES:
            raise ValueError(f"不支持的复权模式: {adjust!r}，可选 {ADJUST_MODES}")
        names = [str(field) for field in fields]
        missing = set(names) - self.available_fields()
        if missing:
            raise ProjectionSchemaError(f"行情投影缺少字段: {sorted(missing)}")
        raw = {
            field: self._read_raw_panel(
                field,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
            )
            for field in names
        }
        if mode == "hfq" and any(field in ADJUSTABLE_FIELDS for field in names):
            factor = raw.get("adj_factor")
            if factor is None:
                if "adj_factor" not in self.available_fields():
                    raise ProjectionSchemaError("后复权需要 adj_factor 投影")
                factor = self._read_raw_panel(
                    "adj_factor",
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols,
                )
            for field in names:
                raw[field] = adjusted_panel(raw[field], factor, field)
        return raw

    def read_panel(
        self,
        field: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
        adjust: str = "none",
    ) -> pd.DataFrame:
        return self.read_panels(
            [field],
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            adjust=adjust,
        )[str(field)]

    def read_market_data(
        self,
        fields: Sequence[str],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
        adjust: str = "none",
        ordered: bool = True,
    ) -> pd.DataFrame:
        panels = self.read_panels(
            fields,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            adjust=adjust,
        )
        if not panels:
            return pd.DataFrame(columns=["ts_code", "trade_date", *fields])
        sample = next(iter(panels.values()))
        dates = sample.index.to_numpy(dtype=str)
        names = sample.columns.to_numpy(dtype=str)
        frame = pd.DataFrame(
            {
                "ts_code": np.tile(names, len(dates)),
                "trade_date": np.repeat(dates, len(names)),
            }
        )
        for field, panel in panels.items():
            frame[field] = panel.to_numpy(dtype=np.float64, copy=False).reshape(-1)
        frame = frame.dropna(subset=list(fields), how="all")
        if ordered and not frame.empty:
            frame = frame.sort_values(["ts_code", "trade_date"])
        return frame.reset_index(drop=True)
