#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite ``market_data`` 的股票日频 float64 面板投影。

SQLite 是唯一真源。投影按 epoch 保存只追加 payload；每次发布生成不可变
commit（layout + manifest），最后原子切换 ``CURRENT``。历史修正或轴变化
重建新 epoch，普通尾部增量复用当前 epoch。进程崩溃只暴露完整旧版本或
完整新版本；不额外做掉电级刷盘。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .bin_storage import (
    Column,
    FactorLayout,
    _FLOAT64,
    _INT64,
    _atomic_write_bytes,
    _atomic_write_text,
    _parse_layout,
    _serialize_lines,
    _sha256,
    listed_mask,
    write_array_all,
)

logger = logging.getLogger("MarketProjection")

MARKET_PROJECTION_FIELDS: Tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
    "adj_factor",
    "up_limit",
    "down_limit",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ttm",
    "total_share",
    "float_share",
    "total_mv",
    "circ_mv",
)

SCHEMA_VERSION = 1
PAYLOAD_DTYPE = np.dtype("<f8")


class ProjectionError(RuntimeError):
    """Base error for market projection publication."""


@dataclass(frozen=True)
class ProjectionPointer:
    epoch: str
    commit: str

    def to_dict(self) -> Dict[str, str]:
        return {"epoch": self.epoch, "commit": self.commit}


def _id(prefix: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:10]}"


def _json_bytes(value: Mapping) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8")


def _sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _read_lines(path: Path, limit: Optional[int] = None) -> List[str]:
    if not path.is_file():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return lines if limit is None else lines[: int(limit)]


def _pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
        ]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _column_lines(columns: Sequence[Column]) -> List[str]:
    return [f"{col.slot}\t{col.symbol}\t{col.list_date}" for col in columns]


def _delist_lines(delist: Mapping[int, str]) -> List[str]:
    return [f"{slot}\t{day}" for slot, day in sorted(delist.items())]


def _layout(
    calendar: Sequence[str],
    columns: Sequence[Column],
    delist: Mapping[int, str],
    indptr: np.ndarray,
) -> FactorLayout:
    cal_lines = list(calendar)
    col_lines = _column_lines(columns)
    del_lines = _delist_lines(delist)
    return FactorLayout(
        n_days=len(calendar),
        n_floats=int(indptr[-1]),
        n_columns=len(columns),
        n_delist=len(del_lines),
        calendar_hash=_sha256(_serialize_lines(cal_lines)),
        columns_hash=_sha256(_serialize_lines(col_lines)),
        delist_hash=_sha256(_serialize_lines(del_lines)),
        indptr=np.asarray(indptr, dtype=_INT64),
    )


def load_pointer(root: Path) -> ProjectionPointer:
    path = Path(root) / "CURRENT"
    if not path.is_file():
        raise ProjectionError(f"行情投影尚未发布: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ProjectionPointer(epoch=str(data["epoch"]), commit=str(data["commit"]))
    except Exception as exc:
        raise ProjectionError(f"行情投影 CURRENT 损坏: {path}") from exc


def load_manifest(root: Path, pointer: Optional[ProjectionPointer] = None) -> Dict:
    pointer = pointer or load_pointer(root)
    path = (
        Path(root)
        / "epochs"
        / pointer.epoch
        / "commits"
        / pointer.commit
        / "manifest.json"
    )
    if not path.is_file():
        raise ProjectionError(f"行情投影 manifest 不存在: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProjectionError(f"行情投影 manifest 损坏: {path}") from exc
    if int(data.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ProjectionError(
            f"不支持的行情投影 schema_version={data.get('schema_version')}"
        )
    if data.get("epoch") != pointer.epoch or data.get("commit") != pointer.commit:
        raise ProjectionError("CURRENT 与 manifest 的 epoch/commit 不一致")
    return data


class MarketProjectionWriter:
    def __init__(self, storage, root_dir: str):
        self.storage = storage
        self.root = Path(root_dir).expanduser().resolve()
        self.epochs_dir = self.root / "epochs"
        self.current_path = self.root / "CURRENT"
        self.lock_path = self.root / ".writer.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        self.epochs_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup_abandoned_temps(self) -> None:
        for path in self.epochs_dir.glob(".epoch-*.tmp"):
            shutil.rmtree(path, ignore_errors=True)
        for commits_dir in self.epochs_dir.glob("epoch-*/commits"):
            for path in commits_dir.glob(".commit-*.tmp"):
                shutil.rmtree(path, ignore_errors=True)

    @contextmanager
    def _writer_lock(self, timeout: float = 120.0) -> Iterator[None]:
        deadline = time.monotonic() + float(timeout)
        self.root.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(
                    str(self.lock_path),
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o666,
                )
                os.write(fd, f"{os.getpid()}\n".encode("ascii"))
                os.close(fd)
                break
            except FileExistsError:
                try:
                    owner = int(self.lock_path.read_text(encoding="ascii").strip())
                except (OSError, ValueError):
                    owner = -1
                if not _pid_alive(owner):
                    try:
                        self.lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise ProjectionError(f"行情投影写锁超时: {self.lock_path}")
                time.sleep(0.1)
        try:
            self._cleanup_abandoned_temps()
            yield
        finally:
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def _available_fields(self, fields: Optional[Sequence[str]]) -> List[str]:
        available = set(self.storage.get_market_fields())
        wanted = list(fields or MARKET_PROJECTION_FIELDS)
        invalid = [name for name in wanted if name not in MARKET_PROJECTION_FIELDS]
        if invalid:
            raise ProjectionError(f"不允许投影的行情字段: {invalid}")
        selected = [name for name in wanted if name in available]
        if "open" not in selected:
            raise ProjectionError("market_data 缺少投影主轴字段 open")
        return selected

    @staticmethod
    def _snapshot_seq(connection) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(seq), 0) AS seq FROM projection_changes"
        ).fetchone()
        return int(row["seq"] if row else 0)

    @staticmethod
    def _snapshot_changes(connection, after_seq: int) -> List[Dict]:
        rows = connection.execute(
            """
            SELECT seq, min_date, max_date, fields_json, axis_dirty
            FROM projection_changes
            WHERE seq > ?
            ORDER BY seq
            """,
            (int(after_seq),),
        ).fetchall()
        changes: List[Dict] = []
        for row in rows:
            try:
                fields = json.loads(row["fields_json"] or "[]")
            except (TypeError, ValueError):
                fields = []
            changes.append(
                {
                    "seq": int(row["seq"]),
                    "min_date": str(row["min_date"] or ""),
                    "max_date": str(row["max_date"] or ""),
                    "fields": [str(field) for field in fields],
                    "axis_dirty": bool(row["axis_dirty"]),
                }
            )
        return changes

    @staticmethod
    def _snapshot_stock_latest(connection) -> Optional[str]:
        row = connection.execute(
            """
            SELECT MAX(trade_date) AS latest
            FROM market_data
            WHERE symbol NOT LIKE 'index\\_%' ESCAPE '\\'
            """
        ).fetchone()
        return str(row["latest"]) if row and row["latest"] else None

    @staticmethod
    def _snapshot_market_fields(connection) -> set[str]:
        return {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(market_data)")
            if row["name"] not in {"symbol", "trade_date"}
        }

    @staticmethod
    def _stock_dates(connection, after: Optional[str] = None) -> List[str]:
        sql = (
            "SELECT DISTINCT m.trade_date "
            "FROM instruments AS i CROSS JOIN market_data AS m "
            "WHERE i.is_index = 0 AND m.symbol = i.symbol"
        )
        params: List[str] = []
        if after:
            sql += " AND m.trade_date > ?"
            params.append(str(after))
        sql += " ORDER BY m.trade_date"
        return [str(row["trade_date"]) for row in connection.execute(sql, params)]

    @staticmethod
    def _stock_axis(connection) -> Tuple[List[Column], Dict[int, str]]:
        rows = connection.execute(
            """
            SELECT i.symbol,
                   i.start_date AS data_start,
                   i.end_date AS data_end,
                   NULLIF(s.list_date, '') AS official_list_date,
                   NULLIF(s.delist_date, '') AS delist_date
            FROM instruments i
            LEFT JOIN stock_basic s ON s.symbol = i.symbol
            WHERE i.is_index = 0
              AND i.symbol NOT LIKE 'index\\_%' ESCAPE '\\'
            ORDER BY MIN(
                COALESCE(NULLIF(s.list_date, ''), i.start_date),
                i.start_date
            ), i.symbol
            """
        ).fetchall()
        columns: List[Column] = []
        delist: Dict[int, str] = {}
        for slot, row in enumerate(rows):
            data_start = pd.Timestamp(str(row["data_start"]))
            official_start = (
                pd.Timestamp(str(row["official_list_date"]))
                if row["official_list_date"]
                else data_start
            )
            list_date = min(data_start, official_start).strftime("%Y-%m-%d")
            columns.append(Column(slot=slot, symbol=str(row["symbol"]), list_date=list_date))
            if row["delist_date"]:
                official_end = pd.Timestamp(str(row["delist_date"]))
                data_end_exclusive = pd.Timestamp(str(row["data_end"])) + pd.Timedelta(days=1)
                delist[slot] = max(official_end, data_end_exclusive).strftime(
                    "%Y-%m-%d"
                )
        return columns, delist

    @staticmethod
    def _prefix_hashes(
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist: Mapping[int, str],
    ) -> Tuple[str, str, str]:
        return (
            _sha256_hex(_serialize_lines(list(calendar))),
            _sha256_hex(_serialize_lines(_column_lines(columns))),
            _sha256_hex(_serialize_lines(_delist_lines(delist))),
        )

    @staticmethod
    def _chunks(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
        for start in range(0, len(values), max(1, int(size))):
            yield values[start : start + max(1, int(size))]

    def _write_payloads(
        self,
        connection,
        epoch_dir: Path,
        fields: Sequence[str],
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist: Mapping[int, str],
        *,
        append_from: int = 0,
        base_indptr: Optional[np.ndarray] = None,
        chunk_days: int = 20,
    ) -> np.ndarray:
        fields_dir = epoch_dir / "fields"
        fields_dir.mkdir(parents=True, exist_ok=True)
        if base_indptr is None:
            indptr = [0]
        else:
            indptr = np.asarray(base_indptr, dtype=np.int64).tolist()
        mode = os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0)
        mode |= os.O_APPEND if append_from else os.O_TRUNC
        fds = {
            field: os.open(str(fields_dir / f"{field}.f64"), mode, 0o666)
            for field in fields
        }
        symbol_order = [col.symbol for col in columns]
        try:
            for days in self._chunks(calendar[append_from:], chunk_days):
                placeholders = ",".join("?" for _ in days)
                quoted = ", ".join(f'"{field}"' for field in fields)
                sql = (
                    f"SELECT symbol, trade_date, {quoted} FROM market_data "
                    f"WHERE trade_date IN ({placeholders}) "
                    "AND symbol NOT LIKE 'index\\_%' ESCAPE '\\' "
                    "ORDER BY trade_date, symbol"
                )
                frame = pd.read_sql_query(sql, connection, params=list(days))
                if not frame.empty:
                    frame["trade_date"] = pd.to_datetime(
                        frame["trade_date"]
                    ).dt.strftime("%Y-%m-%d")
                    frame["symbol"] = frame["symbol"].astype(str)
                grouped = {
                    day: part.set_index("symbol")
                    for day, part in frame.groupby("trade_date", sort=False)
                }
                buffers: Dict[str, List[np.ndarray]] = {field: [] for field in fields}
                for day in days:
                    mask = listed_mask(day, columns, dict(delist))
                    active = [symbol_order[i] for i in np.flatnonzero(mask)]
                    day_frame = grouped.get(day)
                    if day_frame is None:
                        aligned = pd.DataFrame(index=active, columns=fields, dtype=float)
                    else:
                        aligned = day_frame.reindex(active)
                    for field in fields:
                        buffers[field].append(
                            pd.to_numeric(aligned[field], errors="coerce").to_numpy(
                                dtype=PAYLOAD_DTYPE
                            )
                        )
                    indptr.append(indptr[-1] + len(active))
                for field, arrays in buffers.items():
                    if arrays:
                        write_array_all(fds[field], np.concatenate(arrays), PAYLOAD_DTYPE)
        finally:
            for fd in fds.values():
                os.close(fd)
        return np.asarray(indptr, dtype=_INT64)

    def _write_axes(
        self,
        epoch_dir: Path,
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist: Mapping[int, str],
    ) -> None:
        axes = epoch_dir / "axes"
        axes.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(axes / "calendar.txt", _serialize_lines(list(calendar)).decode())
        _atomic_write_text(
            axes / "columns.idx", _serialize_lines(_column_lines(columns)).decode()
        )
        _atomic_write_text(
            axes / "delist.log", _serialize_lines(_delist_lines(delist)).decode()
        )

    def _publish_commit(
        self,
        epoch_dir: Path,
        epoch_id: str,
        fields: Sequence[str],
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist: Mapping[int, str],
        indptr: np.ndarray,
        source_seq: int,
    ) -> Dict:
        commit_id = _id("commit")
        commit_tmp = epoch_dir / "commits" / f".{commit_id}.tmp"
        commit_dir = epoch_dir / "commits" / commit_id
        layouts_dir = commit_tmp / "layouts"
        layouts_dir.mkdir(parents=True, exist_ok=True)
        layout = _layout(calendar, columns, delist, indptr)
        layout_blob = layout.dumps()
        field_meta: Dict[str, Dict] = {}
        for field in fields:
            layout_path = layouts_dir / f"{field}.layout"
            layout_path.write_bytes(layout_blob)
            payload_path = epoch_dir / "fields" / f"{field}.f64"
            expected = int(layout.n_floats) * int(PAYLOAD_DTYPE.itemsize)
            if not payload_path.is_file() or payload_path.stat().st_size < expected:
                raise ProjectionError(f"字段 {field} payload 短于已发布 layout")
            field_meta[field] = {
                "dtype": PAYLOAD_DTYPE.str,
                "payload": f"fields/{field}.f64",
                "layout": f"commits/{commit_id}/layouts/{field}.layout",
                "n_floats": int(layout.n_floats),
                "payload_bytes": expected,
                "layout_sha256": _sha256_hex(layout_blob),
            }
        cal_hash, col_hash, del_hash = self._prefix_hashes(
            calendar, columns, delist
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "epoch": epoch_id,
            "commit": commit_id,
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "source_db": str(self.storage.db_path),
            "source_change_seq": int(source_seq),
            "source_stock_max_date": calendar[-1] if calendar else None,
            "min_date": calendar[0] if calendar else None,
            "max_date": calendar[-1] if calendar else None,
            "n_days": len(calendar),
            "n_columns": len(columns),
            "n_delist": len(delist),
            "n_floats": int(layout.n_floats),
            "dtype": PAYLOAD_DTYPE.str,
            "adjust": "none",
            "calendar_sha256": cal_hash,
            "columns_sha256": col_hash,
            "delist_sha256": del_hash,
            "fields": list(fields),
            "field_meta": field_meta,
        }
        (commit_tmp / "manifest.json").write_bytes(_json_bytes(manifest))
        commit_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(commit_tmp, commit_dir)
        _atomic_write_text(
            self.current_path,
            json.dumps(
                ProjectionPointer(epoch_id, commit_id).to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        return manifest
    def rebuild(
        self,
        fields: Optional[Sequence[str]] = None,
        *,
        chunk_days: int = 20,
        _locked: bool = False,
    ) -> Dict:
        selected = self._available_fields(fields)
        with (nullcontext() if _locked else self._writer_lock()):
            epoch_id = _id("epoch")
            epoch_tmp = self.epochs_dir / f".{epoch_id}.tmp"
            epoch_dir = self.epochs_dir / epoch_id
            if epoch_tmp.exists():
                shutil.rmtree(epoch_tmp)
            epoch_tmp.mkdir(parents=True)
            try:
                with self.storage._connect() as connection:
                    connection.execute("BEGIN")
                    source_seq = self._snapshot_seq(connection)
                    calendar = self._stock_dates(connection)
                    columns, delist = self._stock_axis(connection)
                    if not calendar:
                        raise ProjectionError("SQLite 没有可投影的股票行情")
                    if not columns:
                        raise ProjectionError("SQLite 没有可投影的股票轴")
                    self._write_axes(epoch_tmp, calendar, columns, delist)
                    indptr = self._write_payloads(
                        connection,
                        epoch_tmp,
                        selected,
                        calendar,
                        columns,
                        delist,
                        chunk_days=chunk_days,
                    )
                    connection.rollback()
                os.replace(epoch_tmp, epoch_dir)
                return self._publish_commit(
                    epoch_dir,
                    epoch_id,
                    selected,
                    calendar,
                    columns,
                    delist,
                    indptr,
                    source_seq,
                )
            except Exception:
                shutil.rmtree(epoch_tmp, ignore_errors=True)
                raise

    @staticmethod
    def _read_epoch_axes(
        epoch_dir: Path, manifest: Mapping
    ) -> Tuple[List[str], List[Column], Dict[int, str]]:
        axes = epoch_dir / "axes"
        calendar = _read_lines(axes / "calendar.txt", int(manifest["n_days"]))
        columns = []
        for line in _read_lines(axes / "columns.idx", int(manifest["n_columns"])):
            slot, symbol, list_date = line.split("\t", 2)
            columns.append(Column(int(slot), symbol, list_date))
        delist: Dict[int, str] = {}
        for line in _read_lines(axes / "delist.log", int(manifest["n_delist"])):
            slot, day = line.split("\t", 1)
            delist[int(slot)] = day
        return calendar, columns, delist

    def publish(self, *, chunk_days: int = 20) -> Dict:
        """Apply pending SQLite changes, rebuilding only when append is unsafe."""
        with self._writer_lock():
            try:
                pointer = load_pointer(self.root)
                manifest = load_manifest(self.root, pointer)
            except ProjectionError:
                return self.rebuild(chunk_days=chunk_days, _locked=True)
            epoch_dir = self.epochs_dir / pointer.epoch
            calendar, columns, delist = self._read_epoch_axes(epoch_dir, manifest)
            with self.storage._connect() as connection:
                connection.execute("BEGIN")
                source_seq = self._snapshot_seq(connection)
                pending = self._snapshot_changes(
                    connection,
                    int(manifest.get("source_change_seq", 0)),
                )
                stock_latest = self._snapshot_stock_latest(connection)
                if (
                    not pending
                    and stock_latest == manifest.get("source_stock_max_date")
                ):
                    connection.rollback()
                    return manifest
                old_max = str(manifest.get("max_date") or "")
                changed_fields = {
                    str(field)
                    for change in pending
                    for field in change["fields"]
                }
                projected_fields = set(manifest.get("fields") or ())
                schema_changed = bool(
                    (changed_fields & set(MARKET_PROJECTION_FIELDS))
                    - projected_fields
                ) or bool(
                    projected_fields - self._snapshot_market_fields(connection)
                )
                if (
                    not pending
                    or schema_changed
                    or any(change["axis_dirty"] for change in pending)
                    or any(change["min_date"] <= old_max for change in pending)
                ):
                    connection.rollback()
                    return self.rebuild(
                        chunk_days=chunk_days,
                        _locked=True,
                    )
                fresh_columns, fresh_delist = self._stock_axis(connection)
                if (
                    fresh_columns[: len(columns)] != columns
                    or {
                        slot: day
                        for slot, day in fresh_delist.items()
                        if slot < len(columns)
                    }
                    != delist
                ):
                    connection.rollback()
                    return self.rebuild(
                        chunk_days=chunk_days,
                        _locked=True,
                    )
                new_dates = self._stock_dates(connection, after=calendar[-1])
                if not new_dates:
                    connection.rollback()
                    return self.rebuild(
                        chunk_days=chunk_days,
                        _locked=True,
                    )
                new_calendar = [*calendar, *new_dates]
                fields = list(manifest["fields"])
                first_meta = manifest["field_meta"][fields[0]]
                old_layout_path = epoch_dir / str(first_meta["layout"])
                old_layout = _parse_layout(old_layout_path.read_bytes())
                for field in fields:
                    path = epoch_dir / "fields" / f"{field}.f64"
                    expected = int(old_layout.n_floats) * PAYLOAD_DTYPE.itemsize
                    if not path.is_file() or path.stat().st_size < expected:
                        raise ProjectionError(f"字段 {field} payload 短于当前 commit")
                    if path.stat().st_size > expected:
                        os.truncate(path, expected)
                self._write_axes(epoch_dir, new_calendar, fresh_columns, fresh_delist)
                indptr = self._write_payloads(
                    connection,
                    epoch_dir,
                    fields,
                    new_calendar,
                    fresh_columns,
                    fresh_delist,
                    append_from=len(calendar),
                    base_indptr=old_layout.indptr,
                    chunk_days=chunk_days,
                )
                connection.rollback()
            return self._publish_commit(
                epoch_dir,
                pointer.epoch,
                fields,
                new_calendar,
                fresh_columns,
                fresh_delist,
                indptr,
                source_seq,
            )

    def validate(self) -> Dict:
        pointer = load_pointer(self.root)
        manifest = load_manifest(self.root, pointer)
        epoch_dir = self.epochs_dir / pointer.epoch
        calendar, columns, delist = self._read_epoch_axes(epoch_dir, manifest)
        if len(calendar) != int(manifest["n_days"]):
            raise ProjectionError("投影日历长度与 manifest 不一致")
        if len(columns) != int(manifest["n_columns"]):
            raise ProjectionError("投影股票轴长度与 manifest 不一致")
        reference_indptr: Optional[np.ndarray] = None
        for field in manifest["fields"]:
            meta = manifest["field_meta"][field]
            layout_path = epoch_dir / str(meta["layout"])
            blob = layout_path.read_bytes()
            if _sha256_hex(blob) != meta["layout_sha256"]:
                raise ProjectionError(f"字段 {field} layout 校验失败")
            layout = _parse_layout(blob)
            payload_path = epoch_dir / str(meta["payload"])
            expected = int(layout.n_floats) * PAYLOAD_DTYPE.itemsize
            if not payload_path.is_file() or payload_path.stat().st_size < expected:
                raise ProjectionError(f"字段 {field} payload 被截断")
            if reference_indptr is None:
                reference_indptr = layout.indptr
                for i, day in enumerate(calendar):
                    width = int(layout.indptr[i + 1] - layout.indptr[i])
                    if width != int(listed_mask(day, columns, delist).sum()):
                        raise ProjectionError(f"字段 {field} 在 {day} 行宽错误")
            elif not np.array_equal(layout.indptr, reference_indptr):
                raise ProjectionError(f"字段 {field} layout 行指针不一致")
        return manifest


def main() -> None:
    import argparse

    from yg_quant_repo import default_db_path, default_market_projection_dir

    from .sqlite_storage import SQLiteStorage

    parser = argparse.ArgumentParser(description="构建或增量发布股票日频行情投影")
    parser.add_argument("--rebuild", action="store_true", help="强制全量重建新 epoch")
    parser.add_argument("--chunk-days", type=int, default=20)
    parser.add_argument("--db", default=str(default_db_path()))
    parser.add_argument("--out", default=str(default_market_projection_dir()))
    args = parser.parse_args()
    writer = MarketProjectionWriter(SQLiteStorage(args.db), args.out)
    result = (
        writer.rebuild(chunk_days=args.chunk_days)
        if args.rebuild
        else writer.publish(chunk_days=args.chunk_days)
    )
    writer.validate()
    print(
        f"行情投影已发布 epoch={result['epoch']} commit={result['commit']} "
        f"days={result['n_days']} fields={len(result['fields'])}"
    )


if __name__ == "__main__":
    main()
