#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Packed factor store: one file per factor, variable-width day rows.

Layout::

    {root}/
      calendars/day.txt
      instruments/columns.idx      # slot\\tsymbol\\tlist_date
      instruments/delist.log       # slot\\tdelist_date
      features/{factor}.f32        # little-endian float32, rows concatenated
      features/{factor}.layout     # published parse table (atomic replace)
      _factor_meta.json

A day's row holds still-listed stocks in ``columns.idx`` order:
``list_date <= day`` and (no delist or ``day < delist_date``). Suspended
names stay as NaN; delisted names drop out of later rows.

``.layout`` is little-endian::

    magic YGL1
    n_days, n_floats, n_columns, n_delist   uint64
    calendar/columns/delist SHA-256 of the published prefixes
    indptr int64[n_days+1]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import struct
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("BinStorage")

_FLOAT32 = np.dtype("<f4")
_INT64 = np.dtype("<i8")
_LAYOUT_MAGIC = b"YGL1"
_LAYOUT_HEADER = struct.Struct("<4s4Q32s32s32s")
def _axis_lock(root: Path) -> threading.RLock:
    key = str(root)
    with _AXIS_LOCKS_GUARD:
        lock = _AXIS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _AXIS_LOCKS[key] = lock
        return lock


_AXIS_LOCKS: Dict[str, threading.RLock] = {}
_AXIS_LOCKS_GUARD = threading.Lock()


def _win_sharing_violation(exc: BaseException) -> bool:
    winerr = getattr(exc, "winerror", None)
    return winerr in (5, 32)


def _replace_with_retry(tmp: Path, path: Path, attempts: int = 10) -> None:
    last: Optional[BaseException] = None
    for i in range(max(1, attempts)):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last = exc
        except OSError as exc:
            if not _win_sharing_violation(exc):
                raise
            last = exc
        time.sleep(0.05 * (2 ** min(i, 6)))
    try:
        tmp.unlink()
    except OSError:
        pass
    assert last is not None
    raise last


def _unlink_with_retry(path: Path, attempts: int = 8) -> None:
    last: Optional[BaseException] = None
    for i in range(max(1, attempts)):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            last = exc
        except OSError as exc:
            if not _win_sharing_violation(exc):
                raise
            last = exc
        time.sleep(0.05 * (2 ** min(i, 5)))
    if last is not None:
        raise last


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace_with_retry(tmp, path)


def _atomic_write_bytes(path: Path, blob: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(blob)
    _replace_with_retry(tmp, path)


def _serialize_lines(lines: Sequence[str]) -> bytes:
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _iso_date(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return pd.Timestamp(text).strftime("%Y-%m-%d")


def normalize_symbol(symbol: str) -> str:
    """Unify to DB style: ``000001.SZ`` / ``sz000001`` -> ``SZ000001``."""
    text = str(symbol).strip().upper()
    if "." in text:
        code, market = text.split(".", 1)
        return f"{market}{code}"
    return re.sub(r"[^0-9A-Z]", "", text) or "UNKNOWN"


@dataclass(frozen=True)
class Column:
    slot: int
    symbol: str
    list_date: str


@dataclass
class FactorLayout:
    n_days: int
    n_floats: int
    n_columns: int
    n_delist: int
    calendar_hash: bytes
    columns_hash: bytes
    delist_hash: bytes
    indptr: np.ndarray

    def dumps(self) -> bytes:
        header = _LAYOUT_HEADER.pack(
            _LAYOUT_MAGIC,
            int(self.n_days),
            int(self.n_floats),
            int(self.n_columns),
            int(self.n_delist),
            self.calendar_hash,
            self.columns_hash,
            self.delist_hash,
        )
        return header + np.asarray(self.indptr, dtype=_INT64).tobytes()


def _parse_layout(blob: bytes) -> FactorLayout:
    if len(blob) < _LAYOUT_HEADER.size:
        raise ValueError("layout 短于文件头")
    magic, n_days, n_floats, n_columns, n_delist, cal_h, col_h, del_h = (
        _LAYOUT_HEADER.unpack_from(blob, 0)
    )
    if magic != _LAYOUT_MAGIC:
        raise ValueError(f"layout magic 不是 YGL1: {magic!r}")
    n_days = int(n_days)
    need = _LAYOUT_HEADER.size + (n_days + 1) * 8
    if len(blob) < need:
        raise ValueError("layout indptr 被截断")
    indptr = np.frombuffer(
        blob, dtype=_INT64, count=n_days + 1, offset=_LAYOUT_HEADER.size
    ).copy()
    if int(indptr[0]) != 0:
        raise ValueError("indptr[0] 必须为 0")
    if np.any(np.diff(indptr) < 0):
        raise ValueError("indptr 不得下降")
    if int(indptr[-1]) != int(n_floats):
        raise ValueError("indptr[-1] 与 n_floats 不一致")
    return FactorLayout(
        n_days=n_days,
        n_floats=int(n_floats),
        n_columns=int(n_columns),
        n_delist=int(n_delist),
        calendar_hash=cal_h,
        columns_hash=col_h,
        delist_hash=del_h,
        indptr=indptr,
    )


def listed_mask(
    day: str,
    columns: Sequence[Column],
    delist_by_slot: Dict[int, str],
) -> np.ndarray:
    mask = np.zeros(len(columns), dtype=bool)
    for i, col in enumerate(columns):
        if col.list_date > day:
            continue
        ended = delist_by_slot.get(col.slot)
        if ended is not None and day >= ended:
            continue
        mask[i] = True
    return mask


class BinStorage:
    """Packed per-factor float32 store."""

    def __init__(self, root_dir: str):
        self.root = Path(root_dir).expanduser().resolve()
        self.calendars_dir = self.root / "calendars"
        self.instruments_dir = self.root / "instruments"
        self.features_dir = self.root / "features"
        self.calendars_dir.mkdir(parents=True, exist_ok=True)
        self.instruments_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.root / "_factor_meta.json"
        self._lock = _axis_lock(self.root)
        self._lock_path = self.root / "_axis.lock"
        self._meta = self._load_meta()
        self._discard_stale_parts()
        self._calendar_path = self.calendars_dir / "day.txt"
        self._columns_path = self.instruments_dir / "columns.idx"
        self._delist_path = self.instruments_dir / "delist.log"
        self._columns_cache: Optional[List[Column]] = None
        self._delist_cache: Optional[Dict[int, str]] = None
        self._delist_lines_cache: Optional[List[str]] = None

    def _safe_factor_name(self, factor_name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in factor_name)

    def _factor_paths(self, factor_name: str) -> Tuple[Path, Path]:
        stem = self._safe_factor_name(factor_name)
        return self.features_dir / f"{stem}.f32", self.features_dir / f"{stem}.layout"

    def _calendar_slice(
        self, calendar: Sequence[str], start_date: Optional[str], end_date: Optional[str]
    ) -> Tuple[int, int]:
        start_i = 0
        end_i = len(calendar) - 1
        if start_date:
            start = _iso_date(start_date)
            start_i = next((i for i, d in enumerate(calendar) if d >= start), len(calendar))
        if end_date:
            end = _iso_date(end_date)
            end_i = next(
                (i for i in range(len(calendar) - 1, -1, -1) if calendar[i] <= end),
                -1,
            )
        return start_i, end_i

    @contextmanager
    def _axis_guard(self, timeout: float = 60.0) -> Iterator[None]:
        with self._lock:
            fd = None
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    break
                except FileExistsError:
                    if time.monotonic() >= deadline:
                        logger.warning("轴锁等待超时，接管 %s", self._lock_path)
                        try:
                            os.unlink(self._lock_path)
                        except OSError:
                            pass
                        deadline = time.monotonic() + timeout
                    time.sleep(0.05)
            try:
                yield
            finally:
                os.close(fd)
                try:
                    os.unlink(self._lock_path)
                except OSError:
                    pass

    def _load_meta(self) -> dict:
        if not self._meta_path.exists():
            return {}
        try:
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _commit_meta(
        self, changed: Sequence[str] = (), removed: Sequence[str] = ()
    ) -> None:
        with self._axis_guard():
            disk = self._load_meta()
            for key in changed:
                if key in self._meta:
                    disk[key] = self._meta[key]
            for key in removed:
                disk.pop(key, None)
            self._meta = disk
            _atomic_write_text(
                self._meta_path,
                json.dumps(disk, ensure_ascii=False, indent=2),
            )

    def _discard_stale_parts(self) -> None:
        for path in self.root.glob("_factor_meta.json.*.tmp"):
            try:
                path.unlink()
            except OSError:
                logger.warning("无法删除残留 meta tmp: %s", path, exc_info=True)
        if not self.features_dir.is_dir():
            return
        for path in self.features_dir.glob("*.tmp"):
            try:
                path.unlink()
                logger.warning("丢弃残留临时文件: %s", path)
            except OSError:
                logger.warning("无法删除残留临时文件: %s", path, exc_info=True)

    def _set_factor_meta(
        self,
        factor_name: str,
        *,
        min_date: str,
        max_date: str,
        rows: int,
        n_symbols: int,
        price_adjust: Optional[str] = None,
    ) -> None:
        prev = self._meta.get(factor_name) or {}
        rec = {
            "rows": int(rows),
            "min_date": min_date,
            "max_date": max_date,
            "n_symbols": int(n_symbols),
            "layout": "packed_ygl1",
        }
        adj = price_adjust if price_adjust is not None else prev.get("price_adjust")
        if adj:
            rec["price_adjust"] = str(adj)
        self._meta[factor_name] = rec
        self._commit_meta(changed=[factor_name])

    def _read_text_lines(self, path: Path) -> List[str]:
        if not path.exists():
            return []
        return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def read_calendar(self) -> List[str]:
        return [_iso_date(line) for line in self._read_text_lines(self._calendar_path)]

    def write_calendar(self, dates: Sequence[str]) -> None:
        normalized = sorted({_iso_date(d) for d in dates if d})
        _atomic_write_text(
            self._calendar_path,
            "\n".join(normalized) + ("\n" if normalized else ""),
        )

    def read_columns(self) -> List[Column]:
        if self._columns_cache is not None:
            return list(self._columns_cache)
        columns: List[Column] = []
        for i, line in enumerate(self._read_text_lines(self._columns_path)):
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"columns.idx 第 {i} 行无法解析: {line!r}")
            slot = int(parts[0])
            if slot != i:
                raise ValueError(f"columns.idx 槽位 {slot} 与行号 {i} 不一致")
            columns.append(Column(slot=slot, symbol=normalize_symbol(parts[1]), list_date=_iso_date(parts[2])))
        self._columns_cache = columns
        return list(columns)

    def _write_columns(self, columns: Sequence[Column]) -> None:
        lines = [f"{col.slot}\t{col.symbol}\t{col.list_date}" for col in columns]
        _atomic_write_text(
            self._columns_path,
            "\n".join(lines) + ("\n" if lines else ""),
        )
        self._columns_cache = list(columns)

    def read_delist(self) -> Dict[int, str]:
        if self._delist_cache is not None:
            return dict(self._delist_cache)
        mapping: Dict[int, str] = {}
        lines: List[str] = []
        for line in self._read_text_lines(self._delist_path):
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"delist.log 无法解析: {line!r}")
            slot = int(parts[0])
            if slot in mapping:
                raise ValueError(f"delist.log 槽位 {slot} 重复")
            mapping[slot] = _iso_date(parts[1])
            lines.append(f"{slot}\t{mapping[slot]}")
        self._delist_cache = mapping
        self._delist_lines_cache = lines
        return dict(mapping)

    def _delist_lines(self) -> List[str]:
        self.read_delist()
        return list(self._delist_lines_cache or [])

    @staticmethod
    def _delist_prefix(delist_lines: Sequence[str], n_delist: int) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for line in delist_lines[: int(n_delist)]:
            slot_text, day = line.split("\t", 1)
            mapping[int(slot_text)] = day
        return mapping

    def _write_delist(self, mapping: Dict[int, str], lines: Sequence[str]) -> None:
        _atomic_write_text(
            self._delist_path,
            "\n".join(lines) + ("\n" if lines else ""),
        )
        self._delist_cache = dict(mapping)
        self._delist_lines_cache = list(lines)

    def read_instruments(self) -> Dict[str, Tuple[str, str]]:
        columns = self.read_columns()
        delist = self.read_delist()
        return {
            col.symbol: (col.list_date, delist.get(col.slot, ""))
            for col in columns
        }

    def write_instruments(self, mapping: Dict[str, Tuple[str, str]]) -> None:
        """Append unseen symbols only. Existing list dates stay put. End is ignored."""
        with self._axis_guard():
            self._columns_cache = None
            columns = self.read_columns()
            known = {col.symbol for col in columns}
            changed = False
            for raw, (start, _end) in mapping.items():
                symbol = normalize_symbol(raw)
                if symbol in known:
                    continue
                list_date = _iso_date(start)
                if not list_date:
                    raise ValueError(f"新列 {symbol} 缺少上市日")
                columns.append(Column(slot=len(columns), symbol=symbol, list_date=list_date))
                known.add(symbol)
                changed = True
            if changed:
                self._write_columns(columns)

    def _append_columns(self, additions: Sequence[Tuple[str, str]]) -> List[Column]:
        with self._axis_guard():
            self._columns_cache = None
            columns = self.read_columns()
            known = {col.symbol: col for col in columns}
            changed = False
            for raw, start in additions:
                symbol = normalize_symbol(raw)
                if symbol in known:
                    continue
                list_date = _iso_date(start)
                if not list_date:
                    raise ValueError(f"新列 {symbol} 缺少上市日")
                col = Column(slot=len(columns), symbol=symbol, list_date=list_date)
                columns.append(col)
                known[symbol] = col
                changed = True
            if changed:
                self._write_columns(columns)
            return columns

    def ensure_instruments(self, frame: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
        if frame is None or frame.empty:
            return self.read_instruments()
        additions: List[Tuple[str, str]] = []
        known = {col.symbol for col in self.read_columns()}
        work = frame.loc[:, ["ts_code", "trade_date"]].copy()
        work["ts_code"] = work["ts_code"].map(lambda x: normalize_symbol(str(x)))
        work["trade_date"] = work["trade_date"].map(_iso_date)
        for symbol, group in work.groupby("ts_code", sort=False):
            if symbol in known:
                continue
            additions.append((str(symbol), str(group["trade_date"].min())))
        if additions:
            self._append_columns(additions)
        return self.read_instruments()

    def register_delist(self, symbol: str, delist_date: str) -> None:
        """Append a delist event. Existing slots are never reused or moved."""
        code = normalize_symbol(symbol)
        day = _iso_date(delist_date)
        if not day:
            raise ValueError("退市日不能为空")
        with self._axis_guard():
            self._delist_cache = None
            self._delist_lines_cache = None
            columns = {col.symbol: col for col in self.read_columns()}
            col = columns.get(code)
            if col is None:
                raise ValueError(f"退市登记失败，列轴没有 {code}")
            mapping = self.read_delist()
            lines = self._delist_lines()
            if col.slot in mapping:
                if mapping[col.slot] != day:
                    raise ValueError(
                        f"槽位 {col.slot} 已有退市日 {mapping[col.slot]}，不能改成 {day}"
                    )
                return
            mapping[col.slot] = day
            lines.append(f"{col.slot}\t{day}")
            self._write_delist(mapping, lines)

    def ensure_calendar(self, dates: Sequence[str]) -> List[str]:
        with self._axis_guard():
            before = self.read_calendar()
            incoming = {_iso_date(d) for d in dates if d}
            merged = sorted(set(before) | incoming)
            if merged == before:
                return merged
            if merged[: len(before)] == before:
                self.write_calendar(merged)
                return merged
            self._remap_bins(before, merged)
            return merged

    def _axis_snapshot(self) -> Tuple[List[str], List[Column], Dict[int, str], List[str]]:
        return (
            self.read_calendar(),
            self.read_columns(),
            self.read_delist(),
            self._delist_lines(),
        )

    def _prefix_hashes(
        self,
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist_lines: Sequence[str],
        n_days: int,
        n_columns: int,
        n_delist: int,
    ) -> Tuple[bytes, bytes, bytes]:
        cal_lines = [f"{d}" for d in calendar[:n_days]]
        col_lines = [f"{c.slot}\t{c.symbol}\t{c.list_date}" for c in columns[:n_columns]]
        return (
            _sha256(_serialize_lines(cal_lines)),
            _sha256(_serialize_lines(col_lines)),
            _sha256(_serialize_lines(list(delist_lines[:n_delist]))),
        )

    def _verify_axis_hashes(
        self,
        layout: FactorLayout,
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist_lines: Sequence[str],
    ) -> None:
        if layout.n_days > len(calendar):
            raise ValueError("layout.n_days 长于日历")
        if layout.n_columns > len(columns):
            raise ValueError("layout.n_columns 长于列轴")
        if layout.n_delist > len(delist_lines):
            raise ValueError("layout.n_delist 长于退市日志")
        cal_h, col_h, del_h = self._prefix_hashes(
            calendar, columns, delist_lines,
            layout.n_days, layout.n_columns, layout.n_delist,
        )
        if cal_h != layout.calendar_hash:
            raise ValueError("日历前缀哈希不匹配")
        if col_h != layout.columns_hash:
            raise ValueError("列轴前缀哈希不匹配")
        if del_h != layout.delist_hash:
            raise ValueError("退市前缀哈希不匹配")

    def _verify_row_widths(
        self,
        layout: FactorLayout,
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist: Dict[int, str],
    ) -> None:
        used_cols = columns[: layout.n_columns]
        for day_i in range(layout.n_days):
            width = int(layout.indptr[day_i + 1] - layout.indptr[day_i])
            expected = int(listed_mask(calendar[day_i], used_cols, delist).sum())
            if width != expected:
                raise ValueError(
                    f"第 {day_i} 天 {calendar[day_i]} 行宽 {width} != 仍上市 {expected}"
                )

    def _read_layout(self, factor_name: str) -> Optional[FactorLayout]:
        _f32, layout_path = self._factor_paths(factor_name)
        if not layout_path.exists():
            return None
        return _parse_layout(layout_path.read_bytes())

    def _load_published(
        self, factor_name: str
    ) -> Tuple[FactorLayout, np.ndarray, List[str], List[Column], Dict[int, str], List[str]]:
        layout = self._read_layout(factor_name)
        if layout is None:
            raise ValueError(f"因子 {factor_name} 没有 layout")
        calendar, columns, _delist, delist_lines = self._axis_snapshot()
        published_delist = self._delist_prefix(delist_lines, layout.n_delist)
        self._verify_axis_hashes(layout, calendar, columns, delist_lines)
        self._verify_row_widths(
            layout, calendar, columns[: layout.n_columns], published_delist
        )
        f32_path, _ = self._factor_paths(factor_name)
        if not f32_path.exists():
            raise ValueError(f"因子 {factor_name} 缺少 .f32")
        n_file = os.path.getsize(f32_path) // int(_FLOAT32.itemsize)
        if n_file < layout.n_floats:
            raise ValueError(
                f"因子 {factor_name} 载荷 {n_file} 短于 layout.n_floats {layout.n_floats}"
            )
        payload = np.fromfile(f32_path, dtype=_FLOAT32, count=layout.n_floats)
        if payload.size < layout.n_floats:
            raise ValueError(f"因子 {factor_name} 读取短于 n_floats")
        return layout, payload, calendar, columns, published_delist, delist_lines

    def _pack_days(
        self,
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist: Dict[int, str],
        values: Dict[str, Dict[str, float]],
        start_i: int,
        end_i: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows: List[np.ndarray] = []
        widths: List[int] = []
        for day in calendar[start_i : end_i + 1]:
            mask = listed_mask(day, columns, delist)
            width = int(mask.sum())
            row = np.full(width, np.nan, dtype=_FLOAT32)
            day_vals = values.get(day) or {}
            cursor = 0
            for keep, col in zip(mask.tolist(), columns):
                if not keep:
                    continue
                if col.symbol in day_vals:
                    row[cursor] = np.float32(day_vals[col.symbol])
                cursor += 1
            if cursor != width:
                raise ValueError(f"{day} 组行失败: 写下 {cursor} 期望 {width}")
            rows.append(row)
            widths.append(width)
        if not rows:
            return np.zeros(0, dtype=_FLOAT32), np.zeros(0, dtype=np.int64)
        packed = np.concatenate(rows) if rows else np.zeros(0, dtype=_FLOAT32)
        return packed, np.asarray(widths, dtype=np.int64)

    def _publish_layout(
        self,
        factor_name: str,
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist_lines: Sequence[str],
        indptr: np.ndarray,
    ) -> FactorLayout:
        n_days = len(indptr) - 1
        layout = FactorLayout(
            n_days=n_days,
            n_floats=int(indptr[-1]),
            n_columns=len(columns),
            n_delist=len(delist_lines),
            calendar_hash=b"",
            columns_hash=b"",
            delist_hash=b"",
            indptr=np.asarray(indptr, dtype=_INT64),
        )
        cal_h, col_h, del_h = self._prefix_hashes(
            calendar, columns, delist_lines,
            layout.n_days, layout.n_columns, layout.n_delist,
        )
        layout.calendar_hash = cal_h
        layout.columns_hash = col_h
        layout.delist_hash = del_h
        _f32, layout_path = self._factor_paths(factor_name)
        _atomic_write_bytes(layout_path, layout.dumps())
        return layout

    def _write_payload(self, path: Path, payload: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o666)
        try:
            os.write(fd, np.asarray(payload, dtype=_FLOAT32).tobytes())
            os.fsync(fd)
        finally:
            os.close(fd)
        _replace_with_retry(tmp, path)

    def _append_payload(self, path: Path, payload: np.ndarray) -> None:
        if payload.size == 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
        fd = os.open(str(path), flags, 0o666)
        try:
            os.write(fd, np.asarray(payload, dtype=_FLOAT32).tobytes())
            os.fsync(fd)
        finally:
            os.close(fd)

    def _rewrite_factor(
        self,
        factor_name: str,
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist: Dict[int, str],
        delist_lines: Sequence[str],
        values: Dict[str, Dict[str, float]],
        existing: Optional[Tuple[FactorLayout, np.ndarray, Dict[int, str]]] = None,
    ) -> int:
        packed, widths = self._pack_days(
            calendar, columns, delist, values, 0, len(calendar) - 1
        )
        if existing is not None:
            old_layout, old_payload, old_delist = existing
            old_cols = columns[: old_layout.n_columns]
            for day_i, day in enumerate(calendar[: old_layout.n_days]):
                if day in values:
                    continue
                old_mask = listed_mask(day, old_cols, old_delist)
                start = int(old_layout.indptr[day_i])
                stop = int(old_layout.indptr[day_i + 1])
                old_row = old_payload[start:stop]
                new_mask = listed_mask(day, columns, delist)
                if int(old_mask.sum()) == int(new_mask.sum()) == old_row.size:
                    # same membership: keep bytes
                    dest = int(np.sum(widths[:day_i]))
                    packed[dest : dest + old_row.size] = old_row
                    continue
                rebuilt = np.full(int(new_mask.sum()), np.nan, dtype=_FLOAT32)
                old_syms = [c.symbol for c, keep in zip(old_cols, old_mask.tolist()) if keep]
                new_syms = [c.symbol for c, keep in zip(columns, new_mask.tolist()) if keep]
                lookup = {sym: old_row[j] for j, sym in enumerate(old_syms) if j < old_row.size}
                for j, sym in enumerate(new_syms):
                    if sym in lookup:
                        rebuilt[j] = lookup[sym]
                dest = int(np.sum(widths[:day_i]))
                packed[dest : dest + rebuilt.size] = rebuilt
        indptr = np.zeros(len(calendar) + 1, dtype=_INT64)
        if len(widths):
            np.cumsum(widths, out=indptr[1:])
        f32_path, layout_path = self._factor_paths(factor_name)
        self._write_payload(f32_path, packed)
        self._publish_layout(factor_name, calendar, columns, delist_lines, indptr)
        del layout_path
        return int(packed.size)

    def _append_factor_days(
        self,
        factor_name: str,
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist: Dict[int, str],
        delist_lines: Sequence[str],
        values: Dict[str, Dict[str, float]],
        layout: FactorLayout,
        from_day: int,
    ) -> int:
        if from_day < layout.n_days:
            raise ValueError("追加起点落在已发布日期内")
        if from_day > layout.n_days:
            raise ValueError("追加起点越过已发布日期，中间有空洞")
        packed, widths = self._pack_days(
            calendar, columns, delist, values, from_day, len(calendar) - 1
        )
        for day_i, day in enumerate(calendar[from_day:], start=from_day):
            expected = int(listed_mask(day, columns, delist).sum())
            got = int(widths[day_i - from_day])
            if got != expected:
                raise ValueError(f"{day} 行宽 {got} != 仍上市 {expected}，拒绝发布")
        f32_path, _ = self._factor_paths(factor_name)
        self._append_payload(f32_path, packed)
        indptr = np.zeros(len(calendar) + 1, dtype=_INT64)
        indptr[: layout.n_days + 1] = layout.indptr
        if widths.size:
            indptr[from_day + 1 :] = int(layout.indptr[-1]) + np.cumsum(widths)
        self._publish_layout(factor_name, calendar, columns, delist_lines, indptr)
        return int(packed.size)

    def _patch_published_days(
        self,
        factor_name: str,
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist: Dict[int, str],
        values: Dict[str, Dict[str, float]],
        layout: FactorLayout,
        day_indices: Sequence[int],
    ) -> int:
        f32_path, _ = self._factor_paths(factor_name)
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
        fd = os.open(str(f32_path), flags, 0o666)
        written = 0
        try:
            for day_i in day_indices:
                day = calendar[day_i]
                packed, widths = self._pack_days(
                    calendar, columns[: layout.n_columns], delist, values, day_i, day_i
                )
                expected = int(layout.indptr[day_i + 1] - layout.indptr[day_i])
                if int(widths[0]) != expected:
                    raise ValueError(
                        f"{day} 已发布行宽 {expected}，当前仍上市 {int(widths[0])}，"
                        "不能原地改写"
                    )
                os.lseek(fd, int(layout.indptr[day_i]) * int(_FLOAT32.itemsize), os.SEEK_SET)
                os.write(fd, packed.tobytes())
                written += int(packed.size)
            os.fsync(fd)
        finally:
            os.close(fd)
        return written

    def _remap_bins(self, old: Sequence[str], new: Sequence[str]) -> None:
        logger.warning(
            "日历中间插入 %s 天（%s -> %s），重打包全部因子",
            len(new) - len(old),
            len(old),
            len(new),
        )
        columns = self.read_columns()
        delist = self.read_delist()
        delist_lines = self._delist_lines()
        names = self.list_stored_factor_names()
        staged: List[Tuple[Path, Path, Path, Path, bytes]] = []
        try:
            for name in names:
                try:
                    layout, payload, _cal, cols, dmap, _dlines = self._load_published(name)
                except ValueError:
                    continue
                values: Dict[str, Dict[str, float]] = {}
                used_cols = cols[: layout.n_columns]
                for day_i, day in enumerate(old[: layout.n_days]):
                    mask = listed_mask(day, used_cols, dmap)
                    row = payload[int(layout.indptr[day_i]) : int(layout.indptr[day_i + 1])]
                    day_vals: Dict[str, float] = {}
                    cursor = 0
                    for keep, col in zip(mask.tolist(), used_cols):
                        if not keep:
                            continue
                        if cursor < row.size:
                            day_vals[col.symbol] = float(row[cursor])
                        cursor += 1
                    values[day] = day_vals
                packed, widths = self._pack_days(new, columns, delist, values, 0, len(new) - 1)
                indptr = np.zeros(len(new) + 1, dtype=_INT64)
                if widths.size:
                    np.cumsum(widths, out=indptr[1:])
                cal_h, col_h, del_h = self._prefix_hashes(
                    new, columns, delist_lines, len(new), len(columns), len(delist_lines)
                )
                new_layout = FactorLayout(
                    n_days=len(new),
                    n_floats=int(indptr[-1]),
                    n_columns=len(columns),
                    n_delist=len(delist_lines),
                    calendar_hash=cal_h,
                    columns_hash=col_h,
                    delist_hash=del_h,
                    indptr=indptr,
                )
                f32_path, layout_path = self._factor_paths(name)
                f32_tmp = Path(str(f32_path) + ".part")
                layout_tmp = Path(str(layout_path) + ".part")
                f32_tmp.write_bytes(np.asarray(packed, dtype=_FLOAT32).tobytes())
                layout_tmp.write_bytes(new_layout.dumps())
                staged.append((f32_tmp, f32_path, layout_tmp, layout_path, packed.tobytes()))
            self.write_calendar(list(new))
            for f32_tmp, f32_path, layout_tmp, layout_path, _blob in staged:
                _replace_with_retry(f32_tmp, f32_path)
                _replace_with_retry(layout_tmp, layout_path)
        except Exception:
            for f32_tmp, _f32, layout_tmp, _layout, _blob in staged:
                for part in (f32_tmp, layout_tmp):
                    try:
                        if part.exists():
                            part.unlink()
                    except OSError:
                        pass
            raise

    def upsert_factor_data(
        self,
        factor_name: str,
        data: pd.DataFrame,
        replace: bool = False,
        price_adjust: Optional[str] = None,
        *,
        align_axis: bool = True,
    ) -> int:
        if data is None or data.empty:
            return 0
        required = {"ts_code", "trade_date", "factor_value"}
        if not required.issubset(data.columns):
            raise ValueError(f"因子数据必须包含字段: {sorted(required)}")

        frame = data.loc[:, list(required)].copy()
        frame["ts_code"] = frame["ts_code"].map(lambda x: normalize_symbol(str(x)))
        frame["trade_date"] = frame["trade_date"].map(_iso_date)
        frame["factor_value"] = pd.to_numeric(frame["factor_value"], errors="coerce")
        frame = frame.dropna(subset=["ts_code", "trade_date"])
        if frame.empty:
            return 0
        frame = frame.drop_duplicates(["ts_code", "trade_date"], keep="last")
        written_values = int(frame["factor_value"].notna().sum())
        if written_values == 0:
            logger.error("因子 %s 全为 NaN，拒绝写入并推进水位", factor_name)
            return 0

        dates_in = sorted(frame["trade_date"].unique())
        calendar = self.ensure_calendar(dates_in) if align_axis else self.read_calendar()
        if not calendar:
            raise ValueError("没有交易日历，无法写入因子")
        if align_axis:
            self.ensure_instruments(frame)

        values: Dict[str, Dict[str, float]] = {}
        for row in frame.itertuples(index=False):
            values.setdefault(str(row.trade_date), {})[str(row.ts_code)] = (
                float(row.factor_value) if pd.notna(row.factor_value) else np.nan
            )

        columns = self.read_columns()
        delist = self.read_delist()
        delist_lines = self._delist_lines()
        existing_layout = self._read_layout(factor_name)
        is_new = existing_layout is None and factor_name not in self._meta

        if replace or is_new or existing_layout is None:
            existing = None
            if replace and existing_layout is not None:
                try:
                    loaded = self._load_published(factor_name)
                    existing = (loaded[0], loaded[1], loaded[4])
                except ValueError:
                    existing = None
            self._rewrite_factor(
                factor_name, calendar, columns, delist, delist_lines, values, existing
            )
        else:
            date_index = {d: i for i, d in enumerate(calendar)}
            published = existing_layout.n_days
            patch_days = sorted(
                {
                    date_index[d]
                    for d in values
                    if d in date_index and date_index[d] < published
                }
            )
            need_tail = published < len(calendar)
            width_changed = False
            used_cols = columns[: existing_layout.n_columns]
            for day_i in patch_days:
                expected = int(
                    listed_mask(calendar[day_i], columns, delist).sum()
                )
                published_width = int(
                    existing_layout.indptr[day_i + 1] - existing_layout.indptr[day_i]
                )
                old_expected = int(listed_mask(calendar[day_i], used_cols, delist).sum())
                if expected != published_width or expected != old_expected:
                    width_changed = True
                    break
            if width_changed:
                loaded = self._load_published(factor_name)
                self._rewrite_factor(
                    factor_name,
                    calendar,
                    columns,
                    delist,
                    delist_lines,
                    values,
                    (loaded[0], loaded[1], loaded[4]),
                )
            else:
                if patch_days:
                    self._patch_published_days(
                        factor_name,
                        calendar,
                        columns,
                        delist,
                        values,
                        existing_layout,
                        patch_days,
                    )
                if need_tail:
                    self._append_factor_days(
                        factor_name,
                        calendar,
                        columns,
                        delist,
                        delist_lines,
                        values,
                        existing_layout,
                        published,
                    )

        min_date = str(min(dates_in))
        max_date = str(max(dates_in))
        if not replace and not is_new and factor_name in self._meta:
            prev = self._meta[factor_name]
            if prev.get("min_date"):
                min_date = min(min_date, str(prev["min_date"]))
            prev_max = prev.get("max_date")
            # 旧水位若超前于本次写入（未来开市日被写进日历），丢掉，避免永久跳过增量。
            if prev_max and str(prev_max) <= max_date:
                max_date = max(max_date, str(prev_max))
        self._set_factor_meta(
            factor_name,
            min_date=min_date,
            max_date=max_date,
            rows=written_values,
            n_symbols=len(columns),
            price_adjust=price_adjust,
        )
        logger.info(
            "Bin 写入 %s non_nan=%s symbols=%s calendar_days=%s",
            factor_name,
            written_values,
            len(columns),
            len(calendar),
        )
        return int(written_values)

    def get_factor_latest_date(self, factor_name: str) -> Optional[str]:
        """该因子上次写入的水位，只看 meta，不用日历轴最后一天。

        日历轴可能含未来开市日；数据截止在 SQLite 的
        ``get_latest_market_date()``。
        """
        meta = self._meta.get(factor_name)
        if meta and meta.get("max_date"):
            return str(meta["max_date"])
        return None

    def is_factor_registered(self, factor_name: str) -> bool:
        if factor_name in self._meta:
            return True
        _f32, layout_path = self._factor_paths(factor_name)
        return layout_path.exists()

    def list_stored_factor_names(self) -> List[str]:
        names = set(self._meta)
        if self.features_dir.is_dir():
            for path in self.features_dir.glob("*.layout"):
                names.add(path.stem)
            for path in self.features_dir.glob("*.f32"):
                names.add(path.stem)
        return sorted(names)

    def delete_factor_data(self, factor_name: str) -> int:
        rows_by_name = self.delete_factors([factor_name])
        return int(rows_by_name.get(factor_name, 0))

    def delete_factors(self, names: Sequence[str]) -> Dict[str, int]:
        rows_by_name: Dict[str, int] = {}
        unique = [str(name) for name in names if str(name)]
        if not unique:
            return rows_by_name
        for factor_name in unique:
            meta = self._meta.get(factor_name) or {}
            rows_by_name[factor_name] = int(meta.get("rows") or 0)
            f32_path, layout_path = self._factor_paths(factor_name)
            _unlink_with_retry(f32_path)
            _unlink_with_retry(layout_path)
            self._meta.pop(factor_name, None)
            logger.info("已删下线因子: %s (rows=%s)", factor_name, rows_by_name[factor_name])
        self._commit_meta(removed=unique)
        return rows_by_name

    def _expand_panel(
        self,
        payload: np.ndarray,
        layout: FactorLayout,
        calendar: Sequence[str],
        columns: Sequence[Column],
        delist: Dict[int, str],
        start_i: int,
        end_i: int,
        wanted: Optional[set],
    ) -> pd.DataFrame:
        used_cols = columns[: layout.n_columns]
        symbols = [col.symbol for col in used_cols]
        if wanted is not None:
            col_keep = [i for i, sym in enumerate(symbols) if sym in wanted]
            out_symbols = [symbols[i] for i in col_keep]
        else:
            col_keep = list(range(len(symbols)))
            out_symbols = symbols
        n = end_i - start_i + 1
        matrix = np.full((n, len(out_symbols)), np.nan, dtype=_FLOAT32)
        if not out_symbols:
            return pd.DataFrame(
                index=pd.Index(calendar[start_i : end_i + 1], name="trade_date")
            )
        slot_to_out = {used_cols[i].slot: j for j, i in enumerate(col_keep)}
        day_i = start_i
        while day_i <= end_i:
            if day_i >= layout.n_days:
                break
            day = calendar[day_i]
            mask = listed_mask(day, used_cols, delist)
            width = int(layout.indptr[day_i + 1] - layout.indptr[day_i])
            run_end = day_i + 1
            while run_end <= end_i and run_end < layout.n_days:
                other = listed_mask(calendar[run_end], used_cols, delist)
                if not np.array_equal(mask, other):
                    break
                run_end += 1
            run_n = run_end - day_i
            start = int(layout.indptr[day_i])
            stop = int(layout.indptr[run_end])
            block = payload[start:stop].reshape(run_n, width)
            listed_slots = [col.slot for col, keep in zip(used_cols, mask.tolist()) if keep]
            dest = [slot_to_out[s] for s in listed_slots if s in slot_to_out]
            src = [k for k, s in enumerate(listed_slots) if s in slot_to_out]
            if dest:
                matrix[day_i - start_i : run_end - start_i, dest] = block[:, src]
            day_i = run_end
        index = pd.Index(calendar[start_i : end_i + 1], name="trade_date")
        return pd.DataFrame(matrix, index=index, columns=out_symbols)

    def read_factor_data(
        self,
        factor_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        panel = self.read_factor_panel(
            factor_name, start_date=start_date, end_date=end_date, symbols=symbols
        )
        if panel is None or panel.empty:
            return pd.DataFrame(columns=["ts_code", "trade_date", "factor_value"])
        date_col = panel.index.name or "trade_date"
        long = panel.reset_index().melt(
            id_vars=date_col, var_name="ts_code", value_name="factor_value"
        )
        long = long.dropna(subset=["factor_value"])
        if long.empty:
            return pd.DataFrame(columns=["ts_code", "trade_date", "factor_value"])
        long = long.rename(columns={date_col: "trade_date"})
        return long.loc[:, ["ts_code", "trade_date", "factor_value"]]

    def read_factor_panel(
        self,
        factor_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        panels = self.read_factor_panels(
            [factor_name],
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )
        return panels.get(factor_name, pd.DataFrame())

    def read_factor_panels(
        self,
        factor_names: Sequence[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        names = [str(name) for name in factor_names if str(name)]
        if not names:
            return {}
        calendar = self.read_calendar()
        if not calendar:
            return {name: pd.DataFrame() for name in names}
        start_i, end_i = self._calendar_slice(calendar, start_date, end_date)
        if start_i > end_i:
            return {name: pd.DataFrame() for name in names}
        wanted = {normalize_symbol(str(s)) for s in symbols} if symbols else None
        out: Dict[str, pd.DataFrame] = {}
        for name in names:
            try:
                layout, payload, cal, columns, delist, _lines = self._load_published(name)
            except ValueError as exc:
                raise ValueError(f"拒绝读取因子 {name}: {exc}") from exc
            out[name] = self._expand_panel(
                payload, layout, cal, columns, delist, start_i, end_i, wanted
            )
        return out
