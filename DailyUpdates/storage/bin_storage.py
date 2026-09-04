#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Binary factor store (per-instrument day bins, Qlib-like layout).

Layout::

    {root}/
      calendars/day.txt
      instruments/all.txt
      features/{SH600000}/alpha001.day.bin
      _factor_meta.json

Symbol IDs match the SQLite convention (``SH600000`` / ``SZ000001``), uppercase
everywhere — directories and ``instruments/all.txt`` included.

Each ``*.day.bin`` is little-endian float32. Full files match
``len(calendars/day.txt)``; incremental writes may leave absent symbols short.
Reads treat a shorter file as trailing NaN. A file longer than the calendar is
rejected. Stock identity / dates live only in the axis files.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("BinStorage")

_FLOAT32 = np.dtype("<f4")

# 同一进程内多个 BinStorage 实例（因子并发更新会各建一个）共享同一把锁
_AXIS_LOCKS: Dict[str, threading.RLock] = {}
_AXIS_LOCKS_GUARD = threading.Lock()
_WRITE_POOL: Optional[ThreadPoolExecutor] = None
_WRITE_POOL_GUARD = threading.Lock()
_WRITE_POOL_WORKERS = 0


def _default_bin_write_workers() -> int:
    raw = (os.getenv("YG_QUANT_BIN_WRITE_WORKERS") or "").strip()
    if raw:
        return max(1, int(raw))
    # Windows NTFS + Defender：32 路同时 open 会互相踩，磁盘% 反而掉到个位数
    return 8 if os.name == "nt" else 16


def _bin_write_pool(workers: int) -> ThreadPoolExecutor:
    """进程内复用写线程池，避免每个因子都创建/销毁几十个线程。"""
    global _WRITE_POOL, _WRITE_POOL_WORKERS
    want = max(1, int(workers))
    with _WRITE_POOL_GUARD:
        if _WRITE_POOL is None:
            _WRITE_POOL = ThreadPoolExecutor(
                max_workers=want, thread_name_prefix="bin-write"
            )
            _WRITE_POOL_WORKERS = want
        return _WRITE_POOL


def _shutdown_write_pool() -> None:
    global _WRITE_POOL, _WRITE_POOL_WORKERS
    with _WRITE_POOL_GUARD:
        pool = _WRITE_POOL
        _WRITE_POOL = None
        _WRITE_POOL_WORKERS = 0
    if pool is not None:
        pool.shutdown(wait=False)


atexit.register(_shutdown_write_pool)


def _axis_lock(root: Path) -> threading.RLock:
    key = str(root)
    with _AXIS_LOCKS_GUARD:
        lock = _AXIS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _AXIS_LOCKS[key] = lock
        return lock


def _atomic_write_text(path: Path, text: str) -> None:
    """先写同目录临时文件再 os.replace，避免中途崩溃留下截断的轴文件。"""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_bytes(path: Path, blob: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, path)


def normalize_symbol(symbol: str) -> str:
    """Unify to DB style: ``000001.SZ`` / ``sz000001`` -> ``SZ000001``."""
    text = str(symbol).strip().upper()
    if "." in text:
        code, market = text.split(".", 1)
        return f"{market}{code}"
    return re.sub(r"[^0-9A-Z]", "", text) or "UNKNOWN"


class BinStorage:
    """Factor API compatible with the former ParquetStorage."""

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
        self._instruments_path = self.instruments_dir / "all.txt"
        self._instruments_cache: Optional[Dict[str, Tuple[str, str]]] = None
        self._ensured_dirs: set[str] = set()
        self._write_workers = _default_bin_write_workers()
        self._read_symbol_cache_key: Optional[object] = None
        self._read_symbol_list: List[str] = []

    def _safe_factor_name(self, factor_name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in factor_name)

    def _factor_file(self, symbol: str, factor_name: str) -> Path:
        return self.features_dir / normalize_symbol(symbol) / f"{self._safe_factor_name(factor_name)}.day.bin"

    def _calendar_slice(
        self, calendar: Sequence[str], start_date: Optional[str], end_date: Optional[str]
    ) -> Tuple[int, int]:
        start_i = 0
        end_i = len(calendar) - 1
        if start_date:
            start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
            start_i = next((i for i, d in enumerate(calendar) if d >= start), len(calendar))
        if end_date:
            end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
            end_i = next(
                (i for i in range(len(calendar) - 1, -1, -1) if calendar[i] <= end),
                -1,
            )
        return start_i, end_i

    def _slice_bin(self, path: str, n_calendar: int, start_i: int, n: int) -> np.ndarray:
        """Read ``n`` days from ``start_i``. Short files pad trailing NaN."""
        n_file = self._bin_len(Path(path))
        if n_file is None:
            return np.full(n, np.nan, dtype=_FLOAT32)
        self._reject_bin_if_longer(Path(path), n_calendar, n_file)
        if start_i == 0 and n == n_calendar and n_file == n_calendar:
            return np.fromfile(path, dtype=_FLOAT32)
        out = np.full(n, np.nan, dtype=_FLOAT32)
        if n_file <= start_i or n <= 0:
            return out
        take = min(n, n_file - start_i)
        need_bytes = int(take * _FLOAT32.itemsize)
        try:
            with open(path, "rb") as handle:
                handle.seek(int(start_i * _FLOAT32.itemsize))
                raw = handle.read(need_bytes)
        except OSError:
            return out
        if not raw:
            return out
        got = np.frombuffer(raw, dtype=_FLOAT32)
        copy_n = min(got.size, take)
        if copy_n > 0:
            out[:copy_n] = got[:copy_n]
        return out

    @contextmanager
    def _axis_guard(self, timeout: float = 60.0) -> Iterator[None]:
        """进程内锁 + 跨进程锁文件，保护日历 / instruments / 元数据的读改写。"""
        with self._lock:
            fd = None
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    break
                except FileExistsError:
                    if time.monotonic() >= deadline:
                        # 多半是上次崩溃留下的死锁文件；抢过来继续，别把日更卡死
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
        """读盘合并后原子写回。

        多个因子并发更新时各持一份内存 meta，直接整体覆盖会把别人刚写的条目
        抹掉。这里只把本次真正改动的键合并进磁盘上的最新版本。
        """
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
        """启动时丢掉未完成 remap 的 staging 文件。"""
        if not self.features_dir.is_dir():
            return
        for path in self.features_dir.glob("*/*.day.bin.part"):
            try:
                path.unlink()
                logger.warning("丢弃残留 remap staging: %s", path)
            except OSError:
                logger.warning("无法删除残留 remap staging: %s", path, exc_info=True)

    def _bin_len(self, path: Path) -> Optional[int]:
        try:
            return os.path.getsize(path) // int(_FLOAT32.itemsize)
        except OSError:
            return None

    def _reject_bin_if_longer(self, path: Path, n_days: int, n_file: int) -> None:
        if n_file > int(n_days):
            raise ValueError(
                f"{path} 长度 {n_file} 长于日历 {n_days}，拒绝读取；请重建因子轴"
            )

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
            "layout": "qlib_day_bin",
        }
        adj = price_adjust if price_adjust is not None else prev.get("price_adjust")
        if adj:
            rec["price_adjust"] = str(adj)
        self._meta[factor_name] = rec
        self._commit_meta(changed=[factor_name])

    def read_calendar(self) -> List[str]:
        if not self._calendar_path.exists():
            return []
        out: List[str] = []
        for line in self._calendar_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                out.append(pd.Timestamp(text).strftime("%Y-%m-%d"))
        return out

    def write_calendar(self, dates: Sequence[str]) -> None:
        normalized = sorted({pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates if d})
        _atomic_write_text(
            self._calendar_path,
            "\n".join(normalized) + ("\n" if normalized else ""),
        )

    def ensure_calendar(self, dates: Sequence[str]) -> List[str]:
        """并入新交易日。若是「中间插入」则把所有 bin 按新索引重排。

        bin 文件只存值，日期靠位置对齐日历。补一个历史上漏掉的交易日会把它
        之后的所有值整体错位一格，而且悄无声息 —— 必须同步重映射。
        """
        with self._axis_guard():
            before = self.read_calendar()
            incoming = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates if d}
            merged = sorted(set(before) | incoming)
            if merged == before:
                return merged
            # 纯尾部追加：旧日历是新日历的前缀，位置不变，直接写。
            if merged[: len(before)] == before:
                self.write_calendar(merged)
                return merged
            self._remap_bins(before, merged)
            return merged

    def _remap_bins(self, old: Sequence[str], new: Sequence[str]) -> None:
        new_index = {day: i for i, day in enumerate(new)}
        mapping = np.array([new_index[day] for day in old], dtype=np.int64)
        inserted = len(new) - len(old)
        paths = list(self.features_dir.glob("*/*.day.bin"))
        logger.warning(
            "日历中间插入 %s 天（%s -> %s），重排 %s 个 bin 文件",
            inserted,
            len(old),
            len(new),
            len(paths),
        )
        n_new = len(new)
        staged: List[Tuple[Path, Path]] = []
        try:
            for path in paths:
                raw = np.fromfile(path, dtype=_FLOAT32)
                if raw.size == 0:
                    raw = np.full(mapping.size, np.nan, dtype=_FLOAT32)
                elif raw.size > mapping.size:
                    raise ValueError(
                        f"{path} 长度 {raw.size} 长于旧日历 {mapping.size}，"
                        "拒绝 remap；请重建因子轴"
                    )
                elif raw.size < mapping.size:
                    padded = np.full(mapping.size, np.nan, dtype=_FLOAT32)
                    padded[: raw.size] = raw
                    raw = padded
                series = np.full(n_new, np.nan, dtype=_FLOAT32)
                series[mapping] = raw
                part = Path(str(path) + ".part")
                part.write_bytes(series.tobytes())
                staged.append((part, path))
            self.write_calendar(list(new))
            for part, dest in staged:
                os.replace(part, dest)
        except Exception:
            for part, _dest in staged:
                try:
                    if part.exists():
                        part.unlink()
                except OSError:
                    pass
            raise

    def read_instruments(self) -> Dict[str, Tuple[str, str]]:
        """symbol -> (start_date, end_date)."""
        if self._instruments_cache is not None:
            return self._instruments_cache
        if not self._instruments_path.exists():
            self._instruments_cache = {}
            return self._instruments_cache
        result: Dict[str, Tuple[str, str]] = {}
        for line in self._instruments_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            parts = text.split("\t")
            if len(parts) < 3:
                continue
            inst_id, start, end = parts[0], parts[1], parts[2]
            symbol = normalize_symbol(inst_id)
            result[symbol] = (
                pd.Timestamp(start).strftime("%Y-%m-%d"),
                pd.Timestamp(end).strftime("%Y-%m-%d"),
            )
        self._instruments_cache = result
        return result

    def write_instruments(self, mapping: Dict[str, Tuple[str, str]]) -> None:
        lines = [
            f"{normalize_symbol(symbol)}\t{start}\t{end}"
            for symbol, (start, end) in sorted(mapping.items())
        ]
        _atomic_write_text(
            self._instruments_path,
            "\n".join(lines) + ("\n" if lines else ""),
        )
        self._instruments_cache = dict(mapping)

    def _merge_instruments(self, updates: Dict[str, Tuple[str, str]]) -> Dict[str, Tuple[str, str]]:
        """把区间并进磁盘上的最新版本，避免并发因子更新互相覆盖上市区间。"""
        with self._axis_guard():
            self._instruments_cache = None
            merged = dict(self.read_instruments())
            for symbol, (start, end) in updates.items():
                old = merged.get(symbol)
                merged[symbol] = (
                    (start, end) if old is None else (min(old[0], start), max(old[1], end))
                )
            self.write_instruments(merged)
            return merged

    def ensure_instruments(self, frame: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
        mapping = self.read_instruments()
        updates: Dict[str, Tuple[str, str]] = {}
        dates = pd.unique(frame["trade_date"].astype(str))
        if len(dates) == 1:
            day = str(dates[0])
            for ts_code in pd.unique(frame["ts_code"].astype(str)):
                symbol = normalize_symbol(ts_code)
                old = mapping.get(symbol)
                new = (day, day) if old is None else (min(old[0], day), max(old[1], day))
                if old != new:
                    updates[symbol] = new
        else:
            for ts_code, group in frame.groupby("ts_code")["trade_date"]:
                symbol = normalize_symbol(str(ts_code))
                dmin = str(group.min())
                dmax = str(group.max())
                old = mapping.get(symbol)
                new = (
                    (dmin, dmax)
                    if old is None
                    else (min(old[0], dmin), max(old[1], dmax))
                )
                if old != new:
                    updates[symbol] = new
        if not updates:
            return mapping
        return self._merge_instruments(updates)

    def _read_bin(self, path: Path, n_days: int) -> np.ndarray:
        if not path.exists():
            return np.full(n_days, np.nan, dtype=_FLOAT32)
        n_file = self._bin_len(path)
        if n_file is None:
            return np.full(n_days, np.nan, dtype=_FLOAT32)
        self._reject_bin_if_longer(path, n_days, n_file)
        raw = np.fromfile(path, dtype=_FLOAT32)
        if raw.size == n_days:
            return raw
        out = np.full(n_days, np.nan, dtype=_FLOAT32)
        if raw.size:
            out[: raw.size] = raw
        return out

    def _ensure_parent(self, path: Path) -> None:
        parent = str(path.parent)
        if parent in self._ensured_dirs:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ensured_dirs.add(parent)

    def _write_bytes(self, path: Path, buf: bytes, *, append: bool) -> None:
        flags = os.O_WRONLY | os.O_CREAT
        flags |= os.O_APPEND if append else os.O_TRUNC
        if getattr(os, "O_BINARY", 0):
            flags |= os.O_BINARY
        fd = os.open(path, flags, 0o666)
        try:
            os.write(fd, buf)
        finally:
            os.close(fd)

    def _write_bin(self, path: Path, values: np.ndarray) -> None:
        self._ensure_parent(path)
        self._write_bytes(path, np.asarray(values, dtype=_FLOAT32).tobytes(), append=False)

    def _append_bin(self, path: Path, values: np.ndarray) -> None:
        self._ensure_parent(path)
        self._write_bytes(path, np.asarray(values, dtype=_FLOAT32).tobytes(), append=True)

    def _patch_bin(
        self, path: Path, indices: np.ndarray, values: np.ndarray, n_days: int
    ) -> None:
        """Write values at calendar indices. New days at EOF are appended (4 bytes each)."""
        idx = np.asarray(indices, dtype=np.int64)
        vals = np.asarray(values, dtype=_FLOAT32)
        valid = (idx >= 0) & (idx < n_days)
        idx = idx[valid]
        vals = vals[valid]
        if idx.size == 0:
            return
        order = np.argsort(idx, kind="stable")
        idx = idx[order]
        vals = vals[order]
        self._ensure_parent(path)

        itemsize = int(_FLOAT32.itemsize)
        flags = os.O_RDWR | os.O_CREAT
        if getattr(os, "O_BINARY", 0):
            flags |= os.O_BINARY
        fd = os.open(path, flags, 0o666)
        try:
            raw_n = os.fstat(fd).st_size // itemsize
            min_i = int(idx[0])
            max_i = int(idx[-1])
            if raw_n == 0:
                series = np.full(n_days, np.nan, dtype=_FLOAT32)
                series[idx] = vals
                os.write(fd, series.tobytes())
                return
            if min_i >= raw_n:
                if min_i > raw_n:
                    os.lseek(fd, 0, os.SEEK_END)
                    os.write(fd, np.full(min_i - raw_n, np.nan, dtype=_FLOAT32).tobytes())
                tail = np.full(max_i - min_i + 1, np.nan, dtype=_FLOAT32)
                tail[idx - min_i] = vals
                os.lseek(fd, 0, os.SEEK_END)
                os.write(fd, tail.tobytes())
                return
            if max_i + 1 > raw_n:
                os.lseek(fd, 0, os.SEEK_END)
                os.write(fd, np.full(max_i + 1 - raw_n, np.nan, dtype=_FLOAT32).tobytes())
            blob = vals.tobytes()
            if idx.size == 1 or np.all(np.diff(idx) == 1):
                os.lseek(fd, int(idx[0]) * itemsize, os.SEEK_SET)
                os.write(fd, blob)
            else:
                for offset, start in enumerate(idx.tolist()):
                    os.lseek(fd, int(start) * itemsize, os.SEEK_SET)
                    os.write(fd, blob[offset * itemsize : (offset + 1) * itemsize])
        finally:
            os.close(fd)

    def _list_factor_dirs(self, factor_name: str) -> List[str]:
        return sorted(p.parent.name for p in self.features_dir.glob(f"*/{factor_name}.day.bin"))

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
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
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
        n_days = len(calendar)
        date_index = {d: i for i, d in enumerate(calendar)}
        if align_axis:
            self.ensure_instruments(frame)

        batch_dirs = {normalize_symbol(str(s)) for s in frame["ts_code"].unique()}
        is_new_factor = factor_name not in self._meta
        existing_dirs: set[str] = set()
        if replace:
            existing_dirs = set(self._list_factor_dirs(factor_name))
            is_new_factor = not existing_dirs and is_new_factor

        if replace or is_new_factor:
            for ts_code, group in frame.groupby("ts_code", sort=False):
                symbol = normalize_symbol(str(ts_code))
                series = np.full(n_days, np.nan, dtype=_FLOAT32)
                idx = group["trade_date"].map(date_index).to_numpy()
                vals = group["factor_value"].to_numpy(dtype=np.float64)
                series[idx] = vals.astype(_FLOAT32, copy=False)
                self._write_bin(self._factor_file(symbol, factor_name), series)
            if replace:
                for symbol in existing_dirs - batch_dirs:
                    path = self._factor_file(symbol, factor_name)
                    if path.exists():
                        path.unlink()
        else:
            # 增量：新交易日在日历末尾时只追加。慢的是 8500 次 open，不是 4 字节。
            jobs = []
            for ts_code, group in frame.groupby("ts_code", sort=False):
                symbol = normalize_symbol(str(ts_code))
                mapped = group["trade_date"].map(date_index)
                valid = mapped.notna()
                if not bool(valid.any()):
                    continue
                path = self._factor_file(symbol, factor_name)
                self._ensure_parent(path)
                jobs.append(
                    (
                        path,
                        mapped[valid].to_numpy(dtype=np.int64),
                        group.loc[valid, "factor_value"].to_numpy(dtype=np.float64),
                        n_days,
                    )
                )
            if len(jobs) <= 1 or self._write_workers <= 1:
                for job in jobs:
                    self._patch_bin(*job)
            else:
                workers = min(self._write_workers, len(jobs))
                chunk = max(16, len(jobs) // workers)
                pool = _bin_write_pool(self._write_workers)
                list(pool.map(lambda job: self._patch_bin(*job), jobs, chunksize=chunk))

        min_date = str(frame["trade_date"].min())
        max_date = str(frame["trade_date"].max())
        if not replace and not is_new_factor and factor_name in self._meta:
            prev = self._meta[factor_name]
            if prev.get("min_date"):
                min_date = min(min_date, str(prev["min_date"]))
            if prev.get("max_date"):
                max_date = max(max_date, str(prev["max_date"]))

        if replace or is_new_factor:
            n_symbols = len(self._list_factor_dirs(factor_name))
        else:
            prev_n = int((self._meta.get(factor_name) or {}).get("n_symbols") or 0)
            n_symbols = max(prev_n, len(batch_dirs))
        self._set_factor_meta(
            factor_name,
            min_date=min_date,
            max_date=max_date,
            rows=written_values,
            n_symbols=n_symbols,
            price_adjust=price_adjust,
        )
        logger.info(
            "Bin 写入 %s non_nan=%s symbols=%s calendar_days=%s",
            factor_name,
            written_values,
            n_symbols,
            n_days,
        )
        return int(written_values)

    def get_factor_latest_date(self, factor_name: str) -> Optional[str]:
        meta = self._meta.get(factor_name)
        if meta and meta.get("max_date"):
            return str(meta["max_date"])
        calendar = self.read_calendar()
        sample = next(self.features_dir.glob(f"*/{factor_name}.day.bin"), None)
        if sample is None or not calendar:
            return None
        raw = np.fromfile(sample, dtype=_FLOAT32)
        if raw.size == 0:
            return None
        last_i = min(raw.size, len(calendar)) - 1
        for i in range(last_i, -1, -1):
            if not np.isnan(raw[i]):
                latest = calendar[i]
                break
        else:
            latest = calendar[last_i]
        self._meta.setdefault(factor_name, {})["max_date"] = latest
        self._commit_meta(changed=[factor_name])
        return latest

    def is_factor_registered(self, factor_name: str) -> bool:
        if factor_name in self._meta:
            return True
        return next(self.features_dir.glob(f"*/{factor_name}.day.bin"), None) is not None

    def list_stored_factor_names(self) -> List[str]:
        names = {
            p.name[: -len(".day.bin")]
            for p in self.features_dir.glob("*/*.day.bin")
            if p.name.endswith(".day.bin")
        }
        return sorted(names)

    def delete_factor_data(self, factor_name: str) -> int:
        meta = self._meta.get(factor_name) or {}
        rows = int(meta.get("rows") or 0)
        for path in self.features_dir.glob(f"*/{factor_name}.day.bin"):
            path.unlink()
        self._meta.pop(factor_name, None)
        self._commit_meta(removed=[factor_name])
        return rows

    def read_factor_data(
        self,
        factor_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        calendar = self.read_calendar()
        if not calendar:
            return pd.DataFrame(columns=["ts_code", "trade_date", "factor_value"])

        start_i = 0
        end_i = len(calendar) - 1
        if start_date:
            start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
            start_i = next((i for i, d in enumerate(calendar) if d >= start), len(calendar))
        if end_date:
            end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
            end_i = next(
                (i for i in range(len(calendar) - 1, -1, -1) if calendar[i] <= end),
                -1,
            )
        if start_i > end_i:
            return pd.DataFrame(columns=["ts_code", "trade_date", "factor_value"])

        wanted = {normalize_symbol(str(s)) for s in symbols} if symbols else None
        chunks: List[pd.DataFrame] = []
        for path in self.features_dir.glob(f"*/{factor_name}.day.bin"):
            symbol = normalize_symbol(path.parent.name)
            if wanted is not None and symbol not in wanted:
                continue
            series = self._read_bin(path, len(calendar))
            sl = series[start_i : end_i + 1]
            dates = calendar[start_i : end_i + 1]
            mask = ~np.isnan(sl)
            if not mask.any():
                continue
            chunks.append(
                pd.DataFrame(
                    {
                        "ts_code": symbol,
                        "trade_date": np.asarray(dates, dtype=object)[mask],
                        "factor_value": sl[mask].astype(np.float64, copy=False),
                    }
                )
            )
        if not chunks:
            return pd.DataFrame(columns=["ts_code", "trade_date", "factor_value"])
        return pd.concat(chunks, ignore_index=True)

    def read_factor_panel(
        self,
        factor_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Wide panel: index=trade_date, columns=symbol, missing as NaN."""
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
        """Read several factors with one features/ directory walk (Windows glob is slow)."""
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
        n = end_i - start_i + 1
        dates = calendar[start_i : end_i + 1]
        files = {name: f"{self._safe_factor_name(name)}.day.bin" for name in names}
        n_calendar = len(calendar)
        cache_key = (frozenset(wanted) if wanted is not None else None, start_i, end_i)
        if cache_key == self._read_symbol_cache_key and self._read_symbol_list:
            symbol_list = list(self._read_symbol_list)
        else:
            symbol_list = []
            with os.scandir(self.features_dir) as entries:
                for entry in entries:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    symbol = entry.name
                    if wanted is not None and symbol not in wanted:
                        continue
                    symbol_list.append(symbol)
            symbol_list.sort()
            self._read_symbol_cache_key = cache_key
            self._read_symbol_list = list(symbol_list)

        symbol_index = {symbol: idx for idx, symbol in enumerate(symbol_list)}
        jobs: List[Tuple[int, str, str]] = []
        for symbol in symbol_list:
            idx = symbol_index[symbol]
            dir_path = self.features_dir / symbol
            try:
                present = set(os.listdir(dir_path))
            except OSError:
                continue
            for name in names:
                fname = files[name]
                if fname in present:
                    jobs.append((idx, name, str(dir_path / fname)))

        if not symbol_list:
            empty = pd.DataFrame(index=pd.Index(dates, name="trade_date"))
            return {name: empty.copy() for name in names}

        nan_col = np.full(n, np.nan, dtype=_FLOAT32)
        columns: Dict[str, List[np.ndarray]] = {
            name: [nan_col] * len(symbol_list) for name in names
        }

        def _load(job: Tuple[int, str, str]):
            idx, name, path = job
            return idx, name, self._slice_bin(path, n_calendar, start_i, n)

        workers = min(16, max(4, (os.cpu_count() or 8)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, name, arr in pool.map(_load, jobs, chunksize=64):
                columns[name][idx] = arr

        index = pd.Index(dates, name="trade_date")
        out: Dict[str, pd.DataFrame] = {}
        for name in names:
            matrix = np.column_stack(columns[name])
            out[name] = pd.DataFrame(matrix, index=index, columns=symbol_list)
        return out
