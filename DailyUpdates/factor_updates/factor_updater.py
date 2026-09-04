#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discover pandas factors, calculate them, and persist factor values to Qlib-style bins.

Market / sidecar data still come from SQLite via ``SQLiteStorage``.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

for _p in Path(__file__).resolve().parents:
    if (_p / "yg_quant_repo.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
else:
    raise RuntimeError("找不到 yg_quant 仓库根目录（缺少 yg_quant_repo.py）")
from yg_quant_repo import default_db_path, ensure_repo_root

ensure_repo_root()

from DailyUpdates.factor_updates.base_factor import BaseFactor
from DailyUpdates.storage import BinStorage, SQLiteStorage

# 滚动窗口类因子的默认回看自然日（覆盖约 250 交易日量级）。
_FACTOR_LOOKBACK_DAYS = 450


def _parse_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


class FactorUpdater:
    def __init__(
        self,
        db_path: str,
        max_workers: int = 4,
        *,
        rebuild: Optional[bool] = None,
    ):
        self.storage = SQLiteStorage(db_path)
        self.max_workers = max_workers
        self.rebuild = _env_flag("YG_QUANT_FACTOR_REBUILD") if rebuild is None else bool(rebuild)
        self.logger = logging.getLogger("FactorUpdater")
        if self.rebuild:
            self.logger.warning("强制全量重算模式：所有选中因子将整段覆盖写入")
        self.factors: List[BaseFactor] = []
        self._import_errors: List[str] = []
        # 供需要读库附属表的因子使用（行业映射等）
        os.environ.setdefault("YG_QUANT_DB_PATH", str(self.storage.db_path))
        # 因子数值存 Qlib 风格 bin；行情仍读 SQLite
        factor_dir = os.getenv("YG_QUANT_FACTOR_DIR") or str(
            Path(self.storage.db_path).parent / "factors"
        )
        self.factor_storage = BinStorage(factor_dir)
        self.logger.info("因子存储: Bin dir=%s", factor_dir)
        self._load_factors()

    def _load_factors(self) -> None:
        """递归加载 factors/ 及子目录中所有 BaseFactor 子类。"""
        factors_dir = Path(__file__).parent / "factors"
        if not factors_dir.is_dir():
            self.logger.warning(f"因子目录不存在: {factors_dir}")
            return

        for path in sorted(factors_dir.rglob("*.py")):
            if path.name.startswith("_") or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(factors_dir).with_suffix("")
            module_name = "DailyUpdates.factor_updates.factors." + ".".join(rel.parts)
            try:
                module = importlib.import_module(module_name)
            except Exception:
                self.logger.exception(f"导入因子模块失败: {module_name}")
                self._import_errors.append(module_name)
                continue

            loaded = 0
            for name in dir(module):
                if name.startswith("_"):
                    continue
                candidate = getattr(module, name)
                if (
                    isinstance(candidate, type)
                    and issubclass(candidate, BaseFactor)
                    and candidate is not BaseFactor
                    and candidate.__module__ == module.__name__
                ):
                    factor = candidate()
                    self.factors.append(factor)
                    loaded += 1
                    self.logger.debug(
                        f"加载因子: {factor.get_name()} <- {module_name}"
                    )
            if loaded:
                self.logger.info(f"加载因子模块 {module_name}: {loaded} 个")

    def _history_start_env(self) -> Optional[str]:
        """可选：YG_QUANT_FACTOR_START_DATE 限制首次写入起点；未设置则写满可用历史。"""
        return _parse_date(os.getenv("YG_QUANT_FACTOR_START_DATE"))

    def _factor_output_start(self, factor_name: str) -> Optional[str]:
        if self.rebuild:
            return self._history_start_env()
        latest = self.factor_storage.get_factor_latest_date(factor_name)
        if latest:
            return (pd.Timestamp(latest) + timedelta(days=1)).strftime("%Y-%m-%d")
        return self._history_start_env()

    def _is_full_rebuild(self, factor_name: str) -> bool:
        if self.rebuild:
            return True
        return self.factor_storage.get_factor_latest_date(factor_name) is None

    def _price_adjust(self) -> str:
        return (os.getenv("YG_QUANT_PRICE_ADJUST") or "hfq").strip().lower()

    def _assert_adjust_matches(self, names: Sequence[str]) -> None:
        current = self._price_adjust()
        mismatches = []
        for name in names:
            stored = (self.factor_storage._meta.get(name) or {}).get("price_adjust")
            if stored and str(stored).strip().lower() != current:
                mismatches.append(f"{name} meta={stored} runtime={current}")
        if mismatches:
            raise RuntimeError(
                "因子复权口径与当前 YG_QUANT_PRICE_ADJUST 不一致，请使用 --rebuild: "
                + "; ".join(mismatches)
            )

    def _lookback_days(self, factors: Sequence[BaseFactor]) -> int:
        values = [
            int(getattr(factor, "lookback_days", _FACTOR_LOOKBACK_DAYS) or _FACTOR_LOOKBACK_DAYS)
            for factor in factors
        ]
        return max(values) if values else _FACTOR_LOOKBACK_DAYS

    def _select_factors(self, names: Optional[Sequence[str]]) -> List[BaseFactor]:
        if not names:
            return list(self.factors)
        tokens = [str(token).strip() for token in names if str(token).strip()]
        selected: List[BaseFactor] = []
        seen = set()
        for factor in self.factors:
            name = factor.get_name()
            matched = False
            for token in tokens:
                if token.endswith("*"):
                    matched = name.startswith(token[:-1])
                else:
                    matched = name == token
                if matched:
                    break
            if matched and name not in seen:
                selected.append(factor)
                seen.add(name)
        if not selected:
            raise ValueError(f"没有匹配到因子: {tokens}")
        self.logger.info(
            "仅更新因子 %s 个: %s",
            len(selected),
            ", ".join(f.get_name() for f in selected),
        )
        return selected

    def _resolve_load_start(
        self,
        end_date: Optional[str],
        factors: Optional[Sequence[BaseFactor]] = None,
    ) -> Optional[str]:
        """行情加载起点 = min(各因子输出起点) - lookback；全量历史则返回 None。"""
        selected = list(factors) if factors is not None else self.factors
        output_starts: List[Optional[str]] = [
            self._factor_output_start(factor.get_name()) for factor in selected
        ]
        if any(start is None for start in output_starts):
            self.logger.info(
                "将加载全部历史行情（首次全量或未设 YG_QUANT_FACTOR_START_DATE）"
            )
            return None

        earliest = min(pd.Timestamp(s) for s in output_starts if s is not None)
        lookback = self._lookback_days(selected)
        load_start = (earliest - timedelta(days=lookback)).strftime("%Y-%m-%d")
        self.logger.info(
            "行情加载窗口: %s -> %s (输出最早=%s, lookback=%sd)",
            load_start,
            end_date or "latest",
            earliest.strftime("%Y-%m-%d"),
            lookback,
        )
        return load_start

    def _load_market_data(
        self,
        end_date: Optional[str],
        start_date: Optional[str] = None,
        factors: Optional[Sequence[BaseFactor]] = None,
    ) -> pd.DataFrame:
        selected = list(factors) if factors is not None else self.factors
        fields = sorted(
            {
                dependency
                for factor in selected
                for dependency in factor.dependencies
            }
        )
        if not fields:
            return pd.DataFrame()

        adjust = (os.getenv("YG_QUANT_PRICE_ADJUST") or "hfq").strip().lower()
        self.logger.info(
            "开始加载行情: fields=%s start=%s end=%s adjust=%s (无 SQL ORDER BY)",
            fields,
            start_date or "ALL",
            end_date or "ALL",
            adjust,
        )
        started = time.time()
        data = self.storage.read_market_data(
            fields=fields,
            start_date=start_date,
            end_date=end_date,
            ordered=False,
            adjust=adjust,
        )
        if not data.empty:
            data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.strftime(
                "%Y-%m-%d"
            )
            data = data.sort_values(["ts_code", "trade_date"])
        self.logger.info(
            "行情加载完成: rows=%s symbols=%s elapsed=%.1fs",
            len(data),
            data["ts_code"].nunique() if not data.empty else 0,
            time.time() - started,
        )
        return data

    def _calculate_factor(
        self, factor: BaseFactor, market_data: pd.DataFrame, end_date: Optional[str]
    ) -> Dict:
        name = factor.get_name()
        start = self._factor_output_start(name)
        replace = self._is_full_rebuild(name)
        if start and end_date:
            if start > pd.Timestamp(end_date).strftime("%Y-%m-%d"):
                return {
                    "factor_name": name,
                    "rows": 0,
                    "duration": 0.0,
                    "empty": True,
                }

        started = time.time()
        result = factor.calculate(market_data, start_date=start)
        if end_date and not result.empty:
            keep = pd.to_datetime(result["trade_date"]) <= pd.Timestamp(end_date)
            result = result.loc[keep]
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"{name}.calculate 必须返回 DataFrame，实际为 {type(result)}")
        return {
            "factor_name": name,
            "result": result,
            "replace": replace,
            "started": started,
            "calc_s": time.time() - started,
            "empty": False,
        }

    def _persist_factor(self, payload: Dict, *, align_axis: bool = True) -> Dict:
        name = payload["factor_name"]
        if payload.get("empty"):
            return {"factor_name": name, "rows": 0, "duration": 0.0}
        result = payload["result"]
        replace = bool(payload["replace"])
        started = float(payload["started"])
        self.logger.info(
            "写入因子 %s: rows=%s mode=%s",
            name,
            len(result),
            "REPLACE" if replace else "APPEND",
        )
        write_started = time.time()
        rows = self.factor_storage.upsert_factor_data(
            name,
            result,
            replace=replace,
            price_adjust=self._price_adjust(),
            align_axis=align_axis,
        )
        self.logger.info(
            "因子 %s 写入完成: rows=%s calc=%.1fs write=%.1fs",
            name,
            rows,
            payload.get("calc_s", write_started - started),
            time.time() - write_started,
        )
        del result
        payload["result"] = None
        return {
            "factor_name": name,
            "rows": rows,
            "duration": time.time() - started,
        }

    def _update_factor(
        self, factor: BaseFactor, market_data: pd.DataFrame, end_date: Optional[str]
    ) -> Dict:
        return self._persist_factor(self._calculate_factor(factor, market_data, end_date))

    def _prepare_incremental_axis(
        self,
        market_data: pd.DataFrame,
        factors: Sequence[BaseFactor],
        end_date: Optional[str],
    ) -> None:
        """日频增量先把日历/instruments 对齐一次，避免 6 个因子同时抢轴锁、各写一遍 all.txt。"""
        starts = [self._factor_output_start(factor.get_name()) for factor in factors]
        dated = [start for start in starts if start]
        if market_data.empty:
            return
        frame = market_data.loc[:, ["ts_code", "trade_date"]]
        if dated:
            frame = frame.loc[frame["trade_date"] >= min(dated)]
        if end_date:
            frame = frame.loc[
                frame["trade_date"] <= pd.Timestamp(end_date).strftime("%Y-%m-%d")
            ]
        if frame.empty:
            return
        self.factor_storage.ensure_instruments(frame)
        self.logger.info(
            "已预对齐 instruments: symbols=%s dates=%s..%s",
            frame["ts_code"].nunique(),
            frame["trade_date"].min(),
            frame["trade_date"].max(),
        )

    def purge_removed_factors(self) -> List[Dict]:
        """Delete stored series for factor names no longer present as BaseFactor classes."""
        active = {factor.get_name() for factor in self.factors}
        stored = set(self.factor_storage.list_stored_factor_names())
        orphans = sorted(stored - active)
        removed: List[Dict] = []
        for name in orphans:
            rows = self.factor_storage.delete_factor_data(name)
            removed.append({"factor_name": name, "deleted_rows": rows})
            self.logger.info(f"删除已下线因子: {name} (rows={rows})")
        if not orphans:
            self.logger.info("没有需要删除的下线因子")
        return removed

    def update_factors(
        self,
        end_date: Optional[str] = None,
        names: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        if self._import_errors:
            raise RuntimeError(
                "因子模块导入失败，拒绝更新/清理: " + ", ".join(self._import_errors)
            )
        selected = self._select_factors(names)
        # 全量更新才清已下线因子；指定 names 时不要误删未选中的系列。
        if names is None:
            self.purge_removed_factors()

        if not selected:
            self.logger.warning("没有发现可计算的因子")
            return []

        # 默认截止日 = 交易日历末日（trade_calendar MAX），不扫 market_data。
        if end_date is None:
            end_date = self.storage.get_latest_market_date()
        if not end_date:
            self.logger.error("交易日历为空，无法确定因子 end_date")
            return []
        end_date = _parse_date(end_date)
        self.logger.info("因子更新 end_date=%s（交易日历）", end_date)

        # 用库内交易日历对齐 bin 轴（只增不改序）
        cal_dates = self.storage.list_trade_dates(end_date=end_date)
        if cal_dates:
            self.factor_storage.ensure_calendar(cal_dates)

        load_start = self._resolve_load_start(end_date, selected)
        market_data = self._load_market_data(
            end_date, start_date=load_start, factors=selected
        )
        if market_data.empty:
            self.logger.error("SQLite 中没有可用于因子计算的行情数据")
            return []

        full_rebuild = [
            f for f in selected if self._is_full_rebuild(f.get_name())
        ]
        incremental = [
            f for f in selected if not self._is_full_rebuild(f.get_name())
        ]
        if incremental:
            self._assert_adjust_matches([f.get_name() for f in incremental])
        results: List[Dict] = []

        # 全量重建：串行，算一个写一个并释放，避免多因子结果叠加占满内存
        if full_rebuild:
            self.logger.info(
                "全量重建因子 %s 个：串行计算，每因子写完即释放",
                len(full_rebuild),
            )
            for factor in tqdm(full_rebuild, desc="全量重建因子"):
                results.append(self._update_factor(factor, market_data, end_date))
                gc.collect()

        # 日频增量：计算并行吃 CPU，写入单独排队，避免 6×32 路同时 open 5500 个 bin
        if incremental:
            self._prepare_incremental_axis(market_data, incremental, end_date)
            calc_workers = max(1, min(len(incremental), self.max_workers))
            write_workers = max(
                1, int(os.getenv("YG_QUANT_FACTOR_WRITE_WORKERS", "1"))
            )
            write_workers = min(write_workers, len(incremental))
            self.logger.info(
                "增量更新因子 %s 个：calc_workers=%s write_workers=%s",
                len(incremental),
                calc_workers,
                write_workers,
            )
            write_pool = ThreadPoolExecutor(max_workers=write_workers)
            write_futures = []
            try:
                with ThreadPoolExecutor(max_workers=calc_workers) as calc_pool:
                    calc_futures = {
                        calc_pool.submit(
                            self._calculate_factor, factor, market_data, end_date
                        ): factor.get_name()
                        for factor in incremental
                    }
                    for future in tqdm(
                        as_completed(calc_futures),
                        total=len(calc_futures),
                        desc="增量计算因子",
                    ):
                        write_futures.append(
                            write_pool.submit(
                                self._persist_factor,
                                future.result(),
                                align_axis=False,
                            )
                        )
                for future in tqdm(
                    as_completed(write_futures),
                    total=len(write_futures),
                    desc="增量写入因子",
                ):
                    results.append(future.result())
            finally:
                write_pool.shutdown(wait=True)

        return results

    def get_factor_status(self) -> Dict[str, Dict]:
        return {
            factor.get_name(): {
                "registered": self.factor_storage.is_factor_registered(factor.get_name()),
                "description": factor.description,
                "dependencies": factor.dependencies,
                "latest_date": self.factor_storage.get_factor_latest_date(
                    factor.get_name()
                ),
                "factor_type": factor.__class__.__name__,
            }
            for factor in self.factors
        }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="更新因子并写入 Bin 存储")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制全量重算并覆盖已有序列（等价 YG_QUANT_FACTOR_REBUILD=1），"
        "切换复权口径后必须使用",
    )
    parser.add_argument(
        "--only",
        default=os.getenv("YG_QUANT_FACTOR_ONLY"),
        help="逗号分隔的因子名，支持 prefix* 通配",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("YG_QUANT_FACTOR_WORKERS", "8")),
        help="增量更新计算线程数（写入另由 YG_QUANT_FACTOR_WRITE_WORKERS 控制）",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO)
    names = (
        [part.strip() for part in str(args.only).split(",") if part.strip()]
        if args.only
        else None
    )
    updater = FactorUpdater(
        str(default_db_path()),
        max_workers=max(1, int(args.workers)),
        rebuild=True if args.rebuild else None,
    )
    updater.update_factors(names=names)


if __name__ == "__main__":
    main()
