#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对照上次保存的 dataset_config：新增 install、删减清理、最后统一增量更新。"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from DailyUpdates.data_fetcher.dataset_installer import (
    always_full_sidecar,
    is_industry_dataset,
    is_sidecar_dataset,
)
from DailyUpdates.data_fetcher.data_processor import discover_data_source_classes
from DailyUpdates.data_fetcher.stock_data_updater import StockDataUpdater
from DailyUpdates.storage import SQLiteStorage
from yg_quant_repo import default_db_path

logger = logging.getLogger(__name__)


def _business_fields(config: Dict) -> Set[str]:
    return set(config.get("fields", [])) - {
        "ts_code",
        "trade_date",
        "symbol",
        "date",
        "index_code",
        "con_code",
        "l1_code",
        "l2_code",
        "l3_code",
        "in_date",
        "out_date",
        "ann_date",
        "end_date",
    }


def _sanitize_config(dataset_config: Dict) -> Dict:
    """落盘时去掉 token，避免把密钥写进配置快照。"""
    cleaned = {}
    for name, original in dataset_config.items():
        item = {
            key: value
            for key, value in original.items()
            if key.lower() != "token"
        }
        cleaned[name] = item
    return cleaned


def _index_symbol(ts_code: str) -> str:
    text = str(ts_code).strip().upper()
    if "." in text:
        code, market = text.split(".", 1)
        return f"index_{market}{code}"
    if text.startswith("INDEX_"):
        return text
    return f"index_{text}"


def _sidecar_snapshot(config: Dict) -> Dict:
    """sidecar 数据集用于快照对比的关键配置键。"""
    keys = (
        "data_source",
        "data_type",
        "api_name",
        "src",
        "levels",
        "is_new",
        "list_status",
        "use_vip",
        "periods",
        "index_list",
        "fields",
        "primary_key",
        "pause_seconds",
    )
    return {key: config.get(key) for key in keys if key in config}


class UnifiedScheduler:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or default_db_path()).expanduser().resolve()
        self.storage = SQLiteStorage(str(self.db_path))
        self.config_snapshot_path = self.db_path.parent / "dataset_config.json"

    def _load_previous_config(self) -> Dict:
        if not self.config_snapshot_path.exists():
            return {}
        try:
            with open(self.config_snapshot_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning(f"读取配置快照失败，将视为首次配置: {exc}")
            return {}

    def _save_config_snapshot(self, dataset_config: Dict) -> None:
        payload = _sanitize_config(dataset_config)
        self.config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_snapshot_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        logger.info(f"已保存配置快照: {self.config_snapshot_path}")

    def analyze_config(self, dataset_config: Dict) -> Dict:
        """对照上次保存的 dataset_config，拆分新增 / 删减 / 增量更新。"""
        previous = self._load_previous_config()
        full_rebuild = self.storage.get_latest_market_date() is None

        install_dataset_config: Dict = {}
        remove_datasets: Dict = {}
        remove_fields: Dict[str, List[str]] = {}

        if full_rebuild:
            # 空库：行情走统一重建；sidecar 维表单独 install。
            sidecar_install = {
                name: config.copy()
                for name, config in dataset_config.items()
                if is_sidecar_dataset(config)
            }
            return {
                "full_rebuild": True,
                "install_dataset_config": sidecar_install,
                "remove_datasets": {},
                "remove_fields": {},
                "update_dataset_config": dataset_config,
                "summary": {
                    "total_datasets": len(dataset_config),
                    "new_datasets": len(sidecar_install),
                    "removed_datasets": 0,
                    "new_fields": 0,
                    "removed_fields": 0,
                },
            }

        # 新增数据集 / 新增字段 → install
        for name, config in dataset_config.items():
            if name not in previous:
                install_dataset_config[name] = config.copy()
                continue

            if is_sidecar_dataset(config):
                if _sidecar_snapshot(config) != _sidecar_snapshot(previous[name]):
                    install_dataset_config[name] = config.copy()
                continue

            current_fields = _business_fields(config)
            old_fields = _business_fields(previous[name])
            new_fields = current_fields - old_fields
            if new_fields:
                install_config = config.copy()
                install_config["fields"] = sorted(
                    new_fields | {"ts_code", "trade_date"}
                )
                install_dataset_config[name] = install_config

        # 删减数据集 / 删减字段
        for name, old_config in previous.items():
            if name not in dataset_config:
                remove_datasets[name] = old_config
                continue
            if is_sidecar_dataset(old_config) or is_sidecar_dataset(
                dataset_config[name]
            ):
                continue
            old_fields = _business_fields(old_config)
            current_fields = _business_fields(dataset_config[name])
            deleted = sorted(old_fields - current_fields)
            if deleted:
                remove_fields[name] = deleted

        return {
            "full_rebuild": False,
            "install_dataset_config": install_dataset_config,
            "remove_datasets": remove_datasets,
            "remove_fields": remove_fields,
            "update_dataset_config": dataset_config,
            "summary": {
                "total_datasets": len(dataset_config),
                "new_datasets": sum(
                    1 for name in install_dataset_config if name not in previous
                ),
                "removed_datasets": len(remove_datasets),
                "new_fields": sum(
                    0
                    if is_sidecar_dataset(cfg)
                    else (
                        len(_business_fields(cfg))
                        if name not in previous
                        else len(
                            _business_fields(cfg)
                            - _business_fields(previous[name])
                        )
                    )
                    for name, cfg in install_dataset_config.items()
                ),
                "removed_fields": sum(len(v) for v in remove_fields.values()),
            },
        }

    @staticmethod
    def _with_token(configs: Dict, token: str) -> Dict:
        classes = discover_data_source_classes()
        result = {}
        for name, original in configs.items():
            config = original.copy()
            src_name = str(config.get("data_source") or "Tushare")
            cls = classes.get(src_name)
            if token and cls is not None and getattr(cls, "requires_token", False):
                config.setdefault("token", token)
            result[name] = config
        return result

    def _delete_removed(
        self,
        remove_datasets: Dict,
        remove_fields: Dict[str, List[str]],
        current_config: Dict,
    ) -> List[str]:
        logs: List[str] = []

        for name, old_config in remove_datasets.items():
            data_type = old_config.get("data_type")
            api_name = old_config.get("api_name")
            if is_industry_dataset(old_config):
                src = old_config.get("src", "SW2021")
                self.storage.clear_industry_data(src=src, api_name=api_name)
                logs.append(
                    f"已删除行业数据集 {name} 数据 (src={src}, api={api_name})"
                )
            elif data_type == "stock_info" and api_name == "stock_basic":
                self.storage.clear_stock_basic()
                logs.append(f"已清空 stock_basic ({name})")
            elif data_type == "financial":
                self.storage.clear_financial_indicator()
                logs.append(f"已清空 financial_indicator ({name})")
            elif data_type == "index_constituent":
                self.storage.clear_index_constituents(
                    old_config.get("index_list")
                )
                logs.append(f"已删除指数成分数据集 {name}")
            elif data_type == "index" or old_config.get("index_list"):
                # 行情指数行；勿误删 index_constituent 配置
                if data_type == "index":
                    symbols = [
                        _index_symbol(code)
                        for code in old_config.get("index_list", [])
                    ]
                    deleted = self.storage.delete_symbols(symbols)
                    logs.append(
                        f"已删除数据集 {name} 对应指数行情行: {deleted} 个标的"
                    )
            self.storage.unregister_dataset(name)
            logs.append(f"已从注册表移除数据集 {name}")

        still_used: Set[str] = set()
        for config in current_config.values():
            if is_sidecar_dataset(config):
                continue
            still_used |= _business_fields(config)

        unused_fields: Set[str] = set()
        for name, fields in remove_fields.items():
            cfg = current_config.get(name, {})
            if is_sidecar_dataset(cfg):
                continue
            unused_fields |= set(fields)
        unused_fields -= still_used

        if unused_fields:
            dropped = self.storage.drop_market_fields(unused_fields)
            if dropped:
                logs.append(f"已删除不再使用的字段列: {dropped}")

        return logs

    def execute_schedule(
        self,
        dataset_config: Dict,
        end_date: Optional[str] = None,
        tushare_token: Optional[str] = None,
        start_date: Optional[str] = None,
    ) -> Dict:
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        token = tushare_token or ""
        analysis = self.analyze_config(dataset_config)
        result = {
            "analysis": analysis,
            "execution_log": [],
            "update_results": {},
            "install_results": {},
            "delete_results": {},
            "success": True,
            "error_message": None,
        }

        try:
            installs = self._with_token(
                analysis["install_dataset_config"], token
            )
            full_rebuild = bool(analysis.get("full_rebuild"))
            # 与 StockDataUpdater.first_date_str 对齐；空库 sidecar 不得依赖空日历。
            schedule_start = (
                "".join(ch for ch in str(start_date) if ch.isdigit())[:8]
                if start_date
                else "20050101"
            )

            def _install_datasets(start_date=None, end_date_arg=None):
                if not installs:
                    return
                result["execution_log"].append(
                    f"开始 INSTALL，新增数据集/字段: {list(installs.keys())}"
                )
                installer = StockDataUpdater(
                    str(self.db_path), token, dataset_config=installs
                )
                failed = []
                for name, config in installs.items():
                    ok = installer.install_new_dataset(
                        name,
                        config,
                        start_date=start_date,
                        end_date=end_date_arg,
                    )
                    if not ok:
                        failed.append(name)
                if failed:
                    raise RuntimeError(f"以下数据集历史回填失败: {failed}")
                result["install_results"]["global_install"] = {
                    "status": "success",
                    "datasets": list(installs.keys()),
                }
                result["execution_log"].append("新增配置历史回填完成")

            if full_rebuild:
                # 空库：先建行情日历，再装 sidecar（显式传入本次调度区间）。
                market_updates = self._with_token(
                    {
                        name: config
                        for name, config in dataset_config.items()
                        if not is_sidecar_dataset(config)
                    },
                    token,
                )
                result["execution_log"].append(
                    f"空库先行情 UPDATE，数据集: {list(market_updates.keys())}"
                )
                updater = StockDataUpdater(
                    str(self.db_path), token, dataset_config=market_updates
                )
                if not updater.update_all(end_date, start_date=start_date):
                    raise RuntimeError("空库行情 UPDATE 失败")
                result["update_results"]["global_update"] = {"status": "success"}
                result["execution_log"].append("空库行情重建完成")
                _install_datasets(schedule_start, end_date)
            else:
                _install_datasets()

            if installs:
                # INSTALL 已经把维表写入 SQLite；注册表必须马上跟上，
                # 否则后面 UPDATE 失败会让下一轮把同一数据集再全量拉一遍。
                self._save_config_snapshot(dataset_config)
                self.storage.replace_registry(_sanitize_config(dataset_config))
                result["execution_log"].append("INSTALL 后已写入配置快照")

            delete_logs = self._delete_removed(
                analysis["remove_datasets"],
                analysis["remove_fields"],
                dataset_config,
            )
            if delete_logs:
                result["delete_results"]["logs"] = delete_logs
                result["execution_log"].extend(delete_logs)

            if not full_rebuild:
                updates = self._with_token(dataset_config, token)
                # 本轮刚 INSTALL 过的 always-full sidecar（如行业成分）无需再全量拉一次
                for name in list(installs):
                    cfg = updates.get(name)
                    if cfg is not None and (
                        always_full_sidecar(cfg)
                        or cfg.get("data_type") == "index_constituent"
                    ):
                        updates.pop(name)
                        result["execution_log"].append(
                            f"跳过 UPDATE {name}（本轮 INSTALL 已写入）"
                        )
                result["execution_log"].append(
                    f"开始 UPDATE，数据集: {list(updates.keys())}"
                )
                updater = StockDataUpdater(
                    str(self.db_path), token, dataset_config=updates
                )
                if not updater.update_all(end_date, start_date=start_date):
                    raise RuntimeError("增量 UPDATE 失败")
                result["update_results"]["global_update"] = {"status": "success"}
                result["execution_log"].append("增量更新完成")

            self._save_config_snapshot(dataset_config)
            self.storage.replace_registry(_sanitize_config(dataset_config))
            active_srcs = {
                config.get("src", "SW2021")
                for config in dataset_config.values()
                if is_industry_dataset(config)
            }
            self.storage.prune_industry_sources(active_srcs)
            active_index_codes = set()
            for config in dataset_config.values():
                if config.get("data_type") == "index_constituent":
                    active_index_codes.update(config.get("index_list") or [])
            self.storage.prune_index_constituents(active_index_codes)
            result["execution_log"].append("配置快照与注册表已更新")

        except Exception as exc:
            result["success"] = False
            result["error_message"] = str(exc)
            result["execution_log"].append(f"执行失败: {exc}")
            logger.error(f"调度执行失败: {exc}")
        return result

    def get_registry_info(self) -> Dict:
        datasets = self.storage.get_registry()
        return {"total_datasets": len(datasets), "datasets": datasets}

    def clear_registry(self):
        self.storage.clear_registry()
        if self.config_snapshot_path.exists():
            self.config_snapshot_path.unlink()
