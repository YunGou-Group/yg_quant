#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把已拉取的原始表标准化并合并成行情宽表。无 IO。"""

from typing import Dict, Mapping, Optional, Set

import numpy as np
import pandas as pd

from DailyUpdates.data_fetcher.preprocessing.market_bars import sanitize_market_bars


class DataProcessor:
    """标准化代码/日期，再按 (symbol, date) 合并并补齐缺失股票与字段。"""

    def __init__(self):
        self.namechange: Optional[pd.DataFrame] = None
        self.list_dates: Optional[pd.Series] = None
        self.calendar = None

    def set_limit_context(
        self,
        namechange: Optional[pd.DataFrame] = None,
        list_dates: Optional[pd.Series] = None,
        calendar=None,
    ) -> None:
        self.namechange = namechange
        self.list_dates = list_dates
        self.calendar = calendar

    def sanitize_frame(
        self, frame: pd.DataFrame, *, prev_adj: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        return sanitize_market_bars(
            frame,
            prev_adj=prev_adj,
            namechange=self.namechange,
            list_dates=self.list_dates,
            calendar=self.calendar,
        )

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        config: dict,
        *,
        prev_adj: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """统一股票代码、日期和字段格式，再按入库规则置空脏点。"""
        del dataset_name
        if df.empty:
            return df

        df = df.copy()

        if "ts_code" in df.columns:
            def convert_stock_code(ts_code):
                try:
                    if "." in ts_code:
                        parts = ts_code.split(".")
                        if len(parts) == 2:
                            return f"{parts[1]}{parts[0]}"
                    return ts_code
                except Exception:
                    return ts_code

            df["symbol"] = df["ts_code"].apply(convert_stock_code)
            if config.get("data_type") == "index":
                df["symbol"] = df["symbol"].apply(lambda x: f"index_{x}")

        if "trade_date" in df.columns:
            df["date"] = pd.to_datetime(df["trade_date"])

        keep_columns = []
        if "symbol" in df.columns:
            keep_columns.append("symbol")
        if "date" in df.columns:
            keep_columns.append("date")

        for field in config["fields"]:
            if field not in ["ts_code", "trade_date"] and field in df.columns:
                keep_columns.append(field)

        selected = df.loc[:, keep_columns]
        if not isinstance(selected, pd.DataFrame):
            selected = selected.to_frame()
        return self.sanitize_frame(selected, prev_adj=prev_adj)

    def merge_and_fill(
        self,
        dataset_dfs: Dict[str, pd.DataFrame],
        trade_date_str: str,
        current_stocks: set,
        all_fields: set,
    ) -> pd.DataFrame:
        """按 (symbol, date) 并集合并所有数据集，再补齐缺失股票与字段。"""
        if not dataset_dfs:
            return pd.DataFrame()

        stock_datasets = {}
        index_datasets = {}

        for dataset_name, df in dataset_dfs.items():
            if df.empty:
                print(f"数据集 {dataset_name} 为空，跳过合并")
                continue
            if _is_index_frame(df):
                index_datasets[dataset_name] = df
                print(f"数据集 {dataset_name} 识别为指数数据")
            else:
                stock_datasets[dataset_name] = df
                print(f"数据集 {dataset_name} 识别为股票数据")

        result_df = _outer_merge_frames(stock_datasets)
        if index_datasets:
            print("处理指数数据，先合并内部数据集...")
            merged_index_df = _outer_merge_frames(index_datasets)
            if merged_index_df is not None and not merged_index_df.empty:
                print(f"将合并后的指数数据追加到结果中，指数数据行数: {len(merged_index_df)}")
                result_df = pd.concat([result_df, merged_index_df], ignore_index=True)
                print(f"指数数据已追加，总结果行数: {len(result_df)}")

        if not result_df.empty and current_stocks:
            stock_col = "symbol" if "symbol" in result_df.columns else (
                "ts_code" if "ts_code" in result_df.columns else None
            )
            if stock_col:
                existing_stocks = set(result_df[stock_col].astype(str).unique())
                missing_stocks = current_stocks - existing_stocks
                if missing_stocks:
                    print(f"为缺失的 {len(missing_stocks)} 只股票填充NaN")
                    date_col = "date" if "date" in result_df.columns else (
                        "trade_date" if "trade_date" in result_df.columns else None
                    )
                    if date_col:
                        fill_dates = _panel_dates(result_df[date_col], trade_date_str, date_col)
                        missing_data = []
                        for stock_code in missing_stocks:
                            for day in fill_dates:
                                missing_row = {stock_col: stock_code, date_col: day}
                                for field in all_fields:
                                    if field not in {stock_col, date_col}:
                                        missing_row[field] = np.nan
                                missing_data.append(missing_row)
                        missing_df = pd.DataFrame(missing_data)
                        result_df = pd.concat([result_df, missing_df], ignore_index=True)
                        print(f"已为缺失股票填充NaN，总行数: {len(result_df)}")

        if not result_df.empty:
            for field in all_fields:
                if field not in result_df.columns:
                    print(f"为缺失字段 {field} 填充NaN")
                    result_df[field] = np.nan

        if not result_df.empty:
            sort_cols = []
            if "symbol" in result_df.columns:
                sort_cols.append("symbol")
            elif "ts_code" in result_df.columns:
                sort_cols.append("ts_code")
            if "date" in result_df.columns:
                sort_cols.append("date")
            elif "trade_date" in result_df.columns:
                sort_cols.append("trade_date")
            if sort_cols:
                result_df = result_df.sort_values(sort_cols).reset_index(drop=True)

        return result_df

    def build_panel(
        self,
        dataset_dfs: Mapping[str, pd.DataFrame],
        dataset_config: Mapping[str, dict],
        start_date: str,
        current_stocks: Set[str],
        all_fields: Set[str],
        *,
        prev_adj: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """先标准化每张已拉表，再合并并清洗成可写入的行情面板。"""
        if not dataset_dfs:
            return pd.DataFrame()
        normalized: Dict[str, pd.DataFrame] = {}
        for name, frame in dataset_dfs.items():
            config = dataset_config.get(name) or {}
            normalized[name] = self.normalize_dataframe(
                frame, name, config, prev_adj=prev_adj
            )
        result_df = self.merge_and_fill(normalized, start_date, current_stocks, all_fields)
        result_df = self.sanitize_frame(result_df, prev_adj=prev_adj)
        print(result_df)
        print(f"合并后获取到 {len(result_df)} 条记录，字段: {list(result_df.columns)}")
        return result_df


def _outer_merge_frames(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """按 (symbol, date) 做并集。配置顺序不影响留下哪些主键行。"""
    merged = None
    for dataset_name, frame in frames.items():
        if merged is None:
            merged = frame.copy()
            continue
        merge_keys = _merge_keys(merged, frame)
        if not merge_keys:
            print(f"警告：数据集 {dataset_name} 无法确定合并键，跳过合并")
            continue
        print(f"合并数据集 {dataset_name}，使用合并键: {merge_keys}")
        merged = pd.merge(
            merged,
            frame,
            on=merge_keys,
            how="outer",
            suffixes=("", f"_{dataset_name}"),
        )
        print(f"合并数据集 {dataset_name}，结果行数: {len(merged)}")
    return merged if merged is not None else pd.DataFrame()


def _is_index_frame(frame: pd.DataFrame) -> bool:
    if "symbol" not in frame.columns or frame.empty:
        return False
    return bool(frame["symbol"].astype(str).str.startswith("index_").all())


def _panel_dates(values: pd.Series, trade_date_str: str, date_col: str) -> list:
    """区间面板按已有交易日补停牌行；单日或没有日期时回退到本次窗口起点。"""
    existing = [value for value in values.dropna().unique().tolist()]
    if existing:
        return existing
    if date_col == "date":
        return [pd.to_datetime(trade_date_str)]
    return [trade_date_str]


def _merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list:
    keys = []
    if "symbol" in left.columns and "symbol" in right.columns:
        keys.append("symbol")
    elif "ts_code" in left.columns and "ts_code" in right.columns:
        keys.append("ts_code")
    if "date" in left.columns and "date" in right.columns:
        keys.append("date")
    elif "trade_date" in left.columns and "trade_date" in right.columns:
        keys.append("trade_date")
    return keys
