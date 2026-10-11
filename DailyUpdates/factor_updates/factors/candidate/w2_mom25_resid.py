#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""w2_mom25 风格残差变体。

原始 w2_mom25 的 RankIC 为负、纯化 IC 为正，是因为因子 ≈ 风格投影 + 残差，
投影主要落在 ResVol↑ / Momentum↓ 上，把原始排序拉翻了。

本因子每日在全市场可交易截面上：
1. 把入库 style_* 做成与 ExposureEngine 一致的聚宽 z（含 NLSIZE）
2. 对截面 z(w2_mom25) 做截距 + 风格 OLS
3. 只保留残差 e = F − P_X F

不含行业哑变量（更新器侧不铺申万）；主暴露 ResVol/Momentum 已覆盖符号翻转。
不要交易风格投影 −P_X F；那是低波+动量收割，不是本因子。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from DailyUpdates.factor_updates.base_factor import BaseFactor
from DailyUpdates.factor_updates.factors.jq._panel import stock_frame, to_wide, wide_to_long
from DailyUpdates.storage.bin_storage import BinStorage, normalize_symbol
from FactorEvaluates.exposure_engine import (
    NLSIZE_NAME,
    RAW_STYLE_NAMES,
    SIZE_NAME,
    ExposureEngine,
    transform_style_arrays,
)
from yg_quant_repo import default_db_path

MIN_OBS = 40
SRC_NAME = "w2_mom25"
STYLE_DEPS: List[str] = list(RAW_STYLE_NAMES)


def _align_wide(panel: pd.DataFrame, dates: pd.Index, symbols: pd.Index) -> pd.DataFrame:
    work = panel.copy()
    work.index = pd.Index(
        [pd.Timestamp(idx).strftime("%Y-%m-%d") for idx in work.index],
        name="trade_date",
    )
    work.columns = [normalize_symbol(str(col)) for col in work.columns]
    work = work.loc[:, ~pd.Index(work.columns).duplicated()]
    return work.reindex(index=dates, columns=symbols)


def residualize_w2_on_styles(
    factor: np.ndarray,
    style_raw: Dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    min_obs: int = MIN_OBS,
) -> np.ndarray:
    """逐日：z(F) ~ 1 + 风格z + NLSIZE，返回残差。"""
    arr = np.asarray(factor, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if arr.shape != m.shape:
        raise ValueError(f"因子 {arr.shape} 与 mask {m.shape} 不一致")
    if SIZE_NAME not in style_raw:
        raise ValueError(f"残差化需要 {SIZE_NAME}")
    z = transform_style_arrays(style_raw, m)
    names = [n for n in RAW_STYLE_NAMES if n in z] + (
        [NLSIZE_NAME] if NLSIZE_NAME in z else []
    )
    if not names:
        raise ValueError("风格暴露为空")
    expo = np.stack([np.asarray(z[n], dtype=np.float64) for n in names], axis=2)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    k = int(expo.shape[2])
    floor = max(int(min_obs), k + 8)
    for t in range(arr.shape[0]):
        uni = m[t] & np.isfinite(arr[t])
        if int(uni.sum()) < floor:
            continue
        y = np.where(uni, arr[t], np.nan)[:, None]
        cols = [np.where(uni, 1.0, np.nan)]
        for j in range(k):
            cols.append(np.where(uni, expo[t, :, j], np.nan))
        x = np.column_stack(cols)
        _beta, resid = ExposureEngine.regress_matrix_day(y, x, min_obs=floor)
        out[t] = resid[:, 0]
    return out


class W2Mom25ResidFactor(BaseFactor):
    name = "w2_mom25_resid"
    description = (
        "w2_mom25 对聚宽风格+NLSIZE 的截面 OLS 残差："
        "剥掉 ResVol/Momentum 等投影，保留纯化 IC 为正的那一段"
    )
    dependencies: List[str] = ["close"]
    factor_dependencies: List[str] = [SRC_NAME, *STYLE_DEPS]
    role = "alpha"
    stage = "candidate"
    lookback_days = 60

    def __init__(self):
        super().__init__()
        self._store: Optional[BinStorage] = None

    def _bin_store(self) -> BinStorage:
        if self._store is None:
            factor_dir = os.getenv("YG_QUANT_FACTOR_DIR") or str(
                Path(default_db_path()).parent / "factors"
            )
            self._store = BinStorage(factor_dir)
        return self._store

    def calculate(
        self, data: pd.DataFrame, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame(columns=["ts_code", "trade_date", "factor_value"])
        if "close" not in data.columns:
            raise ValueError("w2_mom25_resid 需要行情字段 close")

        close = to_wide(stock_frame(data), "close")
        close.columns = [normalize_symbol(str(col)) for col in close.columns]
        close = close.loc[:, ~pd.Index(close.columns).duplicated()].sort_index()
        if close.empty:
            return pd.DataFrame(columns=["ts_code", "trade_date", "factor_value"])

        dates = pd.Index(list(close.index), name="trade_date")
        symbols = pd.Index(list(close.columns), name="ts_code")
        store = self._bin_store()
        need = [SRC_NAME, *STYLE_DEPS]
        panels = store.read_factor_panels(
            need,
            start_date=str(dates[0]),
            end_date=str(dates[-1]),
            symbols=list(symbols),
        )
        missing = [n for n in need if panels.get(n) is None or panels[n].empty]
        if missing:
            raise ValueError(
                "w2_mom25_resid 缺少已入库子因子，请先更新: " + ", ".join(missing)
            )

        close_arr = close.reindex(index=dates, columns=symbols).to_numpy(
            dtype=np.float64, copy=False
        )
        mask = np.isfinite(close_arr) & (close_arr > 0)
        src = _align_wide(panels[SRC_NAME], dates, symbols).to_numpy(
            dtype=np.float64, copy=False
        )
        style_raw: Dict[str, np.ndarray] = {}
        for name in STYLE_DEPS:
            style_raw[name] = _align_wide(panels[name], dates, symbols).to_numpy(
                dtype=np.float64, copy=False
            )
        # 有收盘且有 style_size 才进当日回归
        size = style_raw[SIZE_NAME]
        mask = mask & np.isfinite(size)

        resid = residualize_w2_on_styles(src, style_raw, mask)
        wide = pd.DataFrame(resid, index=dates, columns=symbols)
        self._get_logger().info(
            "w2_mom25_resid 完成 %s 日 × %s 只，有效格子 %s",
            wide.shape[0],
            wide.shape[1],
            int(np.isfinite(resid).sum()),
        )
        return wide_to_long(wide, start_date=start_date)
