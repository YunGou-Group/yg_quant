#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8 腿 LightGBM 合成大因子，口径对齐 Strategies.lgbm。

腿：technical_MAC60, basics_sales_to_price_ratio, -emotion_money_flow_20,
alpha042, chip_low_deposit, alpha074, alpha055, alpha080。

每个交易日历上每隔 horizon（默认 20）日用 LambdaRank + blend 打一次分，
再与上期分数 smooth，中间日向前填充，便于评估日频 IC。
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
from Strategies._factor_combo import parse_factor_list
from Strategies.lgbm import (
    DEFAULT_BLEND,
    DEFAULT_HORIZON,
    DEFAULT_LOOKBACK,
    DEFAULT_SMOOTH,
    combine_scores_lgbm,
)
from Strategies.rebalance_schedule import every_n_asofs
from yg_quant_repo import default_db_path

LEGS_SPEC = (
    "technical_MAC60,basics_sales_to_price_ratio,-emotion_money_flow_20,"
    "alpha042,chip_low_deposit,alpha074,alpha055,alpha080"
)


def _align_wide(panel: pd.DataFrame, dates: pd.Index, symbols: pd.Index) -> pd.DataFrame:
    work = panel.copy()
    work.index = pd.Index(
        [pd.Timestamp(idx).strftime("%Y-%m-%d") for idx in work.index],
        name="trade_date",
    )
    work.columns = [normalize_symbol(str(col)) for col in work.columns]
    work = work.loc[:, ~pd.Index(work.columns).duplicated()]
    return work.reindex(index=dates, columns=symbols)


class LgbmCombo8Factor(BaseFactor):
    name = "lgbm_combo_8"
    description = (
        "LGBM 合成：MAC60 + S/P - MF20 + alpha042 + chip_low_deposit "
        "+ alpha074/055/080；LambdaRank blend=0.45 smooth=0.40，每 20 日重拟合"
    )
    dependencies: List[str] = ["close"]
    factor_dependencies: List[str] = [name for name, _sign in parse_factor_list(LEGS_SPEC)]
    role = "alpha"
    stage = "candidate"
    lookback_days = 520

    def __init__(self):
        super().__init__()
        self.legs = parse_factor_list(LEGS_SPEC)
        self.lookback = DEFAULT_LOOKBACK
        self.horizon = DEFAULT_HORIZON
        self.blend = DEFAULT_BLEND
        self.smooth = DEFAULT_SMOOTH
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
            raise ValueError("lgbm_combo_8 需要行情字段 close")

        close = to_wide(stock_frame(data), "close")
        close.columns = [normalize_symbol(str(col)) for col in close.columns]
        close = close.loc[:, ~pd.Index(close.columns).duplicated()].sort_index()
        if close.empty:
            return pd.DataFrame(columns=["ts_code", "trade_date", "factor_value"])

        dates = pd.Index(list(close.index), name="trade_date")
        symbols = pd.Index(list(close.columns), name="ts_code")
        store = self._bin_store()
        names = [name for name, _ in self.legs]
        panels: Dict[str, pd.DataFrame] = store.read_factor_panels(
            names,
            start_date=str(dates[0]),
            end_date=str(dates[-1]),
            symbols=list(symbols),
        )
        missing = [name for name in names if panels.get(name) is None or panels[name].empty]
        if missing:
            raise ValueError(
                "lgbm_combo_8 缺少已入库子因子，请先更新: " + ", ".join(missing)
            )

        close_arr = close.reindex(index=dates, columns=symbols).to_numpy(
            dtype=np.float64, copy=False
        )
        mask = np.isfinite(close_arr)
        factor_arrs = [
            _align_wide(panels[name], dates, symbols).to_numpy(dtype=np.float64, copy=False)
            for name in names
        ]
        signs = [sign for _, sign in self.legs]
        need = int(self.lookback) + int(self.horizon)
        n_dates, n_sym = close_arr.shape
        if n_dates < need:
            self._get_logger().warning(
                "lgbm_combo_8 行情不足 lookback+horizon=%s（实际 %s 日）",
                need,
                n_dates,
            )
            return pd.DataFrame(columns=["ts_code", "trade_date", "factor_value"])

        calendar = store.read_calendar() or [str(d) for d in dates]
        score_dates = every_n_asofs(calendar, int(self.horizon))
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d") if start_date else None

        last_score: Optional[np.ndarray] = None
        have_prev = False
        if start:
            prev_end = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            prev = store.read_factor_panel(self.name, end_date=prev_end, symbols=list(symbols))
            if prev is not None and not prev.empty:
                last_score = (
                    _align_wide(prev, prev.index, symbols).iloc[-1].to_numpy(dtype=np.float64)
                )
                have_prev = True

        out = np.full((n_dates, n_sym), np.nan, dtype=np.float64)
        n_fit = 0
        log = self._get_logger()
        for t in range(need - 1, n_dates):
            date = str(dates[t])
            on_grid = date in score_dates
            should_fit = on_grid and (not have_prev or start is None or date >= start)
            if should_fit:
                sl = slice(t - need + 1, t + 1)
                raw, _model = combine_scores_lgbm(
                    [arr[sl] for arr in factor_arrs],
                    signs,
                    close_arr[sl],
                    mask[sl],
                    lookback=self.lookback,
                    horizon=self.horizon,
                    blend=self.blend,
                )
                if (
                    last_score is not None
                    and last_score.shape == raw.shape
                    and self.smooth > 0
                ):
                    both = np.isfinite(raw) & np.isfinite(last_score)
                    mixed = raw.copy()
                    mixed[both] = (1.0 - self.smooth) * raw[both] + self.smooth * last_score[both]
                    raw = mixed
                last_score = np.asarray(raw, dtype=np.float64)
                n_fit += 1
                if n_fit == 1 or n_fit % 10 == 0:
                    log.info("lgbm_combo_8 拟合 %s 次 @ %s", n_fit, date)
            if start and date < start:
                continue
            if last_score is not None:
                out[t] = last_score

        wide = pd.DataFrame(out, index=dates, columns=symbols)
        log.info("lgbm_combo_8 本段拟合 %s 次", n_fit)
        return wide_to_long(wide, start_date=start)
