#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""共享面板：open、远期收益、股票池、风格原料，只加载一次。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from Universes.catalog import build_mask as build_universe_mask

from ..exposure_engine import RAW_STYLE_NAMES, ExposureEngine
from ..factor_panel_loader import FactorPanelLoader
from ..industry_panel import load_l1_code_panel
from ..market_panel_loader import MarketPanelLoader
from ..return_calculator import ReturnCalculator
from ..metrics.extra_ic_metrics import DECAY_HORIZONS


class SharedPanelStore:
    def __init__(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        universe: str = "all",
        horizon: int = 5,
        decay_horizons: Sequence[int] = DECAY_HORIZONS,
        load_style: bool = True,
        load_xt: bool = False,
        factor_loader: Optional[FactorPanelLoader] = None,
        market_loader: Optional[MarketPanelLoader] = None,
        progress=None,
    ):
        self.factor_loader = factor_loader or FactorPanelLoader()
        self.market_loader = market_loader or MarketPanelLoader()
        self.universe_name = universe
        self.horizon = int(horizon)
        self._say = progress or (lambda *_a, **_k: None)
        self._say("phase", "load_open", 0.02, "加载 open 面板")
        full_open = self.market_loader.load_open()
        if full_open.empty:
            raise ValueError("open 面板为空")
        full_open = full_open.copy()
        full_open.index = pd.to_datetime(full_open.index).strftime("%Y-%m-%d")
        # 标签需要评价窗之后的开盘。先在完整行情上算远期收益，再按 start/end 切评价日。
        self._say("phase", "fwd_ret", 0.08, "计算远期收益")
        calc = ReturnCalculator(full_open, lru_size=max(12, len(decay_horizons) + 1))
        wanted = {int(horizon), *[int(n) for n in decay_horizons]}
        eval_open = full_open
        if start:
            eval_open = eval_open.loc[eval_open.index >= start]
        if end:
            eval_open = eval_open.loc[eval_open.index <= end]
        if eval_open.empty:
            raise ValueError("日期切片后 open 面板为空")
        self.dates = list(eval_open.index)
        self.symbols = [str(c) for c in eval_open.columns]
        self.open = eval_open.to_numpy(dtype=np.float32)
        self.fwd: Dict[int, np.ndarray] = {}
        for n in sorted(wanted):
            panel = calc.get(n).reindex(index=eval_open.index, columns=eval_open.columns)
            self.fwd[n] = panel.to_numpy(dtype=np.float32)
        self._say("phase", "universe", 0.12, "股票池 mask")
        mask_full = build_universe_mask(
            universe,
            full_open.index,
            full_open.columns,
            full_open,
            self.market_loader.storage,
        )
        mask = mask_full.reindex(index=eval_open.index, columns=eval_open.columns)
        self.mask = mask.fillna(False).to_numpy(dtype=bool)
        self.style: Dict[str, np.ndarray] = {}
        self.x_t: Optional[np.ndarray] = None
        self.x_names: List[str] = []
        if load_style:
            self._say("phase", "style", 0.18, "读取风格 Bin")
            loaded = self.factor_loader.load_many(
                RAW_STYLE_NAMES, start_date=self.dates[0], end_date=self.dates[-1]
            )
            for name, panel in loaded.items():
                if panel is None or panel.empty:
                    continue
                aligned = panel.copy()
                aligned.index = pd.to_datetime(aligned.index).strftime("%Y-%m-%d")
                aligned = aligned.reindex(index=eval_open.index, columns=eval_open.columns)
                self.style[name] = aligned.to_numpy(dtype=np.float32)
            if load_xt and "style_size" in self.style:
                self._say("phase", "x_t", 0.22, "组装 X_T")
                raw_frames = {
                    name: pd.DataFrame(arr, index=eval_open.index, columns=eval_open.columns)
                    for name, arr in self.style.items()
                }
                codes, labels = load_l1_code_panel(
                    self.market_loader.storage, eval_open.index, eval_open.columns
                )
                exposures = ExposureEngine().build(
                    raw_frames, mask, industry_codes=codes, industry_labels=labels
                )
                names = list(exposures.names)
                stack = [np.ones((len(self.dates), len(self.symbols)), dtype=np.float32)]
                x_names = ["intercept"]
                for name in names:
                    panel = exposures.panels.get(name)
                    if panel is None:
                        continue
                    stack.append(
                        panel.reindex(index=eval_open.index, columns=eval_open.columns).to_numpy(
                            dtype=np.float32
                        )
                    )
                    x_names.append(name)
                self.x_t = np.stack(stack, axis=2)
                self.x_names = x_names

    def barra_day(self, date_i: int, row_mask: np.ndarray) -> Optional[np.ndarray]:
        if not self.style:
            return None
        cols = []
        for name in RAW_STYLE_NAMES:
            arr = self.style.get(name)
            if arr is None:
                cols.append(np.full(int(row_mask.sum()), np.nan, dtype=np.float64))
            else:
                cols.append(arr[date_i, row_mask].astype(np.float64, copy=False))
        return np.column_stack(cols)

    def size_day(self, date_i: int, row_mask: np.ndarray) -> Optional[np.ndarray]:
        arr = self.style.get("style_size")
        if arr is None:
            return None
        return arr[date_i, row_mask].astype(np.float64, copy=False)
