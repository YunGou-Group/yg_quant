#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽情绪 / 风险 / 技术 / 动量：共享宽表，按因子名现算一列。"""

from __future__ import annotations

import threading
from typing import Dict, Optional

import numpy as np
import pandas as pd
from numba import njit

from . import _panel as panel

_CACHE: Dict[int, "PriceEngine"] = {}
_LOCK = threading.Lock()
RF = 0.04


@njit
def _rolling_mad(arr: np.ndarray, window: int) -> np.ndarray:
    t_n, n_col = arr.shape
    out = np.full((t_n, n_col), np.nan)
    for j in range(n_col):
        col = arr[:, j]
        for t in range(window - 1, t_n):
            acc = 0.0
            ok = True
            for k in range(window):
                v = col[t - window + 1 + k]
                if not np.isfinite(v):
                    ok = False
                    break
                acc += v
            if not ok:
                continue
            mean = acc / window
            mad = 0.0
            for k in range(window):
                mad += abs(col[t - window + 1 + k] - mean)
            out[t, j] = mad / window
    return out


class PriceEngine:
    def __init__(self, data: pd.DataFrame):
        frame = panel.stock_frame(data)
        self.close = panel.to_wide(frame, "close")
        self.open = panel.to_wide(frame, "open") if "open" in frame.columns else self.close
        self.high = panel.to_wide(frame, "high") if "high" in frame.columns else self.close
        self.low = panel.to_wide(frame, "low") if "low" in frame.columns else self.close
        self.vol = panel.to_wide(frame, "vol") if "vol" in frame.columns else self.close * np.nan
        if "amount" in frame.columns:
            self.amount = panel.to_wide(frame, "amount")
        else:
            self.amount = ((self.high + self.low + self.close) / 3.0) * self.vol
        turn = None
        if "turnover_rate" in frame.columns:
            turn = panel.to_wide(frame, "turnover_rate")
        if "turnover_rate_f" in frame.columns:
            raw_f = panel.to_wide(frame, "turnover_rate_f")
            turn = raw_f if turn is None else turn.fillna(raw_f)
        self.turn = turn if turn is not None else self.vol * np.nan
        self.ret = self.close.pct_change()
        prev = self.close.shift(1)
        self.tr = self._wrap(
            np.nanmax(
                np.stack(
                    [
                        (self.high - self.low).to_numpy(dtype=np.float64),
                        (self.high - prev).abs().to_numpy(dtype=np.float64),
                        (self.low - prev).abs().to_numpy(dtype=np.float64),
                    ],
                    axis=0,
                ),
                axis=0,
            )
        )

    def _wrap(self, arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=self.close.index, columns=self.close.columns)

    def _ema(self, wide: pd.DataFrame, span: int) -> pd.DataFrame:
        return wide.ewm(span=span, adjust=False, min_periods=span).mean()

    def _ma(self, wide: pd.DataFrame, window: int) -> pd.DataFrame:
        return wide.rolling(window, min_periods=window).mean()

    def _std(self, wide: pd.DataFrame, window: int) -> pd.DataFrame:
        return wide.rolling(window, min_periods=window).std(ddof=0)

    def _var(self, wide: pd.DataFrame, window: int) -> pd.DataFrame:
        return wide.rolling(window, min_periods=window).var(ddof=0)

    def compute(self, key: str) -> pd.DataFrame:
        fn = getattr(self, f"_{key}", None)
        if fn is None:
            raise KeyError(f"未知价量因子: {key}")
        return fn()

    def _vol20(self) -> pd.DataFrame:
        return self._ma(self.turn, 20)

    def _vol60(self) -> pd.DataFrame:
        return self._ma(self.turn, 60)

    def _davol20(self) -> pd.DataFrame:
        return self._ma(self.turn, 20) / self._ma(self.turn, 120)

    def _vr(self) -> pd.DataFrame:
        up = self.close > self.close.shift(1)
        down = self.close < self.close.shift(1)
        flat = self.close == self.close.shift(1)
        avs = self.vol.where(up, 0.0).rolling(26, min_periods=26).sum()
        bvs = self.vol.where(down, 0.0).rolling(26, min_periods=26).sum()
        cvs = self.vol.where(flat, 0.0).rolling(26, min_periods=26).sum()
        return (avs + 0.5 * cvs) / (bvs + 0.5 * cvs)

    def _ar(self) -> pd.DataFrame:
        num = (self.high - self.open).rolling(26, min_periods=26).sum()
        den = (self.open - self.low).rolling(26, min_periods=26).sum()
        return 100.0 * num / den

    def _br(self) -> pd.DataFrame:
        prev = self.close.shift(1)
        num = (self.high - prev).rolling(26, min_periods=26).sum()
        den = (prev - self.low).rolling(26, min_periods=26).sum()
        return 100.0 * num / den

    def _arbr(self) -> pd.DataFrame:
        return self._ar() - self._br()

    def _vroc12(self) -> pd.DataFrame:
        lagged = self.vol.shift(12)
        return 100.0 * (self.vol - lagged) / lagged

    def _vstd20(self) -> pd.DataFrame:
        return self._std(self.vol, 20)

    def _atr14(self) -> pd.DataFrame:
        return self._ma(self.tr, 14)

    def _wvad(self) -> pd.DataFrame:
        span = self.high - self.low
        daily = (self.close - self.open) / span * self.vol
        daily = daily.where(span > 0)
        return daily.rolling(6, min_periods=6).sum()

    def _psy(self) -> pd.DataFrame:
        prev = self.close.shift(1)
        up = (self.close > prev).where(self.close.notna() & prev.notna())
        return 100.0 * up.rolling(12, min_periods=12).mean()

    def _turnover_volatility(self) -> pd.DataFrame:
        return self._std(self.turn, 20)

    def _money_flow_20(self) -> pd.DataFrame:
        # 文档只定义了当日典型价×量；因子名是 20 日资金流量，对当日值做 20 日求和。
        flow = (self.high + self.low + self.close) / 3.0 * self.vol
        return flow.rolling(20, min_periods=20).sum()

    def _variance20(self) -> pd.DataFrame:
        return self._var(self.ret, 20) * 252.0

    def _variance60(self) -> pd.DataFrame:
        return self._var(self.ret, 60) * 252.0

    def _skewness20(self) -> pd.DataFrame:
        return self.ret.rolling(20, min_periods=20).skew()

    def _skewness60(self) -> pd.DataFrame:
        return self.ret.rolling(60, min_periods=60).skew()

    def _kurtosis20(self) -> pd.DataFrame:
        return self.ret.rolling(20, min_periods=20).kurt()

    def _sharpe(self, window: int) -> pd.DataFrame:
        mu = self.ret.rolling(window, min_periods=window).mean() * 252.0
        sig = self.ret.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252.0)
        return (mu - RF) / sig

    def _sharpe20(self) -> pd.DataFrame:
        return self._sharpe(20)

    def _sharpe60(self) -> pd.DataFrame:
        return self._sharpe(60)

    def _mac5(self) -> pd.DataFrame:
        return self._ma(self.close, 5) / self.close

    def _mac20(self) -> pd.DataFrame:
        return self._ma(self.close, 20) / self.close

    def _mac60(self) -> pd.DataFrame:
        return self._ma(self.close, 60) / self.close

    def _emac12(self) -> pd.DataFrame:
        return self._ema(self.close, 12) / self.close

    def _emac26(self) -> pd.DataFrame:
        return self._ema(self.close, 26) / self.close

    def _macdc(self) -> pd.DataFrame:
        # 聚宽 MACD(SHORT=12, LONG=26, MID=9)：DIF，再除以收盘。MID 用于 DEA，不进该因子值。
        dif = self._ema(self.close, 12) - self._ema(self.close, 26)
        return dif / self.close

    def _boll_up(self) -> pd.DataFrame:
        mid = self._ma(self.close, 20)
        return (mid + 2.0 * self._std(self.close, 20)) / self.close

    def _boll_down(self) -> pd.DataFrame:
        mid = self._ma(self.close, 20)
        return (mid - 2.0 * self._std(self.close, 20)) / self.close

    def _mfi14(self) -> pd.DataFrame:
        tp = (self.high + self.low + self.close) / 3.0
        flow = tp * self.vol
        pos = flow.where(tp > tp.shift(1), 0.0)
        neg = flow.where(tp < tp.shift(1), 0.0)
        mr = pos.rolling(14, min_periods=14).sum() / neg.rolling(14, min_periods=14).sum()
        return 100.0 - 100.0 / (1.0 + mr)

    def _bias(self, window: int) -> pd.DataFrame:
        ma = self._ma(self.close, window)
        return 100.0 * (self.close - ma) / ma

    def _bias5(self) -> pd.DataFrame:
        return self._bias(5)

    def _bias20(self) -> pd.DataFrame:
        return self._bias(20)

    def _roc(self, window: int) -> pd.DataFrame:
        lagged = self.close.shift(window)
        return 100.0 * (self.close - lagged) / lagged

    def _roc20(self) -> pd.DataFrame:
        return self._roc(20)

    def _roc60(self) -> pd.DataFrame:
        return self._roc(60)

    def _cci20(self) -> pd.DataFrame:
        typ = (self.high + self.low + self.close) / 3.0
        ma = self._ma(typ, 20)
        mad = self._wrap(_rolling_mad(typ.to_numpy(dtype=np.float64), 20))
        return (typ - ma) / (0.015 * mad)

    def _price1m(self) -> pd.DataFrame:
        return self.close / self._ma(self.close, 21) - 1.0

    def _price1y(self) -> pd.DataFrame:
        return self.close / self._ma(self.close, 250) - 1.0

    def _plrc12(self) -> pd.DataFrame:
        n = 12
        mean_w = self._ma(self.close, n)
        sum_xy = 0.0
        for k in range(1, n + 1):
            sum_xy = sum_xy + k * self.close.shift(n - k)
        sum_xy = sum_xy / mean_w
        sum_y = self.close.rolling(n, min_periods=n).sum() / mean_w
        sum_x = n * (n + 1) / 2.0
        sum_x2 = n * (n + 1) * (2 * n + 1) / 6.0
        numer = n * sum_xy - sum_x * sum_y
        denom = n * sum_x2 - sum_x * sum_x
        return numer / denom

    def _trix10(self) -> pd.DataFrame:
        mtr = self._ema(self._ema(self._ema(self.close, 10), 10), 10)
        prev = mtr.shift(1)
        return 100.0 * (mtr - prev) / prev

    def _bbic(self) -> pd.DataFrame:
        bbi = (
            self._ma(self.close, 3)
            + self._ma(self.close, 6)
            + self._ma(self.close, 12)
            + self._ma(self.close, 24)
        ) / 4.0
        return bbi / self.close

    def _bull_power(self) -> pd.DataFrame:
        return (self.high - self._ema(self.close, 13)) / self.close

    def _bear_power(self) -> pd.DataFrame:
        return (self.low - self._ema(self.close, 13)) / self.close

    def _close_rank_1y(self) -> pd.DataFrame:
        n = 250
        roller = self.close.rolling(n, min_periods=n)
        rank_fn = getattr(roller, "rank", None)
        if callable(rank_fn):
            try:
                # 从大到小：最高价位置靠前，pct 后高价接近 1/n
                return rank_fn(pct=True, ascending=False)
            except TypeError:
                asc = rank_fn(pct=True)
                return 1.0 - asc
        hi = roller.max()
        lo = roller.min()
        return (hi - self.close) / (hi - lo)

    def _volume1m(self) -> pd.DataFrame:
        vol_mean = self.vol.shift(1).rolling(20, min_periods=20).mean()
        ret_mean = self.ret.shift(1).rolling(20, min_periods=20).mean()
        return (self.vol / vol_mean) * ret_mean


def get_price_engine(data: pd.DataFrame) -> PriceEngine:
    key = id(data)
    with _LOCK:
        engine = _CACHE.get(key)
        if engine is None:
            engine = PriceEngine(data)
            _CACHE.clear()
            _CACHE[key] = engine
        return engine


def extract_price(
    data: pd.DataFrame, key: str, start_date: Optional[str] = None
) -> pd.DataFrame:
    wide = get_price_engine(data).compute(key)
    return panel.wide_to_long(wide, start_date=start_date)
