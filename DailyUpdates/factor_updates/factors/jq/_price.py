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
def _rolling_extreme_age(arr: np.ndarray, window: int, find_max: bool) -> np.ndarray:
    t_n, n_col = arr.shape
    out = np.full((t_n, n_col), np.nan)
    for j in range(n_col):
        col = arr[:, j]
        for t in range(window - 1, t_n):
            best = -np.inf if find_max else np.inf
            age = np.nan
            ok = True
            for k in range(window):
                v = col[t - window + 1 + k]
                if not np.isfinite(v):
                    ok = False
                    break
                take = v >= best if find_max else v <= best
                if take:
                    best = v
                    age = window - 1 - k
            if ok:
                out[t, j] = age
    return out


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
        self.pre_close = (
            panel.to_wide(frame, "pre_close") if "pre_close" in frame.columns else None
        )
        self.up_limit = (
            panel.to_wide(frame, "up_limit") if "up_limit" in frame.columns else None
        )
        self.down_limit = (
            panel.to_wide(frame, "down_limit") if "down_limit" in frame.columns else None
        )
        self.circ_mv = (
            panel.to_wide(frame, "circ_mv") if "circ_mv" in frame.columns else None
        )
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

    def _empty(self) -> pd.DataFrame:
        return self.close * np.nan

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

    def _vol5(self) -> pd.DataFrame:
        return self._ma(self.turn, 5)

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

    def _atr5(self) -> pd.DataFrame:
        return self._ma(self.tr, 5)

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

    def _variance5(self) -> pd.DataFrame:
        return self._var(self.ret, 5) * 252.0

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

    def _mac10(self) -> pd.DataFrame:
        return self._ma(self.close, 10) / self.close

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

    def _bias10(self) -> pd.DataFrame:
        return self._bias(10)

    def _bias20(self) -> pd.DataFrame:
        return self._bias(20)

    def _roc(self, window: int) -> pd.DataFrame:
        lagged = self.close.shift(window)
        return 100.0 * (self.close - lagged) / lagged

    def _roc5(self) -> pd.DataFrame:
        return self._roc(5)

    def _roc20(self) -> pd.DataFrame:
        return self._roc(20)

    def _roc60(self) -> pd.DataFrame:
        return self._roc(60)

    def _cci(self, window: int) -> pd.DataFrame:
        typ = (self.high + self.low + self.close) / 3.0
        ma = self._ma(typ, window)
        mad = self._wrap(_rolling_mad(typ.to_numpy(dtype=np.float64), window))
        return (typ - ma) / (0.015 * mad)

    def _cci10(self) -> pd.DataFrame:
        return self._cci(10)

    def _cci15(self) -> pd.DataFrame:
        return self._cci(15)

    def _cci20(self) -> pd.DataFrame:
        return self._cci(20)

    def _cci88(self) -> pd.DataFrame:
        return self._cci(88)

    def _price1m(self) -> pd.DataFrame:
        return self.close / self._ma(self.close, 21) - 1.0

    def _price3m(self) -> pd.DataFrame:
        return self.close / self._ma(self.close, 61) - 1.0

    def _price1y(self) -> pd.DataFrame:
        return self.close / self._ma(self.close, 250) - 1.0

    def _plrc(self, n: int) -> pd.DataFrame:
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

    def _plrc6(self) -> pd.DataFrame:
        return self._plrc(6)

    def _plrc12(self) -> pd.DataFrame:
        return self._plrc(12)

    def _plrc24(self) -> pd.DataFrame:
        return self._plrc(24)

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

    def _overnight(self) -> pd.DataFrame:
        if self.pre_close is None:
            return self._empty()
        return self.open / self.pre_close - 1.0

    def _intraday(self) -> pd.DataFrame:
        return self.close / self.open - 1.0

    def _ret1(self) -> pd.DataFrame:
        return self.ret

    def _ret5(self) -> pd.DataFrame:
        lagged = self.close.shift(5)
        return self.close / lagged - 1.0

    def _ret10(self) -> pd.DataFrame:
        lagged = self.close.shift(10)
        return self.close / lagged - 1.0

    def _amihud(self, window: int) -> pd.DataFrame:
        illiq = self.ret.abs() / self.amount
        illiq = illiq.where(self.amount > 0)
        return illiq.rolling(window, min_periods=window).mean()

    def _amihud5(self) -> pd.DataFrame:
        return self._amihud(5)

    def _amihud20(self) -> pd.DataFrame:
        return self._amihud(20)

    def _davol5(self) -> pd.DataFrame:
        return self._ma(self.turn, 5) / self._ma(self.turn, 20)

    def _range5(self) -> pd.DataFrame:
        rng = (self.high - self.low) / self.close
        return rng.rolling(5, min_periods=5).mean()

    def _efficiency(self) -> pd.DataFrame:
        span = self.high - self.low
        return (self.close - self.open).abs() / span.where(span > 0)

    def _upper_shadow(self) -> pd.DataFrame:
        top = np.maximum(self.open, self.close)
        return (self.high - top) / self.close

    def _lower_shadow(self) -> pd.DataFrame:
        bot = np.minimum(self.open, self.close)
        return (bot - self.low) / self.close

    def _close_loc(self) -> pd.DataFrame:
        span = self.high - self.low
        return (self.close - self.low) / span.where(span > 0)

    def _limit_up_dist(self) -> pd.DataFrame:
        if self.up_limit is None or self.pre_close is None:
            return self._empty()
        return (self.up_limit - self.close) / self.pre_close

    def _limit_dn_dist(self) -> pd.DataFrame:
        if self.down_limit is None or self.pre_close is None:
            return self._empty()
        return (self.close - self.down_limit) / self.pre_close

    def _maxret20(self) -> pd.DataFrame:
        return self.ret.rolling(20, min_periods=20).max()

    def _minret20(self) -> pd.DataFrame:
        return self.ret.rolling(20, min_periods=20).min()

    def _corr_pv5(self) -> pd.DataFrame:
        return self.ret.rolling(5, min_periods=5).corr(self.vol)

    def _corr_pv20(self) -> pd.DataFrame:
        return self.ret.rolling(20, min_periods=20).corr(self.vol)

    def _vwap_dev(self) -> pd.DataFrame:
        vwap = self.amount / self.vol
        return (self.close / vwap - 1.0).where((self.vol > 0) & (self.amount > 0))

    def _rsi(self, window: int) -> pd.DataFrame:
        delta = self.close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        alpha = 1.0 / float(window)
        au = up.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
        ad = down.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
        return 100.0 - 100.0 / (1.0 + au / ad)

    def _rsi6(self) -> pd.DataFrame:
        return self._rsi(6)

    def _rsi14(self) -> pd.DataFrame:
        return self._rsi(14)

    def _willr14(self) -> pd.DataFrame:
        hh = self.high.rolling(14, min_periods=14).max()
        ll = self.low.rolling(14, min_periods=14).min()
        span = hh - ll
        return -100.0 * (hh - self.close) / span.where(span > 0)

    def _down_vol_ratio20(self) -> pd.DataFrame:
        down = self.close < self.close.shift(1)
        dv = self.vol.where(down, 0.0).rolling(20, min_periods=20).sum()
        tv = self.vol.rolling(20, min_periods=20).sum()
        return dv / tv.where(tv > 0)

    def _amt_to_mv(self) -> pd.DataFrame:
        if self.circ_mv is None:
            return self._empty()
        return (self.amount / self.circ_mv).where(self.circ_mv > 0)

    def _jump5(self) -> pd.DataFrame:
        return self._overnight().abs().rolling(5, min_periods=5).mean()

    def _vol10(self) -> pd.DataFrame:
        return self._ma(self.turn, 10)

    def _vol120(self) -> pd.DataFrame:
        return self._ma(self.turn, 120)

    def _vol240(self) -> pd.DataFrame:
        return self._ma(self.turn, 240)

    def _davol5_120(self) -> pd.DataFrame:
        return self._ma(self.turn, 5) / self._ma(self.turn, 120)

    def _davol10(self) -> pd.DataFrame:
        return self._ma(self.turn, 10) / self._ma(self.turn, 120)

    def _tvma6(self) -> pd.DataFrame:
        return self._ma(self.amount, 6)

    def _tvma20(self) -> pd.DataFrame:
        return self._ma(self.amount, 20)

    def _tvstd6(self) -> pd.DataFrame:
        return self._std(self.amount, 6)

    def _tvstd20(self) -> pd.DataFrame:
        return self._std(self.amount, 20)

    def _vema5(self) -> pd.DataFrame:
        return self._ema(self.vol, 5)

    def _vema10(self) -> pd.DataFrame:
        return self._ema(self.vol, 10)

    def _vema12(self) -> pd.DataFrame:
        return self._ema(self.vol, 12)

    def _vema26(self) -> pd.DataFrame:
        return self._ema(self.vol, 26)

    def _vdiff(self) -> pd.DataFrame:
        return self._ema(self.vol, 12) - self._ema(self.vol, 26)

    def _vdea(self) -> pd.DataFrame:
        return self._ema(self._vdiff(), 9)

    def _vmacd(self) -> pd.DataFrame:
        return self._vdiff() - self._vdea()

    def _vosc(self) -> pd.DataFrame:
        fast = self._ema(self.vol, 12)
        slow = self._ema(self.vol, 26)
        return 100.0 * (fast - slow) / fast

    def _mawvad(self) -> pd.DataFrame:
        return self._ma(self._wvad(), 6)

    def _vstd10(self) -> pd.DataFrame:
        return self._std(self.vol, 10)

    def _vroc6(self) -> pd.DataFrame:
        lagged = self.vol.shift(6)
        return 100.0 * (self.vol - lagged) / lagged

    def _atr6(self) -> pd.DataFrame:
        return self._ma(self.tr, 6)

    def _kurtosis60(self) -> pd.DataFrame:
        return self.ret.rolling(60, min_periods=60).kurt()

    def _variance120(self) -> pd.DataFrame:
        return self._var(self.ret, 120) * 252.0

    def _skewness120(self) -> pd.DataFrame:
        return self.ret.rolling(120, min_periods=120).skew()

    def _kurtosis120(self) -> pd.DataFrame:
        return self.ret.rolling(120, min_periods=120).kurt()

    def _sharpe120(self) -> pd.DataFrame:
        return self._sharpe(120)

    def _ema5(self) -> pd.DataFrame:
        return self._ema(self.close, 5) / self.close

    def _emac10(self) -> pd.DataFrame:
        return self._ema(self.close, 10) / self.close

    def _emac20(self) -> pd.DataFrame:
        return self._ema(self.close, 20) / self.close

    def _emac120(self) -> pd.DataFrame:
        return self._ema(self.close, 120) / self.close

    def _mac120(self) -> pd.DataFrame:
        return self._ma(self.close, 120) / self.close

    def _bias60(self) -> pd.DataFrame:
        return self._bias(60)

    def _roc6(self) -> pd.DataFrame:
        return self._roc(6)

    def _roc12(self) -> pd.DataFrame:
        return self._roc(12)

    def _roc120(self) -> pd.DataFrame:
        return self._roc(120)

    def _trix5(self) -> pd.DataFrame:
        mtr = self._ema(self._ema(self._ema(self.close, 5), 5), 5)
        prev = mtr.shift(1)
        return 100.0 * (mtr - prev) / prev

    def _aroon(self, find_max: bool, window: int = 25) -> pd.DataFrame:
        src = self.high if find_max else self.low
        age = self._wrap(
            _rolling_extreme_age(src.to_numpy(dtype=np.float64), window, find_max)
        )
        return 100.0 * (window - age) / window

    def _arron_up_25(self) -> pd.DataFrame:
        return self._aroon(True, 25)

    def _arron_down_25(self) -> pd.DataFrame:
        return self._aroon(False, 25)

    def _cr20(self) -> pd.DataFrame:
        mid = (self.high.shift(1) + self.low.shift(1)) / 2.0
        up = (self.high - mid).clip(lower=0.0)
        down = (mid - self.low).clip(lower=0.0)
        return 100.0 * up.rolling(20, min_periods=20).sum() / down.rolling(
            20, min_periods=20
        ).sum()

    def _mass(self) -> pd.DataFrame:
        hl = self.high - self.low
        ema1 = self._ema(hl, 9)
        ema2 = self._ema(ema1, 9)
        return (ema1 / ema2).rolling(25, min_periods=25).sum()

    def _rank1m(self) -> pd.DataFrame:
        ret20 = self.close / self.close.shift(20) - 1.0
        rank = ret20.rank(axis=1, method="min", ascending=False)
        count = ret20.notna().sum(axis=1)
        return 1.0 - rank.div(count.replace(0, np.nan), axis=0)

    def _single_day_vpt(self) -> pd.DataFrame:
        return self.ret * self.vol

    def _single_day_vpt_6(self) -> pd.DataFrame:
        return self._ma(self._single_day_vpt(), 6)

    def _single_day_vpt_12(self) -> pd.DataFrame:
        return self._ma(self._single_day_vpt(), 12)


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
