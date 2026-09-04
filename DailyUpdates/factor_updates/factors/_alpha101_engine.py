#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alpha101 engine (wide-panel, PIT industry).

Formulas follow Kakushadze, 101 Formulaic Alphas, 2016,
https://arxiv.org/abs/1601.00991

Repo adaptations:
- Input/output are long-format market panels (ts_code, trade_date, OHLCV...).
- Internally pivot to wide (date x symbol) so rolling is by stock and rank is cross-sectional.
- Module-level cache shares one full compute across thin BaseFactor wrappers.
- Hot operators (ts_rank / decay_linear / product / argmin/max) use Numba, not rolling.apply.
- Industry-neutral alphas use Shenwan L1/L2/L3 via IndNeutralize (PIT in_date/out_date).
- Paper windows that are not integers are rounded by ``_window``.
- alpha056 is enabled when total_mv is available.
- extract_alpha computes one factor at a time; wide panels are cached per market DataFrame.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from numba import njit, prange

logger = logging.getLogger("Alpha101.engine")

_DB_PATH: Optional[str] = None


def configure_db_path(db_path: Optional[str]) -> None:
    """Optional: FactorUpdater sets this so industry maps can be loaded."""
    global _DB_PATH
    _DB_PATH = str(db_path) if db_path else None


def _resolve_db_path() -> Optional[str]:
    if _DB_PATH:
        return _DB_PATH
    from yg_quant_repo import default_db_path

    path = default_db_path()
    return str(path) if path.is_file() else None

# ---------------------------------------------------------------------------
# Operators (wide panel: index=trade_date, columns=symbol)
# Slow path (ts_rank / decay_linear / product / argmin/max) uses Numba on
# contiguous float64 arrays; pandas rolling.apply is avoided.
# ---------------------------------------------------------------------------


def _window(window) -> int:
    return max(1, int(round(float(window))))


def _to_float2d(df):
    """Return (ndarray[T,N], index, columns, is_series)."""
    if isinstance(df, pd.Series):
        frame = df.to_frame("__s__")
        single = True
    else:
        frame = df
        single = False
    values = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
    return values, frame.index, frame.columns, single


def _from_float2d(values, index, columns, single):
    out = pd.DataFrame(values, index=index, columns=columns)
    if single:
        return out.iloc[:, 0]
    return out


def ts_sum(df, window=10):
    return df.rolling(_window(window), min_periods=1).sum()


def sma(df, window=10):
    return df.rolling(_window(window), min_periods=1).mean()


def stddev(df, window=10):
    return df.rolling(_window(window), min_periods=1).std()


def correlation(x, y, window=10):
    return x.rolling(_window(window)).corr(y)


def covariance(x, y, window=10):
    return x.rolling(_window(window)).cov(y)


@njit(cache=True, parallel=True)
def _ts_rank_2d(data: np.ndarray, window: int) -> np.ndarray:
    """1-based average-rank of the last observation in each trailing window."""
    t_count, n_count = data.shape
    out = np.empty((t_count, n_count), dtype=np.float64)
    for j in prange(n_count):
        for i in range(t_count):
            val = data[i, j]
            if np.isnan(val):
                out[i, j] = np.nan
                continue
            start = i - window + 1
            if start < 0:
                start = 0
            less = 0
            equal = 0
            valid = 0
            for k in range(start, i + 1):
                v = data[k, j]
                if np.isnan(v):
                    continue
                valid += 1
                if v < val:
                    less += 1
                elif v == val:
                    equal += 1
            if valid == 0:
                out[i, j] = np.nan
            else:
                # pandas rank(method='average') of the last value
                out[i, j] = less + (equal + 1) * 0.5
    return out


@njit(cache=True, parallel=True)
def _decay_linear_2d(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    window = weights.shape[0]
    t_count, n_count = data.shape
    out = np.empty((t_count, n_count), dtype=np.float64)
    for j in prange(n_count):
        for i in range(t_count):
            if i < window - 1:
                out[i, j] = np.nan
                continue
            acc = 0.0
            base = i - window + 1
            for k in range(window):
                acc += data[base + k, j] * weights[k]
            out[i, j] = acc
    return out


@njit(cache=True, parallel=True)
def _product_2d(data: np.ndarray, window: int) -> np.ndarray:
    t_count, n_count = data.shape
    out = np.empty((t_count, n_count), dtype=np.float64)
    for j in prange(n_count):
        for i in range(t_count):
            if i < window - 1:
                out[i, j] = np.nan
                continue
            acc = 1.0
            seen = False
            base = i - window + 1
            for k in range(window):
                v = data[base + k, j]
                if np.isnan(v):
                    continue
                acc *= v
                seen = True
            out[i, j] = acc if seen else np.nan
    return out


@njit(cache=True, parallel=True)
def _ts_argmax_2d(data: np.ndarray, window: int) -> np.ndarray:
    """1-based index of max within window. All-NaN windows stay NaN."""
    t_count, n_count = data.shape
    out = np.empty((t_count, n_count), dtype=np.float64)
    for j in prange(n_count):
        for i in range(t_count):
            if i < window - 1:
                out[i, j] = np.nan
                continue
            base = i - window + 1
            best = data[base, j]
            best_k = 0
            seen = not np.isnan(best)
            for k in range(1, window):
                v = data[base + k, j]
                if np.isnan(v):
                    continue
                if (not seen) or (v > best):
                    best = v
                    best_k = k
                    seen = True
            out[i, j] = (best_k + 1.0) if seen else np.nan
    return out


@njit(cache=True, parallel=True)
def _ts_argmin_2d(data: np.ndarray, window: int) -> np.ndarray:
    t_count, n_count = data.shape
    out = np.empty((t_count, n_count), dtype=np.float64)
    for j in prange(n_count):
        for i in range(t_count):
            if i < window - 1:
                out[i, j] = np.nan
                continue
            base = i - window + 1
            best = data[base, j]
            best_k = 0
            seen = not np.isnan(best)
            for k in range(1, window):
                v = data[base + k, j]
                if np.isnan(v):
                    continue
                if (not seen) or (v < best):
                    best = v
                    best_k = k
                    seen = True
            out[i, j] = (best_k + 1.0) if seen else np.nan
    return out


def ts_rank(df, window=10):
    w = _window(window)
    values, index, columns, single = _to_float2d(df)
    return _from_float2d(_ts_rank_2d(values, w), index, columns, single)


def product(df, window=10):
    w = _window(window)
    values, index, columns, single = _to_float2d(df)
    return _from_float2d(_product_2d(values, w), index, columns, single)


def ts_min(df, window=10):
    return df.rolling(_window(window), min_periods=1).min()


def ts_max(df, window=10):
    return df.rolling(_window(window), min_periods=1).max()


def delta(df, period=1):
    return df.diff(_window(period))


def delay(df, period=1):
    return df.shift(_window(period))


def rank(df):
    """Cross-sectional rank across symbols at each date."""
    return df.rank(axis=1, pct=True)


def scale(df, k=1):
    """Cross-sectional scale so sum(|x|) == k per date."""
    denom = df.abs().sum(axis=1).replace(0, np.nan)
    return df.mul(k).div(denom, axis=0)


def ts_argmax(df, window=10):
    w = _window(window)
    values, index, columns, single = _to_float2d(df)
    return _from_float2d(_ts_argmax_2d(values, w), index, columns, single)


def ts_argmin(df, window=10):
    w = _window(window)
    values, index, columns, single = _to_float2d(df)
    return _from_float2d(_ts_argmin_2d(values, w), index, columns, single)


def decay_linear(df, period=10):
    """Linear weighted moving average along time (rows)."""
    period = _window(period)
    if isinstance(df, pd.Series):
        frame = df.to_frame("__s__")
        single = True
    else:
        frame = df
        single = False

    # 只能向前填充。原实现的 bfill 会把未来值搬到过去，系统性美化所有用到
    # decay_linear 的 alpha。首个有效观测之前用 0 只是为了让 numba 核能跑，
    # 计算完再把这段窗口置回 NaN。
    cleaned = frame.astype(float).ffill()
    warmup = cleaned.notna().to_numpy().cumsum(axis=0) < period
    values = np.ascontiguousarray(cleaned.fillna(0.0).to_numpy(dtype=np.float64))
    weights = np.arange(1, period + 1, dtype=np.float64)
    weights /= weights.sum()
    out = _decay_linear_2d(values, weights)
    out[warmup] = np.nan
    result = pd.DataFrame(out, index=cleaned.index, columns=cleaned.columns)
    if single:
        return result["__s__"]
    return result


def _elem_max(a, b):
    return a.where(a >= b, b)


def _elem_min(a, b):
    return a.where(a <= b, b)


def _as_float(df):
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        return df.astype(float)
    return df


def _finite_or_nan(df):
    """Keep missing as missing. Inf must not become a tradable 0/1."""
    return df.replace([np.inf, -np.inf], np.nan)


class IndClass:
    """Map WorldQuant IndClass levels to Shenwan L1/L2/L3."""

    sector = "l1"  # 一级
    industry = "l2"  # 二级
    subindustry = "l3"  # 三级


def indneutralize(df: pd.DataFrame, industry) -> pd.DataFrame:
    """Subtract cross-sectional industry mean (per date) from a wide panel.

    ``industry`` 可以是列标签 Series，或 date×symbol 的 PIT 宽表。
    """
    if df is None or df.empty:
        return df
    if isinstance(industry, pd.DataFrame):
        labels = industry.reindex(index=df.index, columns=df.columns)
        stacked = df.stack(future_stack=True)
        lab = labels.stack(future_stack=True)
        dates = stacked.index.get_level_values(0)
        keys = pd.Series(list(zip(dates.to_numpy(), lab.to_numpy())), index=stacked.index)
        means = stacked.groupby(keys, dropna=False).transform("mean")
        out = (stacked - means).unstack()
        out = out.where(lab.unstack().notna())
        return out.reindex(index=df.index, columns=df.columns)
    labels = industry.reindex(df.columns)
    transposed = df.T
    demeaned = transposed.groupby(labels, dropna=False).transform(
        lambda block: block - block.mean()
    )
    missing = labels.isna()
    if missing.any():
        demeaned.loc[missing] = np.nan
    return demeaned.T


def _load_industry_panel(dates, symbols, level: str, src: str = "SW2021") -> pd.DataFrame:
    """date × symbol 行业代码（PIT）。"""
    dates = pd.Index(pd.to_datetime(dates).strftime("%Y-%m-%d"))
    symbols = pd.Index([str(s) for s in symbols])
    empty = pd.DataFrame(np.nan, index=dates, columns=symbols)
    db_path = _resolve_db_path()
    if not db_path:
        logger.warning("未配置数据库路径，行业中性因子将得到空行业映射")
        return empty
    from DailyUpdates.storage import SQLiteStorage
    from FactorEvaluates.industry_panel import load_code_panel

    storage = SQLiteStorage(db_path)
    panel, _ = load_code_panel(storage, dates, symbols, src=src, level=level)
    return panel


class Alphas:
    """Alpha formulas operating on wide panels (date x symbol)."""

    def __init__(self, fields: Dict[str, pd.DataFrame]):
        self.open = fields["open"]
        self.high = fields["high"]
        self.low = fields["low"]
        self.close = fields["close"]
        self.volume = fields["volume"]
        self.returns = fields["returns"]
        self.vwap = fields["vwap"]
        self.cap = fields.get("cap")
        symbols = self.close.columns
        dates = self.close.index
        self._ind_sector = _load_industry_panel(dates, symbols, IndClass.sector)
        self._ind_industry = _load_industry_panel(dates, symbols, IndClass.industry)
        self._ind_subindustry = _load_industry_panel(dates, symbols, IndClass.subindustry)

    def _ind(self, level: str) -> pd.Series:
        if level == IndClass.sector:
            return self._ind_sector
        if level == IndClass.industry:
            return self._ind_industry
        return self._ind_subindustry

    def IndNeutralize(self, df: pd.DataFrame, level: str) -> pd.DataFrame:
        return indneutralize(df, self._ind(level))

    def _adv(self, window: int) -> pd.DataFrame:
        return sma(self.volume, window)

    # rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2), 5)) - 0.5
    def alpha001(self):
        downside = self.returns < 0
        power_base = self.close.where(~downside, stddev(self.returns, 20))
        return rank(ts_argmax(power_base ** 2, 5)) - 0.5

    # -1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)
    def alpha002(self):
        vol_chg = rank(delta(np.log(self.volume), 2))
        day_ret = rank((self.close - self.open) / self.open)
        return (-correlation(vol_chg, day_ret, 6)).pipe(_finite_or_nan)

    # -1 * correlation(rank(open), rank(volume), 10)
    def alpha003(self):
        return (-correlation(rank(self.open), rank(self.volume), 10)).pipe(_finite_or_nan)

    # -1 * Ts_Rank(rank(low), 9)
    def alpha004(self):
        return -ts_rank(rank(self.low), 9)

    # rank(open - sum(vwap, 10) / 10) * (-1 * abs(rank(close - vwap)))
    def alpha005(self):
        open_gap = rank(self.open - ts_sum(self.vwap, 10) / 10)
        close_gap = np.abs(rank(self.close - self.vwap))
        return open_gap * (-close_gap)

    # -1 * correlation(open, volume, 10)
    def alpha006(self):
        return (-correlation(self.open, self.volume, 10)).pipe(_finite_or_nan)

    # (adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : -1
    def alpha007(self):
        close_delta = delta(self.close, 7)
        trend = -ts_rank(np.abs(close_delta), 60) * np.sign(close_delta)
        return trend.mask(self._adv(20) >= self.volume, -1)

    # -1 * rank((sum(open, 5) * sum(returns, 5)) - delay(sum(open, 5) * sum(returns, 5), 10))
    def alpha008(self):
        prod = ts_sum(self.open, 5) * ts_sum(self.returns, 5)
        return -rank(prod - delay(prod, 10))

    # (0 < ts_min(delta(close, 1), 5)) ? delta : ((ts_max(delta, 5) < 0) ? delta : -delta)
    def alpha009(self):
        step = delta(self.close, 1)
        persist = (ts_min(step, 5) > 0) | (ts_max(step, 5) < 0)
        return (-step).mask(persist, step)

    # rank(same as #9 with window 4)
    def alpha010(self):
        step = delta(self.close, 1)
        persist = (ts_min(step, 4) > 0) | (ts_max(step, 4) < 0)
        return rank((-step).mask(persist, step))

    # (rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(delta(volume, 3))
    def alpha011(self):
        gap = self.vwap - self.close
        return (rank(ts_max(gap, 3)) + rank(ts_min(gap, 3))) * rank(delta(self.volume, 3))

    # sign(delta(volume, 1)) * (-1 * delta(close, 1))
    def alpha012(self):
        return np.sign(delta(self.volume, 1)) * (-delta(self.close, 1))

    # -1 * rank(covariance(rank(close), rank(volume), 5))
    def alpha013(self):
        return -rank(covariance(rank(self.close), rank(self.volume), 5))

    # (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)
    def alpha014(self):
        corr = correlation(self.open, self.volume, 10).pipe(_finite_or_nan)
        return -rank(delta(self.returns, 3)) * corr

    # -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)
    def alpha015(self):
        corr = correlation(rank(self.high), rank(self.volume), 3).pipe(_finite_or_nan)
        return -ts_sum(rank(corr), 3)

    # -1 * rank(covariance(rank(high), rank(volume), 5))
    def alpha016(self):
        return -rank(covariance(rank(self.high), rank(self.volume), 5))

    # ((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank(volume / adv20, 5))
    def alpha017(self):
        accel = delta(delta(self.close, 1), 1)
        vol_ratio = self.volume / self._adv(20)
        return -rank(ts_rank(self.close, 10)) * rank(accel) * rank(ts_rank(vol_ratio, 5))

    # -1 * rank(stddev(abs(close - open), 5) + (close - open) + correlation(close, open, 10))
    def alpha018(self):
        candle = self.close - self.open
        corr = correlation(self.close, self.open, 10).pipe(_finite_or_nan)
        return -rank(stddev(np.abs(candle), 5) + candle + corr)

    # (-1 * sign((close - delay(close, 7)) + delta(close, 7))) * (1 + rank(1 + sum(returns, 250)))
    def alpha019(self):
        week_move = (self.close - delay(self.close, 7)) + delta(self.close, 7)
        return (-np.sign(week_move)) * (1 + rank(1 + ts_sum(self.returns, 250)))

    # ((-1 * rank(open - delay(high, 1))) * rank(open - delay(close, 1))) * rank(open - delay(low, 1))
    def alpha020(self):
        return (
            -rank(self.open - delay(self.high, 1))
            * rank(self.open - delay(self.close, 1))
            * rank(self.open - delay(self.low, 1))
        )

    # mean8+std8 < mean2 ? -1 : (mean2 < mean8-std8 ? 1 : (volume/adv20 >= 1 ? 1 : -1))
    def alpha021(self):
        mean8 = sma(self.close, 8)
        std8 = stddev(self.close, 8)
        mean2 = sma(self.close, 2)
        weak = mean8 + std8 < mean2
        strong = mean2 < mean8 - std8
        heavy = self.volume / self._adv(20) >= 1
        out = pd.DataFrame(-1.0, index=self.close.index, columns=self.close.columns)
        return out.mask(heavy, 1).mask(strong, 1).mask(weak, -1)

    # -1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))
    def alpha022(self):
        corr = correlation(self.high, self.volume, 5).pipe(_finite_or_nan)
        return -delta(corr, 5) * rank(stddev(self.close, 20))

    # (sum(high, 20) / 20 < high) ? (-1 * delta(high, 2)) : 0
    def alpha023(self):
        zeros = pd.DataFrame(0.0, index=self.close.index, columns=self.close.columns)
        return zeros.mask(sma(self.high, 20) < self.high, -delta(self.high, 2))

    # (delta(sum(close, 100)/100, 100) / delay(close, 100) <= 0.05) ? -(close - ts_min(close, 100)) : -delta(close, 3)
    def alpha024(self):
        ratio = delta(sma(self.close, 100), 100) / delay(self.close, 100)
        fallback = -delta(self.close, 3)
        drawn = -(self.close - ts_min(self.close, 100))
        return fallback.mask(ratio <= 0.05, drawn)

    # rank((-returns) * adv20 * vwap * (high - close))
    def alpha025(self):
        return rank((-self.returns) * self._adv(20) * self.vwap * (self.high - self.close))

    # -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
    def alpha026(self):
        corr = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        return -ts_max(corr.pipe(_finite_or_nan), 3)

    # (0.5 < rank(sum(correlation(rank(volume), rank(vwap), 6), 2) / 2)) ? -1 : 1
    def alpha027(self):
        # sum(..., 2) / 2 is sma(..., 2); do not divide again
        score = rank(sma(correlation(rank(self.volume), rank(self.vwap), 6), 2))
        return score.mask(score > 0.5, -1).mask(score <= 0.5, 1)

    # scale((correlation(adv20, low, 5) + (high + low) / 2) - close)
    def alpha028(self):
        corr = correlation(self._adv(20), self.low, 5).pipe(_finite_or_nan)
        mid = (self.high + self.low) / 2
        return scale(corr + mid - self.close)

    # min(product(rank(rank(scale(log(sum(ts_min(rank(rank(-rank(delta(close-1, 5)))), 2), 1))))), 1), 5) + ts_rank(delay(-returns, 6), 5)
    def alpha029(self):
        nested = rank(rank(-rank(delta(self.close - 1, 5))))
        compressed = product(rank(rank(scale(np.log(ts_sum(ts_min(nested, 2), 1))))), 1)
        return ts_min(compressed, 5) + ts_rank(delay(-self.returns, 6), 5)

    # (1 - rank(sign(d1) + sign(d2) + sign(d3))) * sum(volume, 5) / sum(volume, 20)
    def alpha030(self):
        step = delta(self.close, 1)
        signs = np.sign(step) + np.sign(delay(step, 1)) + np.sign(delay(step, 2))
        return (1.0 - rank(signs)) * ts_sum(self.volume, 5) / ts_sum(self.volume, 20)

    # rank^3(decay_linear(-rank(rank(delta(close, 10))), 10)) + rank(-delta(close, 3)) + sign(scale(correlation(adv20, low, 12)))
    def alpha031(self):
        decayed = decay_linear(-rank(rank(delta(self.close, 10))), 10)
        corr = correlation(self._adv(20), self.low, 12).pipe(_finite_or_nan)
        return rank(rank(rank(decayed))) + rank(-delta(self.close, 3)) + np.sign(scale(corr))

    # scale(sum(close, 7)/7 - close) + 20 * scale(correlation(vwap, delay(close, 5), 230))
    def alpha032(self):
        mean_gap = scale(sma(self.close, 7) - self.close)
        delayed = scale(correlation(self.vwap, delay(self.close, 5), 230))
        return mean_gap + 20 * delayed

    # rank(-1 * ((1 - open/close)^1))
    def alpha033(self):
        return rank(-1 + self.open / self.close)

    # rank((1 - rank(stddev(returns, 2) / stddev(returns, 5))) + (1 - rank(delta(close, 1))))
    def alpha034(self):
        vol_ratio = (stddev(self.returns, 2) / stddev(self.returns, 5)).pipe(_finite_or_nan)
        return rank(2 - rank(vol_ratio) - rank(delta(self.close, 1)))

    # Ts_Rank(volume, 32) * (1 - Ts_Rank(close + high - low, 16)) * (1 - Ts_Rank(returns, 32))
    def alpha035(self):
        range_rank = ts_rank(self.close + self.high - self.low, 16)
        return ts_rank(self.volume, 32) * (1 - range_rank) * (1 - ts_rank(self.returns, 32))

    # 2.21*rank(corr(close-open, delay(volume,1), 15)) + 0.7*rank(open-close) + 0.73*rank(Ts_Rank(delay(-returns, 6), 5)) + rank(abs(corr(vwap, adv20, 6))) + 0.6*rank((sma(close,200)-open)*(close-open))
    def alpha036(self):
        candle = self.close - self.open
        t1 = 2.21 * rank(correlation(candle, delay(self.volume, 1), 15))
        t2 = 0.7 * rank(self.open - self.close)
        t3 = 0.73 * rank(ts_rank(delay(-self.returns, 6), 5))
        t4 = rank(np.abs(correlation(self.vwap, self._adv(20), 6)))
        t5 = 0.6 * rank((sma(self.close, 200) - self.open) * candle)
        return t1 + t2 + t3 + t4 + t5

    # rank(correlation(delay(open - close, 1), close, 200)) + rank(open - close)
    def alpha037(self):
        lagged = delay(self.open - self.close, 1)
        return rank(correlation(lagged, self.close, 200)) + rank(self.open - self.close)

    # (-1 * rank(Ts_Rank(close, 10))) * rank(close / open)
    def alpha038(self):
        ratio = (self.close / self.open).pipe(_finite_or_nan)
        return -rank(ts_rank(self.close, 10)) * rank(ratio)

    # (-1 * rank(delta(close, 7) * (1 - rank(decay_linear(volume / adv20, 9))))) * (1 + rank(sum(returns, 250)))
    def alpha039(self):
        # existing engine used sma(returns, 250) rather than ts_sum
        faded = 1 - rank(decay_linear(self.volume / self._adv(20), 9))
        return (-rank(delta(self.close, 7) * faded)) * (1 + rank(sma(self.returns, 250)))

    # (-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)
    def alpha040(self):
        return -rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    # (high * low)^0.5 - vwap
    def alpha041(self):
        return (self.high * self.low) ** 0.5 - self.vwap

    # rank(vwap - close) / rank(vwap + close)
    def alpha042(self):
        return rank(self.vwap - self.close) / rank(self.vwap + self.close)

    # ts_rank(volume / adv20, 20) * ts_rank(-delta(close, 7), 8)
    def alpha043(self):
        return ts_rank(self.volume / self._adv(20), 20) * ts_rank(-delta(self.close, 7), 8)

    # -1 * correlation(high, rank(volume), 5)
    def alpha044(self):
        return (-correlation(self.high, rank(self.volume), 5)).pipe(_finite_or_nan)

    # -1 * (rank(sum(delay(close, 5), 20) / 20) * correlation(close, volume, 2) * rank(correlation(sum(close, 5), sum(close, 20), 2)))
    def alpha045(self):
        corr_cv = correlation(self.close, self.volume, 2).pipe(_finite_or_nan)
        corr_sum = correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)
        return -rank(sma(delay(self.close, 5), 20)) * corr_cv * rank(corr_sum)

    # inner > 0.25 ? -1 : (inner < 0 ? 1 : -delta(close))
    def alpha046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - (
            (delay(self.close, 10) - self.close) / 10
        )
        return (-delta(self.close)).mask(inner < 0, 1).mask(inner > 0.25, -1)

    # ((rank(1/close) * volume) / adv20) * ((high * rank(high - close)) / (sum(high, 5)/5)) - rank(vwap - delay(vwap, 5))
    def alpha047(self):
        left = (rank(1 / self.close) * self.volume) / self._adv(20)
        right = (self.high * rank(self.high - self.close)) / sma(self.high, 5)
        return left * right - rank(self.vwap - delay(self.vwap, 5))

    # IndNeutralize((corr(delta(close,1), delta(delay(close,1),1), 250) * delta(close,1)) / close, subindustry) / sum((delta(close,1)/delay(close,1))^2, 250)
    def alpha048(self):
        d1 = delta(self.close, 1)
        numer = correlation(d1, delta(delay(self.close, 1), 1), 250) * d1 / self.close
        denom = ts_sum((d1 / delay(self.close, 1)) ** 2, 250)
        return self.IndNeutralize(numer, IndClass.subindustry) / denom

    # inner < -0.1 ? 1 : -(close - delay(close, 1))
    def alpha049(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - (
            (delay(self.close, 10) - self.close) / 10
        )
        return (-delta(self.close)).mask(inner < -0.1, 1)

    # -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)
    def alpha050(self):
        return -ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5)

    # inner < -0.05 ? 1 : -(close - delay(close, 1))
    def alpha051(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - (
            (delay(self.close, 10) - self.close) / 10
        )
        return (-delta(self.close)).mask(inner < -0.05, 1)

    # ((-ts_min(low, 5) + delay(ts_min(low, 5), 5)) * rank((sum(returns, 240) - sum(returns, 20)) / 220)) * ts_rank(volume, 5)
    def alpha052(self):
        trough = -delta(ts_min(self.low, 5), 5)
        ret_gap = (ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220
        return trough * rank(ret_gap) * ts_rank(self.volume, 5)

    # -1 * delta(((close - low) - (high - close)) / (close - low), 9)
    def alpha053(self):
        denom = (self.close - self.low).replace(0, 0.0001)
        wick = ((self.close - self.low) - (self.high - self.close)) / denom
        return -delta(wick, 9)

    # -1 * ((low - close) * open^5) / ((low - high) * close^5)
    def alpha054(self):
        span = (self.low - self.high).replace(0, -0.0001)
        return -(self.low - self.close) * (self.open ** 5) / (span * (self.close ** 5))

    # -1 * correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)
    def alpha055(self):
        span = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        stoch = (self.close - ts_min(self.low, 12)) / span
        return (-correlation(rank(stoch), rank(self.volume), 6)).pipe(_finite_or_nan)

    # 0 - (rank(sum(returns, 10) / sum(sum(returns, 2), 3)) * rank(returns * cap))
    def alpha056(self):
        if self.cap is None:
            return self.close * np.nan
        quality = rank(ts_sum(self.returns, 10) / ts_sum(ts_sum(self.returns, 2), 3))
        size = rank(self.returns * self.cap)
        return -(quality * size)

    # 0 - ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))
    def alpha057(self):
        weights = decay_linear(rank(ts_argmax(self.close, 30)), 2)
        return -((self.close - self.vwap) / weights)

    # -1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, sector), volume, 3.92795), 7.89291), 5.50322)
    def alpha058(self):
        neutralized = self.IndNeutralize(self.vwap, IndClass.sector)
        return -ts_rank(decay_linear(correlation(neutralized, self.volume, 4), 8), 6)

    # -1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap*0.728317 + vwap*(1-0.728317), industry), volume, 4.25197), 16.2289), 8.19648)
    def alpha059(self):
        mixed = self.vwap * 0.728317 + self.vwap * (1 - 0.728317)
        neutralized = self.IndNeutralize(mixed, IndClass.industry)
        return -ts_rank(decay_linear(correlation(neutralized, self.volume, 4), 16), 8)

    # 0 - ((2 * scale(rank((((close-low)-(high-close))/(high-low))*volume))) - scale(rank(ts_argmax(close, 10))))
    def alpha060(self):
        span = (self.high - self.low).replace(0, 0.0001)
        clv = ((self.close - self.low) - (self.high - self.close)) * self.volume / span
        return -(2 * scale(rank(clv)) - scale(rank(ts_argmax(self.close, 10))))

    # rank(vwap - ts_min(vwap, 16.1219)) < rank(correlation(vwap, adv180, 17.9282))
    def alpha061(self):
        left = rank(self.vwap - ts_min(self.vwap, 16))
        right = rank(correlation(self.vwap, self._adv(180), 18))
        return left < right

    # (rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high)))) * -1
    def alpha062(self):
        left = rank(correlation(self.vwap, sma(self._adv(20), 22), 10))
        cmp = (rank(self.open) + rank(self.open)) < (
            rank((self.high + self.low) / 2) + rank(self.high)
        )
        return (left < rank(cmp)) * -1

    # (rank(decay_linear(delta(IndNeutralize(close, industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(vwap*0.318108+open*(1-0.318108), sum(adv180, 37.2467), 13.557), 12.2883))) * -1
    def alpha063(self):
        p1 = rank(decay_linear(delta(self.IndNeutralize(self.close, IndClass.industry), 2), 8))
        mix = self.vwap * 0.318108 + self.open * (1 - 0.318108)
        p2 = rank(decay_linear(correlation(mix, sma(self._adv(180), 37), 14), 12))
        return (p1 - p2) * -1

    # (rank(correlation(sum(open*0.178404+low*(1-0.178404), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((high+low)/2)*0.178404 + vwap*(1-0.178404), 3.69741))) * -1
    def alpha064(self):
        mix_ol = self.open * 0.178404 + self.low * (1 - 0.178404)
        mix_hl = ((self.high + self.low) / 2) * 0.178404 + self.vwap * (1 - 0.178404)
        left = rank(correlation(sma(mix_ol, 13), sma(self._adv(120), 13), 17))
        return (left < rank(delta(mix_hl, 3.69741))) * -1

    # (rank(correlation(open*0.00817205+vwap*(1-0.00817205), sum(adv60, 8.6911), 6.40374)) < rank(open - ts_min(open, 13.635))) * -1
    def alpha065(self):
        mix = self.open * 0.00817205 + self.vwap * (1 - 0.00817205)
        left = rank(correlation(mix, sma(self._adv(60), 9), 6))
        return (left < rank(self.open - ts_min(self.open, 14))) * -1

    # (rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((low*0.96633+low*(1-0.96633))-vwap)/(open-(high+low)/2), 11.4157), 6.72611)) * -1
    def alpha066(self):
        mix_low = self.low * 0.96633 + self.low * (1 - 0.96633)
        mid = (self.high + self.low) / 2
        p1 = rank(decay_linear(delta(self.vwap, 4), 7))
        p2 = ts_rank(decay_linear((mix_low - self.vwap) / (self.open - mid), 11), 7)
        return (p1 + p2) * -1

    # (rank(high - ts_min(high, 2.14593)) ^ rank(correlation(IndNeutralize(vwap, sector), IndNeutralize(adv20, subindustry), 6.02936))) * -1
    def alpha067(self):
        left = rank(self.high - ts_min(self.high, 2))
        corr = correlation(
            self.IndNeutralize(self.vwap, IndClass.sector),
            self.IndNeutralize(self._adv(20), IndClass.subindustry),
            6,
        )
        return left.pow(rank(corr)) * -1

    # (Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(close*0.518371+low*(1-0.518371), 1.06157))) * -1
    def alpha068(self):
        left = ts_rank(correlation(rank(self.high), rank(self._adv(15)), 9), 14)
        mix = self.close * 0.518371 + self.low * (1 - 0.518371)
        return (left < rank(delta(mix, 1.06157))) * -1

    # (rank(ts_max(delta(IndNeutralize(vwap, industry), 2.72412), 4.79344)) ^ Ts_Rank(correlation(close*0.490655+vwap*(1-0.490655), adv20, 4.92416), 9.0615)) * -1
    def alpha069(self):
        left = rank(ts_max(delta(self.IndNeutralize(self.vwap, IndClass.industry), 3), 5))
        mix = self.close * 0.490655 + self.vwap * (1 - 0.490655)
        return left.pow(ts_rank(correlation(mix, self._adv(20), 5), 9)) * -1

    # (rank(delta(vwap, 1.29456)) ^ Ts_Rank(correlation(IndNeutralize(close, industry), adv50, 17.8256), 17.9171)) * -1
    def alpha070(self):
        left = rank(delta(self.vwap, 1))
        corr = correlation(self.IndNeutralize(self.close, IndClass.industry), self._adv(50), 18)
        return left.pow(ts_rank(corr, 18)) * -1

    # max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear(rank((low+open)-(vwap+vwap))^2, 16.4662), 4.4388))
    def alpha071(self):
        p1 = ts_rank(
            decay_linear(correlation(ts_rank(self.close, 3), ts_rank(self._adv(180), 12), 18), 4),
            16,
        )
        p2 = ts_rank(
            decay_linear(rank((self.low + self.open) - (self.vwap + self.vwap)).pow(2), 16),
            4,
        )
        return _elem_max(p1, p2)

    # rank(decay_linear(correlation((high+low)/2, adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011))
    def alpha072(self):
        mid = (self.high + self.low) / 2
        numer = rank(decay_linear(correlation(mid, self._adv(40), 9), 10))
        denom = rank(decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7), 3))
        return numer / denom

    # (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear((delta(open*0.147155+low*(1-0.147155), 2.03608) / (open*0.147155+low*(1-0.147155))) * -1, 3.33829), 16.7411)) * -1
    def alpha073(self):
        p1 = rank(decay_linear(delta(self.vwap, 5), 3))
        mix = self.open * 0.147155 + self.low * (1 - 0.147155)
        p2 = ts_rank(decay_linear((delta(mix, 2) / mix) * -1, 3), 17)
        return -_elem_max(p1, p2)

    # (rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(high*0.0261661+vwap*(1-0.0261661)), rank(volume), 11.4791))) * -1
    def alpha074(self):
        left = rank(correlation(self.close, sma(self._adv(30), 37), 15))
        mix = self.high * 0.0261661 + self.vwap * (1 - 0.0261661)
        right = rank(correlation(rank(mix), rank(self.volume), 11))
        return (left < right) * -1

    # rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413))
    def alpha075(self):
        left = rank(correlation(self.vwap, self.volume, 4))
        right = rank(correlation(rank(self.low), rank(self._adv(50)), 12))
        return left < right

    # (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1
    def alpha076(self):
        p1 = rank(decay_linear(delta(self.vwap, 1), 12))
        corr = correlation(self.IndNeutralize(self.low, IndClass.sector), self._adv(81), 8)
        p2 = ts_rank(decay_linear(ts_rank(corr, 20), 17), 19)
        return -_elem_max(p1, p2)

    # min(rank(decay_linear(((high+low)/2 + high) - (vwap + high), 20.0451)), rank(decay_linear(correlation((high+low)/2, adv40, 3.1614), 5.64125)))
    def alpha077(self):
        mid = (self.high + self.low) / 2
        p1 = rank(decay_linear((mid + self.high) - (self.vwap + self.high), 20))
        p2 = rank(decay_linear(correlation(mid, self._adv(40), 3), 6))
        return _elem_min(p1, p2)

    # rank(correlation(sum(low*0.352233+vwap*(1-0.352233), 19.7428), sum(adv40, 19.7428), 6.83313)) ^ rank(correlation(rank(vwap), rank(volume), 5.77492))
    def alpha078(self):
        mix = self.low * 0.352233 + self.vwap * (1 - 0.352233)
        left = rank(correlation(ts_sum(mix, 20), ts_sum(self._adv(40), 20), 7))
        return left.pow(rank(correlation(rank(self.vwap), rank(self.volume), 6)))

    # rank(delta(IndNeutralize(close*0.60733+open*(1-0.60733), sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644))
    def alpha079(self):
        mix = self.close * 0.60733 + self.open * (1 - 0.60733)
        left = rank(delta(self.IndNeutralize(mix, IndClass.sector), 1))
        right = rank(correlation(ts_rank(self.vwap, 4), ts_rank(self._adv(150), 9), 15))
        return left < right

    # (rank(Sign(delta(IndNeutralize(open*0.868128+high*(1-0.868128), industry), 4.04545))) ^ Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1
    def alpha080(self):
        mix = self.open * 0.868128 + self.high * (1 - 0.868128)
        left = rank(np.sign(delta(self.IndNeutralize(mix, IndClass.industry), 4)))
        return left.pow(ts_rank(correlation(self.high, self._adv(10), 5), 6)) * -1

    # (rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1
    def alpha081(self):
        powered = rank(correlation(self.vwap, ts_sum(self._adv(10), 50), 8)).pow(4)
        left = rank(np.log(product(rank(powered), 15)))
        right = rank(correlation(rank(self.vwap), rank(self.volume), 5))
        return (left < right) * -1

    # (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, sector), open*0.634196+open*(1-0.634196), 17.4842), 6.92131), 13.4283)) * -1
    def alpha082(self):
        p1 = rank(decay_linear(delta(self.open, 1), 15))
        corr = correlation(self.IndNeutralize(self.volume, IndClass.sector), self.open, 17)
        p2 = ts_rank(decay_linear(corr, 7), 13)
        return -_elem_min(p1, p2)

    # (rank(delay((high-low)/(sum(close,5)/5), 2)) * rank(rank(volume))) / (((high-low)/(sum(close,5)/5)) / (vwap - close))
    def alpha083(self):
        rng = (self.high - self.low) / (ts_sum(self.close, 5) / 5)
        return (rank(delay(rng, 2)) * rank(rank(self.volume))) / (rng / (self.vwap - self.close))

    # SignedPower(Ts_Rank(vwap - ts_max(vwap, 15.3217), 20.7127), delta(close, 4.96796))
    def alpha084(self):
        return ts_rank(self.vwap - ts_max(self.vwap, 15), 21).pow(delta(self.close, 5))

    # rank(correlation(high*0.876703+close*(1-0.876703), adv30, 9.61331)) ^ rank(correlation(Ts_Rank((high+low)/2, 3.70596), Ts_Rank(volume, 10.1595), 7.11408))
    def alpha085(self):
        mix = self.high * 0.876703 + self.close * (1 - 0.876703)
        left = rank(correlation(mix, self._adv(30), 10))
        right = rank(correlation(ts_rank((self.high + self.low) / 2, 4), ts_rank(self.volume, 10), 7))
        return left.pow(right)

    # (Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank((open+close) - (vwap+open))) * -1
    def alpha086(self):
        left = ts_rank(correlation(self.close, sma(self._adv(20), 15), 6), 20)
        right = rank((self.open + self.close) - (self.vwap + self.open))
        return (left < right) * -1

    # (max(rank(decay_linear(delta(close*0.369701+vwap*(1-0.369701), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, industry), close, 13.4132)), 4.89768), 14.4535)) * -1
    def alpha087(self):
        mix = self.close * 0.369701 + self.vwap * (1 - 0.369701)
        p1 = rank(decay_linear(delta(mix, 2), 3))
        corr = np.abs(
            correlation(self.IndNeutralize(self._adv(81), IndClass.industry), self.close, 13)
        )
        p2 = ts_rank(decay_linear(corr, 5), 14)
        return -_elem_max(p1, p2)

    # min(rank(decay_linear((rank(open)+rank(low))-(rank(high)+rank(close)), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
    def alpha088(self):
        p1 = rank(
            decay_linear((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close)), 8)
        )
        p2 = ts_rank(
            decay_linear(correlation(ts_rank(self.close, 8), ts_rank(self._adv(60), 21), 8), 7),
            3,
        )
        return _elem_min(p1, p2)

    # Ts_Rank(decay_linear(correlation(low*0.967285+low*(1-0.967285), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, industry), 3.48158), 10.1466), 15.3012)
    def alpha089(self):
        p1 = ts_rank(decay_linear(correlation(self.low, self._adv(10), 7), 6), 4)
        p2 = ts_rank(
            decay_linear(delta(self.IndNeutralize(self.vwap, IndClass.industry), 3), 10),
            15,
        )
        return p1 - p2

    # (rank(close - ts_max(close, 4.66719)) ^ Ts_Rank(correlation(IndNeutralize(adv40, subindustry), low, 5.38375), 3.21856)) * -1
    def alpha090(self):
        left = rank(self.close - ts_max(self.close, 5))
        corr = correlation(self.IndNeutralize(self._adv(40), IndClass.subindustry), self.low, 5)
        return left.pow(ts_rank(corr, 3)) * -1

    # (Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1
    def alpha091(self):
        inner = correlation(self.IndNeutralize(self.close, IndClass.industry), self.volume, 10)
        p1 = ts_rank(decay_linear(decay_linear(inner, 16), 4), 5)
        p2 = rank(decay_linear(correlation(self.vwap, self._adv(30), 4), 3))
        return (p1 - p2) * -1

    # min(Ts_Rank(decay_linear(((high+low)/2 + close) < (low+open), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
    def alpha092(self):
        flag = ((self.high + self.low) / 2 + self.close) < (self.low + self.open)
        p1 = ts_rank(decay_linear(flag, 15), 19)
        p2 = ts_rank(decay_linear(correlation(rank(self.low), rank(self._adv(30)), 8), 7), 7)
        return _elem_min(p1, p2)

    # Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(close*0.524434+vwap*(1-0.524434), 2.77377), 16.2664))
    def alpha093(self):
        numer = ts_rank(
            decay_linear(
                correlation(self.IndNeutralize(self.vwap, IndClass.industry), self._adv(81), 17),
                20,
            ),
            8,
        )
        mix = self.close * 0.524434 + self.vwap * (1 - 0.524434)
        return numer / rank(decay_linear(delta(mix, 3), 16))

    # (rank(vwap - ts_min(vwap, 11.5783)) ^ Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1
    def alpha094(self):
        left = rank(self.vwap - ts_min(self.vwap, 12))
        right = ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(self._adv(60), 4), 18), 3)
        return left.pow(right) * -1

    # rank(open - ts_min(open, 12.4105)) < Ts_Rank(rank(correlation(sum((high+low)/2, 19.1351), sum(adv40, 19.1351), 12.8742))^5, 11.7584)
    def alpha095(self):
        left = rank(self.open - ts_min(self.open, 12))
        mid = sma((self.high + self.low) / 2, 19)
        powered = rank(correlation(mid, sma(self._adv(40), 19), 13)).pow(5)
        return left < ts_rank(powered, 12)

    # (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1
    def alpha096(self):
        p1 = ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4), 8)
        p2 = ts_rank(
            decay_linear(
                ts_argmax(correlation(ts_rank(self.close, 7), ts_rank(self._adv(60), 4), 4), 13),
                14,
            ),
            13,
        )
        return -_elem_max(p1, p2)

    # (rank(decay_linear(delta(IndNeutralize(low*0.721001+vwap*(1-0.721001), industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1
    def alpha097(self):
        mix = self.low * 0.721001 + self.vwap * (1 - 0.721001)
        p1 = rank(decay_linear(delta(self.IndNeutralize(mix, IndClass.industry), 3), 20))
        p2 = ts_rank(
            decay_linear(
                ts_rank(correlation(ts_rank(self.low, 8), ts_rank(self._adv(60), 17), 5), 19),
                16,
            ),
            7,
        )
        return (p1 - p2) * -1

    # rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206))
    def alpha098(self):
        p1 = rank(decay_linear(correlation(self.vwap, sma(self._adv(5), 26), 5), 7))
        p2 = rank(
            decay_linear(
                ts_rank(ts_argmin(correlation(rank(self.open), rank(self._adv(15)), 21), 9), 7),
                8,
            )
        )
        return p1 - p2

    # (rank(correlation(sum((high+low)/2, 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1
    def alpha099(self):
        left = rank(correlation(ts_sum((self.high + self.low) / 2, 20), ts_sum(self._adv(60), 20), 9))
        return (left < rank(correlation(self.low, self.volume, 6))) * -1

    # 0 - (((1.5 * scale(IndNeutralize(IndNeutralize(rank((((close-low)-(high-close))/(high-low))*volume), sub), sub))) - scale(IndNeutralize(correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30)), sub))) * (volume / adv20))
    def alpha100(self):
        span = (self.high - self.low).replace(0, 0.0001)
        clv = ((self.close - self.low) - (self.high - self.close)) / span * self.volume
        p1 = 1.5 * scale(
            self.IndNeutralize(
                self.IndNeutralize(rank(clv), IndClass.subindustry),
                IndClass.subindustry,
            )
        )
        p2 = scale(
            self.IndNeutralize(
                correlation(self.close, rank(self._adv(20)), 5) - rank(ts_argmin(self.close, 30)),
                IndClass.subindustry,
            )
        )
        return -((p1 - p2) * (self.volume / self._adv(20)))

    # (close - open) / ((high - low) + 0.001)
    def alpha101(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)


# Implemented alpha ids (including industry-neutral formulas when industry_member exists).
IMPLEMENTED_ALPHAS: List[int] = sorted(
    int(name[5:])
    for name in dir(Alphas)
    if name.startswith("alpha") and name[5:].isdigit() and callable(getattr(Alphas, name))
)

_CACHE_LOCK = threading.Lock()
# 缓存宽表面板引擎（按行情 DataFrame id），不缓存 101 列结果表。
_ENGINE_CACHE: Dict[int, tuple] = {}


def _normalize_market(data: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    frame = data.copy() if copy else data
    if "ts_code" not in frame.columns and "symbol" in frame.columns:
        frame = frame.rename(columns={"symbol": "ts_code"})
    if "trade_date" not in frame.columns and "date" in frame.columns:
        frame = frame.rename(columns={"date": "trade_date"})
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(["ts_code", "trade_date"])

    required = ["open", "high", "low", "close", "vol", "amount"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Alpha101 缺少行情字段: {missing}")

    # Tushare 手/千元 → 股/元量纲。float32 降低全量面板峰值内存。
    vol = frame["vol"].to_numpy(dtype=np.float32, copy=False)
    amount = frame["amount"].to_numpy(dtype=np.float32, copy=False)
    frame["volume"] = vol * np.float32(100.0)
    frame["vwap"] = (amount * np.float32(1000.0)) / (vol * np.float32(100.0) + np.float32(1.0))
    if "pct_chg" in frame.columns:
        frame["returns"] = frame["pct_chg"].to_numpy(dtype=np.float32, copy=False) / np.float32(
            100.0
        )
    else:
        frame["returns"] = (
            frame.groupby("ts_code", sort=False)["close"]
            .pct_change()
            .astype(np.float32)
        )
    if "total_mv" in frame.columns:
        frame["cap"] = frame["total_mv"].to_numpy(dtype=np.float32, copy=False)
    for col in ("open", "high", "low", "close"):
        frame[col] = frame[col].astype(np.float32, copy=False)
    return frame


def _to_wide(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    wide = frame.pivot(index="trade_date", columns="ts_code", values=field)
    # pivot 常升为 float64；压回 float32 省一半宽表内存
    if np.issubdtype(wide.values.dtype, np.floating):
        wide = wide.astype(np.float32, copy=False)
    return wide.sort_index()


def _slice_template(
    template: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    work = template
    if start_date:
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        work = work.loc[work.index >= start]
    if end_date:
        end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        work = work.loc[work.index <= end]
    return work


def _wide_to_long(
    values,
    template: pd.DataFrame,
    name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Align any alpha output to the close panel and stack to long format."""
    work = _slice_template(template, start_date, end_date)
    if isinstance(values, pd.DataFrame):
        wide = values.reindex(index=work.index, columns=work.columns)
    elif isinstance(values, pd.Series):
        if isinstance(values.index, pd.MultiIndex) and values.index.nlevels >= 2:
            stacked = values.rename(name).reset_index()
            stacked.columns = ["trade_date", "ts_code", name]
            if start_date:
                start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
                stacked = stacked[stacked["trade_date"] >= start]
            if end_date:
                end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
                stacked = stacked[stacked["trade_date"] <= end]
            return stacked
        # Unexpected 1-D series — leave empty rather than broadcast wrongly.
        wide = pd.DataFrame(
            np.nan, index=work.index, columns=work.columns
        )
    else:
        arr = np.asarray(values)
        if arr.shape == template.shape:
            if len(work) == len(template) and work.index.equals(template.index):
                wide = pd.DataFrame(arr, index=template.index, columns=template.columns)
            else:
                row_i = template.index.get_indexer(work.index)
                wide = pd.DataFrame(arr[row_i], index=work.index, columns=work.columns)
        else:
            wide = pd.DataFrame(
                np.nan, index=work.index, columns=work.columns
            )
    if wide.dtypes.apply(lambda t: np.issubdtype(t, np.bool_)).any():
        wide = wide.astype(np.float32)
    # 没有收盘价的格子不得留下有限因子值，否则无行情证券会污染截面 rank。
    close_ok = np.isfinite(work.to_numpy(dtype=np.float64, copy=False))
    wide = wide.where(pd.DataFrame(close_ok, index=work.index, columns=work.columns))
    stacked = wide.stack(future_stack=True).rename(name).reset_index()
    stacked.columns = ["trade_date", "ts_code", name]
    return stacked


def prepare_alpha_engine(data: pd.DataFrame, copy: bool = True):
    """Build wide panels + Alphas engine."""
    frame = _normalize_market(data, copy=copy)
    fields = {
        "open": _to_wide(frame, "open"),
        "high": _to_wide(frame, "high"),
        "low": _to_wide(frame, "low"),
        "close": _to_wide(frame, "close"),
        "volume": _to_wide(frame, "volume"),
        "returns": _to_wide(frame, "returns"),
        "vwap": _to_wide(frame, "vwap"),
    }
    if "cap" in frame.columns:
        fields["cap"] = _to_wide(frame, "cap")
    template = fields["close"]
    engine = Alphas(fields)
    return engine, template


def get_prepared_engine(data: pd.DataFrame):
    """Reuse wide-panel engine for the same market DataFrame instance."""
    key = id(data)
    with _CACHE_LOCK:
        cached = _ENGINE_CACHE.get(key)
        if cached is not None:
            return cached
        logger.info(
            "Alpha101 准备宽表引擎: rows=%s symbols~=%s",
            len(data),
            data["ts_code"].nunique()
            if "ts_code" in data.columns
            else data["symbol"].nunique()
            if "symbol" in data.columns
            else "?",
        )
        engine, template = prepare_alpha_engine(data, copy=True)
        _ENGINE_CACHE.clear()
        _ENGINE_CACHE[key] = (engine, template)
        return engine, template


def clear_prepared_engine() -> None:
    with _CACHE_LOCK:
        _ENGINE_CACHE.clear()


def compute_one_alpha(
    engine: "Alphas",
    template: pd.DataFrame,
    alpha_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Compute a single alpha as long ``ts_code, trade_date, factor_value``."""
    col = f"alpha{alpha_id:03d}"
    method = getattr(engine, col)
    values = method()
    stacked = _wide_to_long(
        values, template, col, start_date=start_date, end_date=end_date
    )
    del values
    frame = stacked.rename(columns={col: "factor_value"})
    del stacked
    # 不落库 NaN，显著减小全量写入峰值与磁盘
    frame = frame.dropna(subset=["factor_value"])
    return frame.reset_index(drop=True)


def extract_alpha(
    data: pd.DataFrame, alpha_id: int, start_date: Optional[str] = None
) -> pd.DataFrame:
    """只计算单个 alpha；宽表引擎按行情对象缓存，避免 101 列大表。"""
    engine, template = get_prepared_engine(data)
    t0 = time.perf_counter()
    frame = compute_one_alpha(engine, template, alpha_id, start_date=start_date)
    logger.info(
        "Alpha101 完成 alpha%03d rows=%s (%.2fs)",
        alpha_id,
        len(frame),
        time.perf_counter() - t0,
    )
    return frame
