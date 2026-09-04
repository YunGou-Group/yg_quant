"""Fixed synthetic panel + PIT industry for Alpha101 paper-parity checks.

Used to freeze goldens from the pre-rewrite engine and to rebuild the same
inputs after the formula rewrite. Not a pytest module.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

PARITY_SEED = 20260904
PARITY_N_DAYS = 260
PARITY_SYMBOLS: List[str] = [f"{i:06d}.SZ" for i in range(1, 9)]
PARITY_ALPHA_IDS: Tuple[int, ...] = tuple(range(1, 102))


def make_parity_market(seed: int = PARITY_SEED) -> pd.DataFrame:
    """8 names × 260 sessions: geometric walk, consistent OHLC, a few NaNs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=PARITY_N_DAYS).strftime("%Y-%m-%d")
    rows = []
    for col, symbol in enumerate(PARITY_SYMBOLS):
        shock = rng.normal(0.0004, 0.018, size=PARITY_N_DAYS)
        close = 12.0 * (1.0 + 0.15 * col) * np.exp(np.cumsum(shock))
        open_ = close * np.exp(rng.normal(0.0, 0.004, size=PARITY_N_DAYS))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.012, size=PARITY_N_DAYS))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.012, size=PARITY_N_DAYS))
        vol = rng.uniform(8.0e4, 4.0e5, size=PARITY_N_DAYS) * (1.0 + 0.2 * col)
        amount = ((open_ + high + low + close) / 4.0) * vol / 10.0
        prev = np.empty_like(close)
        prev[0] = close[0]
        prev[1:] = close[:-1]
        pct_chg = (close / prev - 1.0) * 100.0
        total_mv = close * rng.uniform(8.0e8, 4.0e9) * (1.0 + 0.3 * col)
        punch = rng.integers(40, PARITY_N_DAYS - 10)
        close[punch] = np.nan
        open_[punch] = np.nan
        high[punch] = np.nan
        low[punch] = np.nan
        vol[punch] = np.nan
        amount[punch] = np.nan
        pct_chg[punch] = np.nan
        total_mv[punch] = np.nan
        for i, trade_date in enumerate(dates):
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "open": open_[i],
                    "high": high[i],
                    "low": low[i],
                    "close": close[i],
                    "vol": vol[i],
                    "amount": amount[i],
                    "pct_chg": pct_chg[i],
                    "total_mv": total_mv[i],
                }
            )
    return pd.DataFrame(rows)


def make_parity_industry(
    dates, symbols, level: str, src: str = "SW2021"
) -> pd.DataFrame:
    """2–3 groups with a mid-sample membership flip (PIT)."""
    del src
    dates = pd.Index(pd.to_datetime(dates).strftime("%Y-%m-%d"))
    symbols = pd.Index([str(s) for s in symbols])
    mid = dates[len(dates) // 2]
    n = len(symbols)
    if level == "l1":
        early = ["SEC_A"] * (n // 2) + ["SEC_B"] * (n - n // 2)
        late = list(early)
        if n >= 5:
            late[2] = "SEC_B"
            late[4] = "SEC_A"
    elif level == "l2":
        cycle = ["IND_X", "IND_Y", "IND_Z"]
        early = [cycle[i % 3] for i in range(n)]
        late = list(early)
        if n >= 6:
            late[1] = "IND_Z"
            late[5] = "IND_X"
    else:
        cycle = ["SUB_P", "SUB_Q", "SUB_R"]
        early = [cycle[i % 3] for i in range(n)]
        late = list(early)
        if n >= 4:
            late[0] = "SUB_R"
            late[3] = "SUB_P"
    panel = pd.DataFrame(index=dates, columns=symbols, dtype=object)
    for j, symbol in enumerate(symbols):
        panel.loc[dates < mid, symbol] = early[j]
        panel.loc[dates >= mid, symbol] = late[j]
    return panel


def patch_industry_loader(engine_mod) -> Callable:
    """Replace ``_load_industry_panel``; return the original for restore."""
    original = engine_mod._load_industry_panel
    engine_mod._load_industry_panel = make_parity_industry
    return original
