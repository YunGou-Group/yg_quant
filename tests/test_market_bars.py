"""入库前行情清洗：占位价、OHLC、成交量、adj 跳变。"""

import numpy as np
import pandas as pd

from DailyUpdates.preprocessing.market_bars import sanitize_market_bars


def _bars(**cols):
    return pd.DataFrame(cols)


def test_tiny_open_nulls_prices_index_untouched():
    frame = _bars(
        symbol=["BJ920227", "index_SH000300"],
        date=["2022-02-25", "2022-02-25"],
        open=[0.01, 0.01],
        high=[10.0, 4000.0],
        low=[10.0, 3900.0],
        close=[10.0, 3950.0],
        pre_close=[9.0, 3940.0],
    )
    out = sanitize_market_bars(frame)
    stock = out[out["symbol"] == "BJ920227"].iloc[0]
    assert pd.isna(stock["open"])
    assert pd.isna(stock["close"])
    index = out[out["symbol"] == "index_SH000300"].iloc[0]
    assert index["open"] == 0.01
    assert index["close"] == 3950.0


def test_adj_jump_in_batch_and_via_prev_adj():
    batch = _bars(
        symbol=["SH600000", "SH600000"],
        date=["2026-01-05", "2026-01-06"],
        adj_factor=[1.0, 10.0],
    )
    out = sanitize_market_bars(batch)
    assert out.loc[out["date"] == "2026-01-06", "adj_factor"].isna().all()
    assert float(out.loc[out["date"] == "2026-01-05", "adj_factor"].iloc[0]) == 1.0

    nxt = _bars(symbol=["SH600000"], date=["2026-01-07"], adj_factor=[80.0])
    out2 = sanitize_market_bars(nxt, prev_adj=pd.Series({"SH600000": 10.0}))
    assert out2["adj_factor"].isna().all()

    ok = sanitize_market_bars(nxt, prev_adj=pd.Series({"SH600000": 79.0}))
    assert float(ok["adj_factor"].iloc[0]) == 80.0


def test_ohlc_cross_vol_inf_and_dedup():
    frame = _bars(
        symbol=["SH600000", "SH600000", "SZ000001"],
        date=["2026-01-06", "2026-01-06", "2026-01-06"],
        open=[10.0, 10.0, 5.0],
        high=[9.0, 11.0, 5.2],
        low=[9.5, 9.8, 4.9],
        close=[10.5, 10.2, np.inf],
        vol=[0.0, 100.0, 10.0],
        amount=[1.0, 1000.0, -3.0],
        adj_factor=[1.0, 1.0, 0.0],
    )
    out = sanitize_market_bars(frame)
    assert len(out) == 2
    kept = out.set_index("symbol")
    sh = kept.loc["SH600000"]
    assert float(sh["high"]) == 11.0
    assert float(sh["vol"]) == 100.0
    sz = kept.loc["SZ000001"]
    assert pd.isna(sz["close"])
    assert pd.isna(sz["amount"])
    assert pd.isna(sz["adj_factor"])
