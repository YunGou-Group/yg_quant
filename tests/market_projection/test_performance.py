from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from DataStore import MarketProjectionReader
from DailyUpdates.storage import SQLiteStorage
from DailyUpdates.storage.market_projection import MarketProjectionWriter


@pytest.mark.skipif(
    os.getenv("YG_QUANT_PERF") != "1",
    reason="set YG_QUANT_PERF=1 to run projection benchmark",
)
def test_projection_cold_read_benchmark(tmp_path):
    n_symbols = 250
    n_days = 250
    symbols = [f"SH{600000 + i:06d}" for i in range(n_symbols)]
    dates = pd.bdate_range("2023-01-02", periods=n_days).strftime("%Y-%m-%d")
    db = SQLiteStorage(str(tmp_path / "market.db"))
    db.upsert_stock_basic(
        pd.DataFrame(
            {
                "symbol": symbols,
                "name": symbols,
                "list_date": ["20200101"] * n_symbols,
                "list_status": ["L"] * n_symbols,
            }
        )
    )
    rng = np.random.default_rng(7)
    for block in np.array_split(dates, 10):
        index = pd.MultiIndex.from_product(
            [symbols, block], names=["ts_code", "trade_date"]
        )
        frame = index.to_frame(index=False)
        for field in ("open", "close", "vol", "amount"):
            frame[field] = rng.random(len(frame)) * 100
        frame["adj_factor"] = 1.0
        db.upsert_market_data(frame)
    root = tmp_path / "projection"
    MarketProjectionWriter(db, str(root)).rebuild(chunk_days=25)

    start = time.perf_counter()
    raw = db.read_market_data(fields=["open"], ordered=False)
    sqlite_panel = raw.pivot(index="trade_date", columns="ts_code", values="open")
    sqlite_seconds = time.perf_counter() - start

    start = time.perf_counter()
    projected = MarketProjectionReader(
        str(root), sidecar_storage=db
    ).read_panel("open")
    projection_seconds = time.perf_counter() - start

    pd.testing.assert_frame_equal(projected, sqlite_panel, check_names=False)
    print(
        f"sqlite+pivot={sqlite_seconds:.3f}s projection={projection_seconds:.3f}s "
        f"speedup={sqlite_seconds / max(projection_seconds, 1e-9):.2f}x"
    )
