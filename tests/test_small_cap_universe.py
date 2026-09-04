"""小市值多指数成分：逐指数 as-of 再并集。"""

from Strategies.small_cap import _constituent_snap, _universe_asof
import pandas as pd


def test_universe_asof_unions_stale_index_snapshots():
    a = {"2026-01-31": {"SH600000"}}
    b = {"2026-02-01": {"SZ000001"}}
    assert _universe_asof([a, b], "2026-02-02") == {"SH600000", "SZ000001"}
    assert _universe_asof([a, b], "2026-01-31") == {"SH600000"}
    assert _universe_asof([a, b], "2026-01-15") == set()


def test_constituent_snap_groups_by_index_date():
    frame = pd.DataFrame(
        {
            "trade_date": ["20260131", "20260201"],
            "symbol": ["SH600000", "SZ000001"],
        }
    )
    snap = _constituent_snap(frame)
    assert snap["2026-01-31"] == {"SH600000"}
    assert snap["2026-02-01"] == {"SZ000001"}
