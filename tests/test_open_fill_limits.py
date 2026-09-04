"""涨跌停：只有成交价触及限价才拒单，不扩大一档。"""

import numpy as np

from StrategyEngine.backtest.executor import OpenFillExecutor
from StrategyEngine.holdings import TargetHoldings


def _fill(px, *, shares=None, cash=11_000.0, up=11.0, down=9.0, target_w=1.0):
    exe = OpenFillExecutor(commission=0.0, stamp=0.0)
    if shares is None:
        shares = np.zeros(1, dtype=np.float64)
    else:
        shares = np.asarray(shares, dtype=np.float64)
    target = TargetHoldings(
        asof="2026-09-01",
        execute_on="2026-09-02",
        weights={"AAA": target_w} if target_w > 0 else {},
    )
    return exe.fill(
        shares,
        cash,
        target,
        ["AAA"],
        np.array([px], dtype=np.float64),
        up_limit=np.array([up], dtype=np.float64),
        down_limit=np.array([down], dtype=np.float64),
        date="2026-09-02",
    )


def test_one_tick_below_up_limit_can_buy():
    shares, _cash, report = _fill(10.99)
    assert shares[0] >= 100
    assert report.reason[0] != "limit_up"


def test_open_at_up_limit_blocks_buy():
    shares, _cash, report = _fill(11.00)
    assert shares[0] == 0
    assert report.reason[0] == "limit_up"


def test_one_tick_above_down_limit_can_sell():
    shares, _cash, report = _fill(9.01, shares=[1000.0], cash=0.0, target_w=0.0)
    assert shares[0] == 0
    assert report.reason[0] != "limit_down"


def test_open_at_down_limit_blocks_sell():
    shares, _cash, report = _fill(9.00, shares=[1000.0], cash=0.0, target_w=0.0)
    assert shares[0] == 1000
    assert report.reason[0] == "limit_down"
