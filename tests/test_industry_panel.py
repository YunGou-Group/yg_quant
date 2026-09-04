"""行业 PIT 宽表：未来调样不得回写历史。"""

import pandas as pd

from FactorEvaluates.industry_panel import load_code_panel


class _Store:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def read_industry_members(self, src=None, level=None, is_new=None):
        del src, is_new
        if level and level != "l2":
            return pd.DataFrame()
        return self.frame


def test_load_code_panel_respects_in_out_dates():
    members = pd.DataFrame(
        {
            "symbol": ["SH600000", "SH600000"],
            "industry_code": ["A", "B"],
            "industry_name": ["甲", "乙"],
            "in_date": ["2026-01-01", "2026-02-01"],
            "out_date": ["2026-02-01", ""],
        }
    )
    dates = pd.Index(["2026-01-15", "2026-02-15"])
    symbols = pd.Index(["SH600000"])
    panel, labels = load_code_panel(_Store(members), dates, symbols, level="l2")
    assert panel.loc["2026-01-15", "SH600000"] == "A"
    assert panel.loc["2026-02-15", "SH600000"] == "B"
    assert labels["A"] == "甲"
