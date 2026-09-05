"""事后归因：从成交还原持仓，并调用 run_attribution。"""

import pandas as pd

from StrategyEngine.attribution import ledger
from StrategyEngine.attribution.engine import run_attribution
from StrategyEngine.fill import FillReport


def test_positions_accumulate_signed_fills():
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    trades = pd.DataFrame(
        [
            {"date": "2026-01-06", "symbol": "SH600000", "side": "buy", "filled": 1000, "price": 10.0, "fee": 3.0},
            {"date": "2026-01-07", "symbol": "SH600000", "side": "sell", "filled": 400, "price": 11.0, "fee": 2.0},
        ]
    )
    pos = ledger.positions_by_date(dates, trades)
    assert pos["2026-01-05"] == {}
    assert pos["2026-01-06"] == {"SH600000": 1000.0}
    assert pos["2026-01-07"] == {"SH600000": 600.0}
    start = ledger.previous_positions(dates, pos)
    assert start["2026-01-06"] == {}
    assert start["2026-01-07"] == {"SH600000": 1000.0}


def test_close_ledger_and_brison_trade_holding():
    dates = ["2026-01-05", "2026-01-06"]
    trades = pd.DataFrame(
        [
            {
                "date": "2026-01-06",
                "symbol": "SH600000",
                "side": "buy",
                "filled": 1000,
                "price": 10.0,
                "fee": 3.0,
            }
        ]
    )
    pos = ledger.positions_by_date(dates, trades)
    start = ledger.previous_positions(dates, pos)
    close = {
        "2026-01-05": {"SH600000": 9.5},
        "2026-01-06": {"SH600000": 10.5},
    }
    cash = {"2026-01-05": 100_000.0, "2026-01-06": 89_997.0}
    returns = ledger.period_returns(
        dates,
        cash,
        pos,
        close,
        initial_cash=100_000.0,
        benchmark_close={"2026-01-05": 100.0, "2026-01-06": 101.0},
    )
    payload = {
        "asset_pool": ["stock"],
        "returns": returns,
        "trades": ledger.trade_rows(trades, close),
        "holdings": ledger.holding_rows(dates, start, close),
    }
    result = run_attribution(payload)
    assert result["pnl"]["totals"]["trade_mtm"] > 0
    # 买入 1000 股，开盘 10 → 收盘 10.5，费用 3
    assert abs(result["pnl"]["totals"]["trade_mtm"] - (1000 * 0.5 - 3)) < 1e-6
    assert result["pnl"]["totals"]["holding_mtm"] == 0.0


def test_web_view_omits_trade_ticks():
    from StrategyEngine.attribution.evaluate import web_view

    view = web_view(
        {
            "run_id": "demo",
            "notes": ["收盘盯市"],
            "attribution": {
                "status": "ok",
                "summary": {"portfolio_return": 0.01, "trade_mtm": 10},
                "pnl": {
                    "trades": [{"asset_id": "SH600000"}] * 3,
                    "by_period": [
                        {"period": "2026-01-06", "trade_return": 0.001, "holding_return": 0.0}
                    ],
                },
                "sections": {
                    "industry": {
                        "title": "行业",
                        "rows": [
                            {"industry": "银行", "allocation": 0.01, "selection": -0.002, "total_effect": 0.008},
                            {"industry": "银行", "allocation": 0.01, "selection": 0.0, "total_effect": 0.01},
                        ],
                    }
                },
                "quality_warnings": [{"code": "x", "message": "info", "severity": "info"}],
            },
        }
    )
    assert "trades" not in view
    assert view["pnl"][0]["period"] == "2026-01-06"
    assert view["industry"]["rows"][0]["industry"] == "银行"
    assert abs(view["industry"]["rows"][0]["allocation"] - 0.02) < 1e-12
    assert view["warnings"][0]["code"] == "x"
    labels = [row["label"] for row in view["returns_tree"] if row.get("kind") != "header"]
    assert labels[0] == "总收益"
    assert "股票配置收益" in labels
    assert "风格偏好" not in labels
    assert view["factors"]["rows"] == []
    assert {row["group"] for row in view["factors"]["groups"]} == {
        "风格偏好",
        "行业偏好",
        "市场联动",
        "特异收益",
    }


def test_web_view_builds_factor_onion():
    from StrategyEngine.attribution.evaluate import web_view

    view = web_view(
        {
            "run_id": "demo",
            "attribution": {
                "status": "ok",
                "summary": {
                    "portfolio_return": 0.09,
                    "benchmark_return": 0.06,
                    "active_return": 0.03,
                    "trade_return": 0.004,
                    "holding_return": 0.086,
                    "leverage_return": 0.0,
                    "holding_active_return": 0.032,
                    "benchmark_holding_return": 0.054,
                    "allocation_effect": 0.01,
                    "selection_effect": 0.022,
                },
                "sections": {
                    "factor": {
                        "title": "股票因子归因",
                        "rows": [
                            {
                                "period": "2026-01-06",
                                "factor": "momentum",
                                "portfolio_exposure": 0.3,
                                "benchmark_exposure": 0.1,
                                "active_exposure": 0.2,
                                "factor_return": 0.01,
                                "contribution": 0.002,
                            },
                            {
                                "period": "2026-01-07",
                                "factor": "momentum",
                                "portfolio_exposure": 0.5,
                                "benchmark_exposure": 0.1,
                                "active_exposure": 0.4,
                                "factor_return": 0.02,
                                "contribution": 0.008,
                            },
                            {
                                "period": "2026-01-06",
                                "factor": "comovement",
                                "contribution": 0.001,
                            },
                            {
                                "period": "2026-01-06",
                                "factor": "industry:银行",
                                "contribution": 0.003,
                            },
                            {
                                "period": "2026-01-06",
                                "factor": "Specific",
                                "contribution": 0.004,
                            },
                        ],
                    }
                },
            },
        }
    )
    labels = [row["label"] for row in view["returns_tree"]]
    assert labels[:4] == ["总收益", "交易收益", "杠杆收益", "持仓收益"]
    assert "动量" not in labels
    rows = {row["key"]: row for row in view["factors"]["rows"]}
    assert rows["momentum"]["group"] == "风格偏好"
    assert abs(rows["momentum"]["contribution"] - 0.01) < 1e-12
    assert abs(rows["momentum"]["portfolio_exposure"] - 0.4) < 1e-12
    groups = {row["group"]: row["contribution"] for row in view["factors"]["groups"]}
    assert abs(groups["风格偏好"] - 0.01) < 1e-12
    assert abs(groups["行业偏好"] - 0.003) < 1e-12
    assert abs(groups["市场联动"] - 0.001) < 1e-12
    assert abs(groups["特异收益"] - 0.004) < 1e-12


def test_beginning_weights_match_industry_nav():
    dates = ["2026-01-05", "2026-01-06"]
    start = {"2026-01-05": {}, "2026-01-06": {"SH600000": 1000.0}}
    close = {
        "2026-01-05": {"SH600000": 10.0},
        "2026-01-06": {"SH600000": 11.0},
    }
    cash = {"2026-01-05": 90_000.0, "2026-01-06": 90_000.0}
    weights = ledger.beginning_weights(dates, start, close, cash)
    assert abs(weights["2026-01-06"]["SH600000"] - 0.1) < 1e-12


def test_factor_payload_feeds_brison_engine():
    payload = {
        "asset_pool": ["stock"],
        "returns": [
            {
                "period": "2026-01-06",
                "portfolio_return": 0.02,
                "benchmark_return": 0.01,
                "holding_return": 0.02,
                "benchmark_holding_return": 0.01,
                "trade_return": 0.0,
                "leverage_return": 0.0,
                "nav_begin": 100_000.0,
            }
        ],
        "stock": {
            "factor": {
                "exposures": [
                    {
                        "period": "2026-01-06",
                        "factor": "momentum",
                        "portfolio_exposure": 0.5,
                        "benchmark_exposure": 0.1,
                        "active_exposure": 0.4,
                    }
                ],
                "factor_returns": [
                    {"period": "2026-01-06", "factor": "momentum", "factor_return": 0.02}
                ],
                "specific_returns": [{"period": "2026-01-06", "specific_contribution": 0.002}],
            }
        },
    }
    result = run_attribution(payload)
    assert "factor" in result["sections"]
    contrib = {
        row["factor"]: row["contribution"] for row in result["sections"]["factor"]["rows"]
    }
    assert abs(contrib["momentum"] - 0.008) < 1e-12
    assert abs(contrib["Specific"] - 0.002) < 1e-12
    codes = {item["code"] for item in result["quality_warnings"]}
    assert "missing_factor_data" not in codes


def test_web_view_fills_tree_from_industry_when_summary_effects_are_zero():
    from StrategyEngine.attribution.evaluate import web_view

    view = web_view(
        {
            "run_id": "demo",
            "attribution": {
                "status": "ok",
                "summary": {
                    "portfolio_return": 0.076,
                    "benchmark_return": 0.055,
                    "holding_active_return": 0.021,
                    "allocation_effect": 0.0,
                    "selection_effect": 0.0,
                },
                "periods": [
                    {
                        "period": "P1",
                        "holding_return": 0.076,
                        "benchmark_holding_return": 0.055,
                        "portfolio_return": 0.076,
                        "benchmark_return": 0.055,
                    }
                ],
                "sections": {
                    "industry": {
                        "title": "行业",
                        "rows": [
                            {
                                "period": "P1",
                                "industry": "A",
                                "allocation": 0.005,
                                "selection": 0.014,
                                "total_effect": 0.019,
                            },
                            {
                                "period": "P1",
                                "industry": "B",
                                "allocation": 0.005,
                                "selection": -0.003,
                                "total_effect": 0.002,
                            },
                        ],
                    },
                    "factor": {"rows": []},
                },
            },
        }
    )
    tree = {row["label"]: row["value"] for row in view["returns_tree"]}
    assert abs(tree["股票配置收益"] - 0.010) < 1e-12
    assert abs(tree["股票选择收益"] - 0.011) < 1e-12
    assert abs(view["summary"]["allocation_effect"] - 0.010) < 1e-12


def test_returns_tree_children_sum_to_parents():
    from StrategyEngine.attribution.tree import returns_tree

    tree = returns_tree(
        {},
        periods=[
            {
                "period": "P1",
                "portfolio_return": 0.10,
                "trade_return": 0.02,
                "holding_return": 0.08,
                "leverage_return": 0.0,
                "benchmark_return": 0.05,
                "benchmark_holding_return": 0.05,
            },
            {
                "period": "P2",
                "portfolio_return": 0.20,
                "trade_return": 0.05,
                "holding_return": 0.15,
                "leverage_return": 0.0,
                "benchmark_return": 0.10,
                "benchmark_holding_return": 0.10,
            },
        ],
        effect_rows=[
            {"period": "P1", "industry": "A", "allocation": 0.01, "selection": 0.02},
            {"period": "P2", "industry": "A", "allocation": 0.02, "selection": 0.03},
        ],
    )
    values = {row["label"]: row["value"] for row in tree}
    assert abs(values["总收益"] - (values["交易收益"] + values["杠杆收益"] + values["持仓收益"])) < 1e-12
    assert abs(values["持仓收益"] - (values["主动收益"] + values["基准持仓收益"])) < 1e-12
    assert abs(
        values["主动收益"]
        - (values["股票配置收益"] + values["股票选择收益"] + values["归因残差"])
    ) < 1e-12
    assert abs(values["基准持仓收益"] - (values["基准收益"] + values["穿透效应"])) < 1e-12


def test_onion_tree_matches_rqpattr_levels():
    from StrategyEngine.attribution.tree import returns_tree

    tree = returns_tree(
        {
            "portfolio_return": 0.0934,
            "benchmark_return": 0.0612,
            "active_return": 0.0322,
            "trade_return": 0.0041,
            "holding_return": 0.0889,
            "leverage_return": 0.0004,
            "holding_active_return": 0.0322,
            "benchmark_holding_return": 0.0567,
            "allocation_effect": 0.0101,
            "selection_effect": 0.0211,
        },
    )
    labels = [row["label"] for row in tree]
    assert labels == [
        "总收益",
        "交易收益",
        "杠杆收益",
        "持仓收益",
        "主动收益",
        "股票配置收益",
        "股票选择收益",
        "归因残差",
        "基准持仓收益",
        "基准收益",
        "穿透效应",
    ]
    pen = next(row for row in tree if row["label"] == "穿透效应")
    assert abs(pen["value"] - (0.0567 - 0.0612)) < 1e-12


def test_fill_report_allocates_fee():
    report = FillReport(
        date="2026-01-06",
        symbols=["SH600000", "SZ000001"],
        intended=[1000.0, 500.0],
        filled=[1000.0, 500.0],
        price=[10.0, 20.0],
        fee=30.0,
    )
    rows = report.to_rows()
    fees = {row["symbol"]: row["fee"] for row in rows}
    assert abs(fees["SH600000"] - 15.0) < 1e-9
    assert abs(fees["SZ000001"] - 15.0) < 1e-9


def test_report_requires_factor_section():
    from StrategyEngine.attribution.evaluate import is_current_report

    assert is_current_report(
        {"attribution": {"sections": {"factor": {}, "industry": {}}}}
    )
    assert not is_current_report({"attribution": {"sections": {"industry": {}}}})
    assert not is_current_report({})


def test_attribution_engine_lives_in_package():
    from pathlib import Path

    from StrategyEngine.attribution import engine as pkg

    root = Path(pkg.__file__).resolve().parent
    assert (root / "engine.py").is_file()
    assert (root / "mtm.py").is_file()
    assert root.name == "attribution"
    assert root.parent.name == "StrategyEngine"
