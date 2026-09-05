"""Deterministic unit tests for the attribution core.

Run with either ``pytest`` or only the Python standard library::

    python -m unittest discover -s tests -p "test_*.py"
"""

from __future__ import annotations

import json
import math
import unittest

from StrategyEngine.attribution.engine import run_attribution
from StrategyEngine.attribution.models import normalize_asset_class, normalize_asset_classes
from StrategyEngine.attribution.mtm import (
    calculate_holding_mtm,
    calculate_signed_trade_mtm,
)
from StrategyEngine.attribution.brinson import (
    multi_asset_bhb,
    stock_industry_brinson_fachler,
)
from StrategyEngine.attribution.linking import carino_link
from StrategyEngine.attribution.factor_decomp import factor_attribution
from StrategyEngine.attribution.specialized import (
    bond_attribution,
    futures_attribution,
    option_attribution,
)


class AttributionCoreTests(unittest.TestCase):
    def assertClose(self, actual, expected, tolerance=1e-12):  # noqa: N802
        self.assertTrue(
            math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance),
            f"{actual!r} != {expected!r}",
        )

    def test_asset_class_normalization(self):
        self.assertEqual(normalize_asset_class("Equities"), "stock")
        self.assertEqual(normalize_asset_class("fixed-income"), "bond")
        self.assertEqual(normalize_asset_class("期权"), "option")
        self.assertEqual(
            normalize_asset_classes("stock+ETF, futures+cash"),
            ["stock", "etf", "future", "cash"],
        )
        with self.assertRaises(ValueError):
            normalize_asset_class("real_estate")

    def test_signed_trade_and_holding_mtm_include_fees(self):
        trades = calculate_signed_trade_mtm(
            [
                {
                    "period": "2026-01-02",
                    "asset_id": "BUY",
                    "asset_class": "stock",
                    "side": "buy",
                    "quantity": 10,
                    "execution_price": 100,
                    "mark_price": 105,
                    "fee": 2,
                },
                {
                    "period": "2026-01-02",
                    "asset_id": "SELL",
                    "asset_class": "stock",
                    "side": "sell",
                    "quantity": 5,
                    "execution_price": 100,
                    "mark_price": 90,
                    "fee": 1,
                },
            ]
        )
        self.assertClose(trades.loc[0, "trade_mtm"], 48.0)
        self.assertClose(trades.loc[1, "trade_mtm"], 49.0)
        self.assertClose(trades["trade_mtm"].sum(), 97.0)

        holdings = calculate_holding_mtm(
            [
                {
                    "asset_id": "LONG",
                    "asset_class": "etf",
                    "quantity": 10,
                    "start_price": 100,
                    "end_price": 103,
                    "fees": 1,
                },
                {
                    "asset_id": "SHORT",
                    "asset_class": "future",
                    "quantity": -2,
                    "start_price": 50,
                    "end_price": 40,
                },
            ]
        )
        self.assertClose(holdings.loc[0, "holding_mtm"], 29.0)
        self.assertClose(holdings.loc[1, "holding_mtm"], 20.0)

    def test_multi_asset_bhb_reconciles_and_assigns_one_sided_to_allocation(self):
        portfolio = [
            {"period": "P1", "asset_class": "stock", "weight": 0.6, "return": 0.10},
            {"period": "P1", "asset_class": "bond", "weight": 0.4, "return": 0.05},
        ]
        benchmark = [
            {"period": "P1", "asset_class": "stock", "weight": 0.8, "return": 0.08},
            {"period": "P1", "asset_class": "cash", "weight": 0.2, "return": 0.01},
        ]
        result = multi_asset_bhb(portfolio, benchmark)
        self.assertClose(result["total_effect"].sum(), 0.014)
        for asset_class in ("bond", "cash"):
            row = result[result["asset_class"] == asset_class].iloc[0]
            self.assertClose(row["selection"], 0.0)
            self.assertClose(row["allocation"], row["active_contribution"])
        self.assertTrue(all(item["passed"] for item in result.attrs["reconciliation"]))

    def test_industry_bf_no_interaction_and_one_sided_rule(self):
        portfolio = [
            {"industry": "A", "weight": 0.7, "return": 0.10},
            {"industry": "B", "weight": 0.2, "return": 0.02},
            {"industry": "NEW", "weight": 0.1, "return": 0.15},
        ]
        benchmark = [
            {"industry": "A", "weight": 0.5, "return": 0.08},
            {"industry": "B", "weight": 0.5, "return": 0.03},
        ]
        result = stock_industry_brinson_fachler(portfolio, benchmark)
        self.assertClose(result["total_effect"].sum(), 0.034)
        new = result[result["industry"] == "NEW"].iloc[0]
        self.assertClose(new["selection"], 0.0)
        self.assertClose(new["allocation"], 0.1 * (0.15 - 0.055))
        self.assertTrue(result.attrs["reconciliation"][0]["passed"])

    def test_carino_linking_matches_compounded_active_return(self):
        returns = [
            {"period": "P1", "portfolio_return": 0.10, "benchmark_return": 0.05},
            {"period": "P2", "portfolio_return": -0.05, "benchmark_return": -0.02},
        ]
        effects = [
            {"period": "P1", "allocation": 0.02, "selection": 0.03},
            {"period": "P2", "allocation": -0.01, "selection": -0.02},
        ]
        result = carino_link(returns, effects, effect_columns=["allocation", "selection"])
        expected = (1.10 * 0.95 - 1.0) - (1.05 * 0.98 - 1.0)
        linked = result[["allocation_linked", "selection_linked"]].sum().sum()
        self.assertClose(linked, expected)
        self.assertTrue(result.attrs["reconciliation"][0]["passed"])

    def test_factor_model_includes_specific_and_reconciling_residual(self):
        exposures = [
            {"period": "P1", "factor": "Size", "active_exposure": 0.2},
            {"period": "P1", "factor": "Value", "active_exposure": -0.1},
        ]
        factor_returns = [
            {"period": "P1", "factor": "Size", "factor_return": 0.03},
            {"period": "P1", "factor": "Value", "factor_return": -0.02},
        ]
        result = factor_attribution(
            exposures,
            factor_returns,
            [{"period": "P1", "specific_contribution": 0.001}],
            actual_active_returns=[{"period": "P1", "active_return": 0.010}],
        )
        contributions = result.set_index("factor")["contribution"]
        self.assertClose(contributions["Size"], 0.006)
        self.assertClose(contributions["Value"], 0.002)
        self.assertClose(contributions["Specific"], 0.001)
        self.assertClose(contributions["Residual"], 0.001)
        self.assertClose(result["contribution"].sum(), 0.010)

    def test_bond_futures_and_option_residuals_enforce_reconciliation(self):
        bond = bond_attribution(
            [
                {
                    "asset_id": "B1",
                    "actual_return": 0.006,
                    "carry_return": 0.010,
                    "modified_duration": 2.0,
                    "risk_free_yield_change": 0.001,
                    "spread_duration": 1.5,
                    "spread_change": 0.002,
                    "convexity": 10.0,
                }
            ]
        )
        self.assertClose(bond.loc[0, "curve_return"], -0.002)
        self.assertClose(bond.loc[0, "spread_return"], -0.003)
        self.assertClose(bond.loc[0, "convexity_return"], 0.000045)
        self.assertClose(bond.loc[0, "residual_return"], 0.000955)
        self.assertClose(bond.loc[0, "explained_contribution"], 0.006)

        future = futures_attribution(
            [
                {
                    "asset_id": "F1",
                    "actual_return": 0.040,
                    "underlying_return": 0.020,
                    "basis_return": 0.001,
                    "roll_return": -0.002,
                    "collateral_return": 0.0005,
                    "leverage": 2.0,
                }
            ]
        )
        self.assertClose(future.loc[0, "leverage_return"], 0.019)
        self.assertClose(future.loc[0, "residual_return"], 0.0015)
        self.assertClose(future.loc[0, "explained_contribution"], 0.040)

        option = option_attribution(
            [
                {
                    "asset_id": "O1",
                    "actual_return": 0.120,
                    "start_value": 10.0,
                    "delta": 0.5,
                    "gamma": 0.1,
                    "vega": 2.0,
                    "theta": -0.01,
                    "rho": 1.0,
                    "underlying_price_change": 2.0,
                    "volatility_change": 0.01,
                    "rate_change": 0.001,
                    "elapsed_days": 1,
                }
            ]
        )
        self.assertClose(option.loc[0, "delta_return"], 0.10)
        self.assertClose(option.loc[0, "gamma_return"], 0.02)
        self.assertClose(option.loc[0, "vega_return"], 0.002)
        self.assertClose(option.loc[0, "theta_return"], -0.001)
        self.assertClose(option.loc[0, "rho_return"], 0.0001)
        self.assertClose(option.loc[0, "residual_return"], -0.0011)
        self.assertClose(option.loc[0, "explained_contribution"], 0.120)

    def test_engine_result_is_json_safe_and_self_describing(self):
        payload = {
            "asset_pool": ["stock"],
            "returns": [{"period": "P1", "portfolio_return": 0.08, "benchmark_return": 0.06}],
            "trades": [
                {
                    "period": "P1",
                    "asset_id": "S1",
                    "asset_class": "stock",
                    "side": "buy",
                    "quantity": 10,
                    "execution_price": 10,
                    "mark_price": 11,
                    "fees": 1,
                }
            ],
            "asset_class_attribution": {
                "portfolio": [{"period": "P1", "asset_class": "stock", "weight": 1, "return": 0.08}],
                "benchmark": [{"period": "P1", "asset_class": "stock", "weight": 1, "return": 0.06}],
            },
        }
        result = run_attribution(payload)
        # json.dumps is the contract used by the web layer.
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertIn("decimal_return", encoded)
        self.assertEqual(result["meta"]["asset_pool"], ["stock"])
        self.assertIn("asset_class", result["sections"])
        self.assertIn("linked", result["sections"])
        self.assertEqual(
            result["meta"]["schemas"]["periods"],
            [
                "period",
                "portfolio_return",
                "benchmark_return",
                "active_return",
                "trade_return",
                "holding_return",
                "total_mtm_return",
                "leverage_return",
                "benchmark_holding_return",
                "holding_active_return",
                "pnl_return_residual",
            ],
        )
        self.assertClose(result["summary"]["active_return"], 0.02)
        # Missing NAV forces the explicitly-labelled total-return fallback.
        self.assertClose(result["summary"]["holding_active_return"], 0.02)
        self.assertTrue(result["summary"]["reconciled"])
        self.assertIsNone(result["summary"]["trade_return"])
        self.assertIn(
            "missing_nav_begin_for_pnl_returns",
            {item["code"] for item in result["quality_warnings"]},
        )

    def test_engine_converts_mtm_to_returns_and_wealth_links_periods(self):
        result = run_attribution(
            {
                "asset_pool": ["stock"],
                "portfolio_daily": [
                    {
                        "period": "P1",
                        "portfolio_return": 0.10,
                        "benchmark_return": 0.08,
                        "beginning_nav": 100.0,
                    },
                    {
                        "period": "P2",
                        "portfolio_return": 0.20,
                        "benchmark_return": 0.15,
                        "beginning_nav": 110.0,
                    },
                ],
                "trades": [
                    {
                        "period": "P1",
                        "asset_id": "S1",
                        "asset_class": "stock",
                        "signed_quantity": 1,
                        "execution_price": 10,
                        "mark_price": 16,
                        "fees": 1,
                    },
                    {
                        "period": "P2",
                        "asset_id": "S1",
                        "asset_class": "stock",
                        "signed_quantity": 1,
                        "execution_price": 10,
                        "mark_price": 21,
                    },
                ],
                "holdings": [
                    {
                        "period": "P1",
                        "asset_id": "S0",
                        "asset_class": "stock",
                        "quantity": 1,
                        "start_price": 10,
                        "end_price": 13,
                    },
                    {
                        "period": "P2",
                        "asset_id": "S0",
                        "asset_class": "stock",
                        "quantity": 1,
                        "start_price": 10,
                        "end_price": 21,
                    },
                ],
            }
        )
        periods = {row["period"]: row for row in result["periods"]}
        self.assertClose(periods["P1"]["trade_return"], 0.05)
        self.assertClose(periods["P1"]["holding_return"], 0.03)
        self.assertClose(periods["P2"]["trade_return"], 0.10)
        self.assertClose(periods["P2"]["holding_return"], 0.10)
        self.assertClose(result["summary"]["trade_return"], 0.05 * 1.20 + 0.10)
        self.assertClose(result["summary"]["holding_return"], 0.03 * 1.20 + 0.10)
        # .10 and .20 compound to .32; trade + holding + residual reconcile.
        self.assertClose(
            result["summary"]["trade_return"]
            + result["summary"]["holding_return"]
            + result["summary"]["pnl_return_residual"],
            0.32,
        )

    def test_brinson_and_carino_use_holding_active_return_after_trade_peel(self):
        result = run_attribution(
            {
                "asset_pool": ["stock"],
                "returns": [
                    {
                        "period": "P1",
                        "portfolio_return": 0.08,
                        "benchmark_return": 0.05,
                        "nav_begin": 100.0,
                    }
                ],
                # Trade return is 3%; with no separate holdings ledger the
                # remaining 5% becomes holding return.
                "trades": [
                    {
                        "period": "P1",
                        "asset_id": "S1",
                        "asset_class": "stock",
                        "signed_quantity": 1,
                        "execution_price": 10,
                        "mark_price": 13,
                    }
                ],
                "asset_class_attribution": {
                    "portfolio": [
                        {"period": "P1", "asset_class": "stock", "weight": 1, "return": 0.05}
                    ],
                    "benchmark": [
                        {"period": "P1", "asset_class": "stock", "weight": 1, "return": 0.05}
                    ],
                },
            }
        )
        self.assertClose(result["periods"][0]["trade_return"], 0.03)
        self.assertClose(result["periods"][0]["holding_return"], 0.05)
        self.assertClose(result["periods"][0]["holding_active_return"], 0.0)
        self.assertClose(result["summary"]["active_return"], 0.03)
        self.assertClose(result["summary"]["holding_active_return"], 0.0)
        self.assertClose(result["summary"]["attribution_total"], 0.0)
        self.assertClose(result["summary"]["attribution_gap"], 0.0)
        self.assertEqual(result["meta"]["methodology"]["return_basis"], "holding_return")

    def test_stock_only_summary_uses_industry_bf(self):
        result = run_attribution(
            {
                "asset_pool": ["stock"],
                "returns": [
                    {
                        "period": "P1",
                        "portfolio_return": 0.076,
                        "benchmark_return": 0.055,
                        "nav_begin": 100.0,
                    }
                ],
                "stock": {
                    "industry": {
                        "portfolio": [
                            {"period": "P1", "industry": "A", "weight": 0.7, "return": 0.10},
                            {"period": "P1", "industry": "B", "weight": 0.3, "return": 0.02},
                        ],
                        "benchmark": [
                            {"period": "P1", "industry": "A", "weight": 0.5, "return": 0.08},
                            {"period": "P1", "industry": "B", "weight": 0.5, "return": 0.03},
                        ],
                    }
                },
            }
        )
        self.assertEqual(result["meta"]["methodology"]["allocation_selection"], "industry")
        self.assertIn("industry", result["sections"])
        self.assertIn("linked", result["sections"])
        self.assertClose(result["summary"]["allocation_effect"], 0.010)
        self.assertClose(result["summary"]["selection_effect"], 0.011)
        self.assertClose(result["summary"]["holding_active_return"], 0.021)

    def test_engine_accepts_combined_dict_of_lists(self):
        result = run_attribution(
            {
                "asset_pool": "stock+bond",
                "returns": {
                    "period": ["P1"],
                    "portfolio_return": [0.07],
                    "benchmark_return": [0.05],
                },
                "asset_class_attribution": {
                    "period": ["P1", "P1"],
                    "asset_class": ["stock", "bond"],
                    "portfolio_weight": [0.6, 0.4],
                    "benchmark_weight": [0.5, 0.5],
                    "portfolio_return": [0.10, 0.025],
                    "benchmark_return": [0.08, 0.02],
                },
            }
        )
        self.assertIn("asset_class", result["sections"])
        rows = result["sections"]["asset_class"]["rows"]
        self.assertEqual({row["asset_class"] for row in rows}, {"stock", "bond"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
