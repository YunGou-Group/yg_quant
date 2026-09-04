"""QMT 对接：代码转换、原子写、计划过期与账户隔离。"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from StrategyEngine.backtest.panel import PanelStore
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.live import from_qmt_code, lots_from_weight, plan_from_target, to_qmt_code
from StrategyEngine.live.planner import next_session, plan_live


def _json_exec():
    path = Path(__file__).resolve().parents[1] / "qmt_scripts" / "json_exec.py"
    spec = importlib.util.spec_from_file_location("json_exec_client", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_to_qmt_code():
    assert to_qmt_code("SH600000") == "600000.SH"
    assert to_qmt_code("600000.SH") == "600000.SH"
    assert to_qmt_code("SZ000001") == "000001.SZ"
    assert to_qmt_code("000001") == "000001.SZ"
    assert from_qmt_code("600000.SH") == "SH600000"


def test_lots_from_weight_floors_to_board_lot():
    assert lots_from_weight(0.1, 100_000, 10.0) == 1000
    assert lots_from_weight(0.1, 1000, 10.0) == 0
    assert lots_from_weight(0.5, 1_000_000, 33.3) == 15000


def test_plan_json_uses_qmt_codes(tmp_path: Path):
    target = TargetHoldings(
        asof="2026-09-01",
        execute_on="2026-09-02",
        weights={"SH600000": 0.4, "SZ000001": 0.6},
    )
    plan = plan_from_target(
        target,
        {"SH600000": 10.0, "SZ000001": 20.0},
        capital=100_000,
        strategy="small_cap",
        allocator="equal",
        account="acc1",
    )
    assert [row.code for row in plan.orders] == ["600000.SH", "000001.SZ"]
    assert plan.orders[0].volume == 4000
    assert plan.orders[1].volume == 3000
    path = tmp_path / "qmt_orders.json"
    plan.write(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["execute_on"] == "2026-09-02"
    assert payload["holdings"][0]["code"] == "600000.SH"
    assert not path.with_name(path.name + ".tmp").exists()


def test_next_session_skips_asof():
    days = ["2026-09-01", "2026-09-02", "2026-09-03"]
    assert next_session("2026-09-01", days) == "2026-09-02"
    assert next_session("2026-09-03", days) is None


def test_rebalance_does_not_sell_unknown_holdings():
    je = _json_exec()
    payload = {
        "holdings": [{"code": "600000.SH", "volume": 100}],
    }
    current = {"600000.SH": 100, "000001.SZ": 200}
    target = je.target_from_payload(payload)
    deltas = je.rebalance_deltas(target, current)
    assert deltas == []
    payload = {"holdings": [{"code": "600000.SH", "volume": 200}]}
    deltas = je.rebalance_deltas(je.target_from_payload(payload), current)
    assert deltas == [("600000.SH", 100)]


def test_execute_on_missing_is_rejected():
    je = _json_exec()
    today = "20260904"
    assert je.execute_on_today({"execute_on": "2026-09-04"}, today=today) is True
    assert je.execute_on_today({"execute_on": "20260904"}, today=today) is True
    assert je.execute_on_today({"execute_on": "2026-09-03"}, today=today) is False
    assert je.execute_on_today({}, today=today) is False
    assert je.execute_on_today({"execute_on": ""}, today=today) is False
    assert je.execute_on_today({"execute_on": None}, today=today) is False


def test_plan_live_requires_execute_on(monkeypatch):
    monkeypatch.setattr(
        "StrategyEngine.live.planner.next_session", lambda *args, **kwargs: None
    )
    open_px = np.array([[10.0, 10.0]], dtype=np.float64)
    store = PanelStore.from_arrays(
        ["2026-09-03"],
        ["SH600000", "SZ000001"],
        open=open_px,
        close=open_px,
        mask=np.ones((1, 2), dtype=bool),
    )

    class _Hold:
        def score(self, ctx):
            w = np.zeros(len(ctx.symbols))
            w[0] = 1.0
            return TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, w)

    with pytest.raises(ValueError, match="execute_on"):
        plan_live(
            store,
            _Hold(),
            capital=100_000,
            strategy_name="hold",
            allocator="equal",
        )
