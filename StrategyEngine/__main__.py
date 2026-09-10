#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI：python -m StrategyEngine --mode backtest --start 2010-01-01 --factor alpha001"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from yg_quant_repo import relaunch_as_module

relaunch_as_module("StrategyEngine")

import argparse
from datetime import datetime

from FactorEvaluates.market_panel_loader import HS300_SYMBOL, MarketPanelLoader
from yg_quant_repo import default_data_dir

from .backtest import Engine, summarize, trades_frame, write_run_snapshot
from .allocators import REGISTRY
from Strategies import HOLD_N

from Universes.catalog import names as universe_names

from .catalog import by_name
from .live import default_order_path, plan_live
from .run import build, _run_params


def main() -> None:
    names = by_name()
    if not names:
        raise SystemExit("Strategies/ 下没有可发现的策略（需要 name + from_cli + panel_kwargs + run_tag）")
    default = "topk" if "topk" in names else sorted(names)[0]
    parser = argparse.ArgumentParser(
        description="日频策略：回测撮合、实盘 QMT JSON，或读取已有回测做业绩归因"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["backtest", "live", "attribute"],
        help="必填：backtest 走开盘撮合；live 写出 QMT JSON；attribute 读某次回测做业绩归因",
    )
    parser.add_argument(
        "--strategy",
        default=default,
        choices=sorted(names),
        help="策略名，对应 Strategies/ 下已发现的类",
    )
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--universe",
        default="all",
        choices=universe_names(),
        help="Universes/ 已注册股票池，asof PIT",
    )
    parser.add_argument(
        "--factor",
        default=None,
        help="topk：单因子名；multifactor / icir：逗号分隔，如 a,b,-c（前缀 - 取负）",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="topk / multifactor / icir 持仓数，或小市值候选数",
    )
    parser.add_argument(
        "--rebalance",
        default="daily",
        help="topk / multifactor / icir 调仓：daily、weekly（周五收盘），或 N 个交易日如 5 / 20 / every20",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="multifactor / icir：回归或 ICIR 回看交易日数，默认 60",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="multifactor / icir：远期收益持有交易日，默认 5",
    )
    parser.add_argument("--hold", type=int, default=HOLD_N, help="小市值：剔除最小后取到第 N 名")
    parser.add_argument("--anti-tail", action="store_true", help="小市值防尾声")
    parser.add_argument(
        "--allocator",
        default="equal",
        choices=sorted(REGISTRY),
        help="仓位分配：选出候选后分配权重，默认等权",
    )
    parser.add_argument(
        "--allocator-lookback",
        type=int,
        default=252,
        help="优化器收益回看交易日，默认 252（与 doc 年频样本一致）",
    )
    parser.add_argument("--max-weight", type=float, default=1.0, help="单票上限，默认 1（不限制）；doc 网格常用 0.2")
    parser.add_argument("--risk-free-rate", type=float, default=0.02, help="max_sharpe 年化无风险利率，默认 0.02")
    parser.add_argument("--risk-aversion", type=float, default=1.0, help="max_quadratic_utility 风险厌恶 δ，默认 1")
    parser.add_argument(
        "--target-return",
        type=float,
        default=None,
        help="efficient_return 目标年化收益；默认取候选 μ 的 60%% 分位",
    )
    parser.add_argument(
        "--target-volatility",
        type=float,
        default=None,
        help="efficient_risk 目标年化波动；默认等权波动 × 1.15",
    )
    parser.add_argument("--l2-gamma", type=float, default=0.1, help="multiobjective / min_l2 的 L2 系数，默认 0.1")
    parser.add_argument("--tc-rate", type=float, default=0.001, help="multiobjective 换手惩罚，默认 0.001")
    parser.add_argument("--tail-confidence", type=float, default=0.95, help="min_cvar / min_cdar 的 β，默认 0.95")
    parser.add_argument(
        "--mu-model",
        default="geometric",
        choices=["geometric", "arithmetic", "ema"],
        help="需要 μ 的优化器：几何 / 算术 / EMA，默认几何（pypfopt mean_historical_return）",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=1_000_000.0,
        help="回测初始资金 / 实盘分配给该策略的资金（换算手数）",
    )
    parser.add_argument("--account", default="", help="实盘：写入 JSON 的 QMT 资金账号")
    parser.add_argument(
        "--run",
        default=None,
        help="attribute：回测快照 id（strategy_runs 下 json 名）或 JSON 路径",
    )
    parser.add_argument(
        "--qmt-json",
        default=None,
        help="实盘对接 JSON 路径，默认 data/strategy_runs/qmt_orders.json",
    )
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--stamp", type=float, default=0.0005)
    parser.add_argument("--out", default=None, help="回测净值 CSV / 归因 JSON 路径")
    parser.add_argument(
        "--benchmark",
        default=HS300_SYMBOL,
        help="基准指数代码，默认沪深300；空字符串关闭",
    )
    parser.add_argument("--no-benchmark", action="store_true", help="关闭基准对照")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    if args.mode == "attribute":
        _run_attribute(args)
        return
    store, strategy, tag = build(args)
    if args.mode == "live":
        _run_live(args, store, strategy, tag)
        return
    print(f"开始回测 strategy={args.strategy} {args.start} ~ {args.end or '最新'}", flush=True)
    print("面板已加载，开始逐日撮合", flush=True)
    result = Engine(
        store,
        initial_cash=args.cash,
        commission=args.commission,
        stamp=args.stamp,
    ).run(strategy)
    equity = result.equity()
    out = Path(args.out) if args.out else (
        default_data_dir() / "strategy_runs" / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    equity.to_csv(out, encoding="utf-8")
    trades = trades_frame(result)
    trades_path = out.with_name(f"{out.stem}_trades.csv")
    trades.to_csv(trades_path, index=False, encoding="utf-8")
    bench = None
    symbol = "" if args.no_benchmark else str(args.benchmark or "").strip()
    if symbol and symbol.lower() not in {"none", "off"}:
        try:
            bench = MarketPanelLoader().load_index_close(symbol)
        except Exception:
            logging.getLogger("StrategyEngine").warning("基准 %s 加载失败，跳过超额", symbol)
            bench = None
    stats = summarize(result, benchmark=bench, risk_free_rate=args.risk_free_rate)
    snap_path = out.with_suffix(".json")
    write_run_snapshot(
        snap_path,
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "tag": tag,
            "strategy": args.strategy,
            "allocator": args.allocator,
            "files": {"equity": str(out), "trades": str(trades_path)},
            "params": _run_params(args),
            "metrics": stats,
        },
    )
    parts = [
        f"净值 {out}",
        f"成交 {trades_path}",
        f"快照 {snap_path}",
        f"年化 {_pct(stats.get('annualized'))}",
        f"回撤 {_pct(stats.get('max_drawdown'))}",
        f"波动 {_pct(stats.get('volatility'))}",
        f"夏普 {_num(stats.get('sharpe'))}",
        f"换手 {_pct(stats.get('turnover_mean'))}",
    ]
    if stats.get("fee_total") is not None:
        parts.append(f"费用 {stats['fee_total']:.2f}")
        if stats.get("gross_return") is not None:
            parts.append(f"毛收益 {_pct(stats.get('gross_return'))}")
    if stats.get("excess_annualized") is not None:
        parts.append(f"超额 {_pct(stats['excess_annualized'])}")
    print("  ".join(parts))


def _run_attribute(args) -> None:
    if not str(args.run or "").strip():
        raise SystemExit("归因必须用 --run 指定回测快照 id 或 JSON 路径")
    print(f"业绩归因 run={args.run}", flush=True)
    from .attribution import attribute_snapshot

    report = attribute_snapshot(
        args.run,
        out=Path(args.out) if args.out else None,
    )
    dest = (report.get("files") or {}).get("attribution")
    brief = report.get("summary") or {}
    parts = [
        f"归因 {dest}",
        f"status={brief.get('status')}",
        f"warnings={brief.get('warnings')}",
    ]
    tables = brief.get("tables") or []
    if tables:
        parts.append("表 " + ",".join(str(x) for x in tables))
    print("  ".join(parts), flush=True)


def _run_live(args, store, strategy, tag) -> None:
    if float(args.cash) <= 0:
        raise SystemExit("实盘必须用 --cash 指定分配给该策略的资金")
    print(
        f"实盘对接 strategy={args.strategy} allocator={args.allocator} "
        f"资金={args.cash:.0f} {args.start} ~ {args.end or '最新'}",
        flush=True,
    )
    plan = plan_live(
        store,
        strategy,
        capital=args.cash,
        strategy_name=args.strategy,
        allocator=args.allocator,
        account=args.account,
        params=_run_params(args),
    )
    dest = Path(args.qmt_json) if args.qmt_json else default_order_path()
    plan.write(dest)
    stamp = dest.with_name(
        f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_qmt.json"
    )
    plan.write(stamp)
    print(
        f"对接 {dest}  快照 {stamp}  asof={plan.asof}  "
        f"execute_on={plan.execute_on}  目标 {len(plan.holdings)} 只  "
        f"跳过 {len(plan.skipped)}",
        flush=True,
    )


def _pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2%}"


def _num(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


if __name__ == "__main__":
    main()
