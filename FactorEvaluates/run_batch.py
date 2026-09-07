#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全库因子评估 CLI。默认不含 X_T（exposure/pure_ic/attribution），加 ``--xt`` 才开。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from yg_quant_repo import relaunch_as_module

relaunch_as_module("FactorEvaluates.run_batch")

import argparse
import json

from Universes.catalog import names as universe_names

from .batch.batch_eval_runner import BatchEvalRunner
from .metric_discoverer import MetricDiscoverer

XT_METRICS: Tuple[str, ...] = ("exposure", "pure_ic", "attribution")


def _parse_csv(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    items = [part.strip() for part in str(text).split(",") if part.strip()]
    return items or None


def resolve_metrics(
    *,
    metrics: Optional[str] = None,
    exclude: Optional[str] = None,
    xt: bool = False,
) -> Tuple[List[str], List[str]]:
    """返回 (要跑的指标, 因未开 --xt 而跳过的 X_T 指标)。默认 = single + library。"""
    selected = _parse_csv(metrics)
    excluded = set(_parse_csv(exclude) or ())
    disc = MetricDiscoverer()
    if selected is None:
        names = [m.get_name() for m in disc.metrics_for("single")]
        names.extend(m.get_name() for m in disc.metrics_for("library"))
    else:
        names = list(selected)
    names = [name for name in names if name not in excluded]
    skipped_xt = [name for name in names if name in XT_METRICS]
    if xt:
        out = names
        skipped_xt = []
    else:
        out = [name for name in names if name not in XT_METRICS]
    if not out:
        raise SystemExit(
            "过滤后没有可跑的指标。"
            "只跑暴露/归因时请加 --xt，例如: --xt --metrics exposure,pure_ic,attribution"
        )
    return out, skipped_xt


def _xt_in(names: Sequence[str]) -> List[str]:
    return [name for name in names if name in XT_METRICS]


def main() -> None:
    parser = argparse.ArgumentParser(description="全库因子评估")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--universe", default="all", choices=universe_names())
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="每批装入内存的因子个数",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=0,
        help="日度进程数；0=按内存自动选，>0=按指定值",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--metrics",
        default=None,
        help="逗号分隔指标名；默认全部（仍默认不含 X_T 三项）",
    )
    parser.add_argument(
        "--exclude-metrics",
        default=None,
        help="逗号分隔，从所选/默认集合里去掉",
    )
    parser.add_argument(
        "--xt",
        action="store_true",
        help="启用 exposure/pure_ic/attribution 并组装 X_T",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的 run_id 目录",
    )
    args = parser.parse_args()
    xt = bool(args.xt)
    metric_names, skipped_xt = resolve_metrics(
        metrics=args.metrics,
        exclude=args.exclude_metrics,
        xt=xt,
    )
    xt_running = _xt_in(metric_names)

    def progress(phase, detail, pct, message):
        print(f"[batch] {pct:5.0%} {phase} {message}", flush=True)

    if xt_running:
        print(
            f"[batch] --xt：组装 X_T，计算 {', '.join(xt_running)}",
            flush=True,
        )
    elif xt:
        print(
            "[batch] 已加 --xt，但所选指标不含 "
            + "/".join(XT_METRICS)
            + "，不组装 X_T",
            flush=True,
        )
    else:
        print(
            "[batch] 跳过 X_T（"
            + ", ".join(XT_METRICS)
            + "）；需要时加 --xt",
            flush=True,
        )
        if skipped_xt:
            print(f"[batch] 已去掉: {', '.join(skipped_xt)}", flush=True)

    print(f"[batch] metrics={len(metric_names)}: {', '.join(metric_names)}", flush=True)

    runner = BatchEvalRunner(batch_size=args.batch_size, n_workers=args.n_workers)
    request = {
        "start": args.start,
        "end": args.end,
        "universe": args.universe,
        "horizon": args.horizon,
        "run_id": args.run_id,
        "batch_size": args.batch_size,
        "n_workers": args.n_workers,
        "overwrite": bool(args.overwrite),
        "metrics": metric_names,
        "xt": bool(xt_running),
    }
    result = runner.run(request, progress=progress)
    meta = result.get("meta") or {}
    print(
        f"[batch] has_xt={bool(meta.get('has_xt'))} "
        f"n_workers={meta.get('n_workers')} n_dates={meta.get('n_dates')}",
        flush=True,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
