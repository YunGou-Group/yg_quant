#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI：python -m FactorEvaluates.run_batch --start 2010-01-01"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from yg_quant_repo import relaunch_as_module

relaunch_as_module("FactorEvaluates.run_batch")

import argparse
import json

from Universes.catalog import names as universe_names

from .batch.batch_eval_runner import BatchEvalRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="全库因子评估")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--universe", default="all", choices=universe_names())
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32, help="一次装入内存的因子个数，越大读盘轮次越少、占用越高")
    parser.add_argument("--n-workers", type=int, default=0, help="日度进程数，0 表示自动（CPU 逻辑核数 - 1）")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    def progress(phase, detail, pct, message):
        print(f"[batch] {pct:5.0%} {phase} {message}", flush=True)

    runner = BatchEvalRunner(batch_size=args.batch_size, n_workers=args.n_workers)
    result = runner.run(
        {
            "start": args.start,
            "end": args.end,
            "universe": args.universe,
            "horizon": args.horizon,
            "run_id": args.run_id,
            "batch_size": args.batch_size,
            "n_workers": args.n_workers,
        },
        progress=progress,
    )
    print(json.dumps(result["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
