#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI：python -m StyleCrowding run | incremental"""

from __future__ import annotations

import argparse
import logging
import sys

from yg_quant_repo import ensure_repo_root

ensure_repo_root()

from Universes.catalog import names as universe_names
from StyleCrowding.config import default_settings
from StyleCrowding.runner import run_style_crowding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Barra Style 拥挤监测（独立风控模块）")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="全量或指定区间发布")
    run_p.add_argument("--start", default="2018-01-01")
    run_p.add_argument("--end", default="")
    run_p.add_argument("--universe", default="all", choices=universe_names())
    run_p.add_argument("--skip", nargs="*", default=[], help="跳过指标，如 pairwise")
    run_p.add_argument("--group", type=int, nargs="*", default=None, help="仅跑指定分组，如 10 5")
    run_p.add_argument("--orientation", nargs="*", default=None, help="positive reverse")

    inc_p = sub.add_parser("incremental", help="从 manifest 最后日期增量")
    inc_p.add_argument("--end", default="")
    inc_p.add_argument("--skip", nargs="*", default=[])

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    incremental = args.command == "incremental"
    settings = default_settings(
        start_date="" if incremental else getattr(args, "start", "2018-01-01"),
        end_date=args.end if hasattr(args, "end") else "",
        universe=getattr(args, "universe", "all"),
        skip_indicators=tuple(getattr(args, "skip", []) or ()),
    )
    if getattr(args, "group", None):
        settings.group_counts = tuple(int(g) for g in args.group)
    if getattr(args, "orientation", None):
        settings.orientations = tuple(args.orientation)

    result = run_style_crowding(settings, incremental=incremental)
    print(f"发布完成 → {result['root']} ({result.get('elapsed_sec')}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
