#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI：python -m StrategyEngine.attribution --run <回测快照>"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from yg_quant_repo import relaunch_as_module

relaunch_as_module("StrategyEngine.attribution")

from .evaluate import attribute_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="读取某次回测快照，输出 Brison 业绩归因")
    parser.add_argument("--run", required=True, help="回测快照 id 或 JSON 路径")
    parser.add_argument("--out", default=None, help="归因 JSON 路径，默认写在快照旁 *_attribution.json")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    report = attribute_snapshot(args.run, out=Path(args.out) if args.out else None)
    dest = (report.get("files") or {}).get("attribution")
    brief = report.get("summary") or {}
    print(
        f"归因 {dest}  status={brief.get('status')}  warnings={brief.get('warnings')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
