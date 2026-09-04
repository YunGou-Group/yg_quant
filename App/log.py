#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging

logger = logging.getLogger("App")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [app] %(message)s", "%H:%M:%S"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def say(message: str) -> None:
    print(f"[app] {message}", flush=True)
    logger.info(message)
