#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仓库符号（SH600000）与 QMT 代码（600000.SH）互转。"""

from __future__ import annotations


def to_qmt_code(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        left, right = text.split(".", 1)
        if left in {"SH", "SZ", "BJ"}:
            return f"{right}.{left}"
        return text
    if text.startswith(("SH", "SZ", "BJ")) and len(text) > 2:
        return f"{text[2:]}.{text[:2]}"
    if text.startswith(("6", "9")):
        return f"{text}.SH"
    if text.startswith(("4", "8")):
        return f"{text}.BJ"
    return f"{text}.SZ"


def from_qmt_code(code: str) -> str:
    text = str(code or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        number, market = text.split(".", 1)
        if market in {"SH", "SZ", "BJ"}:
            return f"{market}{number}"
        if number in {"SH", "SZ", "BJ"}:
            return f"{number}{market}"
    if text.startswith(("SH", "SZ", "BJ")):
        return text
    return to_qmt_code(text).replace(".", "")
