#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子家族键：alphaNNN → alpha101，style_* → barra，其余取首段前缀。"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Optional

_ALPHA = re.compile(r"^alpha\d+", re.I)


def family_of(name: str, overrides: Optional[Mapping[str, str]] = None) -> str:
    text = str(name or "").strip()
    if not text:
        return "other"
    if overrides and text in overrides:
        return str(overrides[text])
    if _ALPHA.match(text):
        return "alpha101"
    if text.startswith("style_"):
        return "barra"
    for sep in ("_", "-", "."):
        if sep in text:
            return text.split(sep, 1)[0] or "other"
    return text


def family_map(names: Iterable[str], overrides: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    return {name: family_of(name, overrides) for name in names}