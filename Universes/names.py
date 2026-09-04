#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""证券简称历史：按 asof 取当时名称，禁止用今天的名字回溯。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .rules import iso_day

logger = logging.getLogger("Universes")


@dataclass(frozen=True)
class NameSpan:
    """一段有效期内的证券简称。end 为空串表示至今仍然有效。"""

    start: str
    end: str
    value: str

    def covers(self, asof: str) -> bool:
        if self.start and asof < self.start:
            return False
        if self.end and asof > self.end:
            return False
        return True


@dataclass
class NameHistory:
    spans: Dict[str, List[NameSpan]] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.spans

    def name_asof(self, symbol: str, asof: str) -> Optional[str]:
        """asof 当日的证券简称；没有历史记录时返回 None。"""
        for span in self.spans.get(str(symbol), ()):
            if span.covers(asof):
                return span.value
        return None

    @classmethod
    def from_storage(cls, storage) -> "NameHistory":
        try:
            frame = storage.read_stock_namechange()
        except Exception:
            logger.warning("读取 stock_namechange 失败，跳过 PIT 名称", exc_info=True)
            return cls()
        if frame is None or frame.empty:
            return cls()
        spans: Dict[str, List[NameSpan]] = {}
        for rec in frame.itertuples(index=False):
            symbol = str(getattr(rec, "symbol", "") or "")
            if not symbol:
                continue
            spans.setdefault(symbol, []).append(
                NameSpan(
                    iso_day(getattr(rec, "start_date", "")),
                    iso_day(getattr(rec, "end_date", "")),
                    str(getattr(rec, "name", "") or ""),
                )
            )
        for items in spans.values():
            items.sort(key=lambda s: s.start)
        return cls(spans)
