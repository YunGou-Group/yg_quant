#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""股票池基类：底座规则 AND 可选指数成分 AND 可选板块。"""

from __future__ import annotations

from typing import ClassVar, Tuple

import pandas as pd

from .rules import apply_base_rules, board_member_mask, index_member_mask


class Universe:
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    min_list_days: ClassVar[int] = 60
    index_codes: ClassVar[Tuple[str, ...]] = ()
    boards: ClassVar[Tuple[str, ...]] = ()

    def build_mask(
        self,
        dates,
        symbols,
        open_panel: pd.DataFrame,
        storage,
        *,
        require_next_open: bool = False,
        min_list_days: int | None = None,
    ) -> pd.DataFrame:
        dates = pd.Index(pd.to_datetime(dates).strftime("%Y-%m-%d"))
        symbols = pd.Index([str(s) for s in symbols])
        mask = pd.DataFrame(True, index=dates, columns=symbols)
        days = self.min_list_days if min_list_days is None else int(min_list_days)
        mask = apply_base_rules(
            mask,
            open_panel,
            storage,
            min_list_days=days,
            require_next_open=require_next_open,
        )
        if self.index_codes:
            mask = mask & index_member_mask(
                dates, symbols, storage, self.index_codes
            )
        if self.boards:
            mask = mask & board_member_mask(dates, symbols, self.boards)
        return mask
