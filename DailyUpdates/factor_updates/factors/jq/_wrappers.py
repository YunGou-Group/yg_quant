#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给每个聚宽因子生成可发现的 BaseFactor 子类。"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import pandas as pd

from DailyUpdates.factor_updates.base_factor import BaseFactor

Spec = Tuple[str, str, str, int, Sequence[str]]


def install(
    namespace: dict,
    module_name: str,
    specs: Sequence[Spec],
    extract: Callable[..., pd.DataFrame],
) -> List[str]:
    names: List[str] = []
    for factor_name, key, desc, lookback, deps in specs:
        cls = _make(factor_name, key, desc, lookback, deps, extract)
        cls.__module__ = module_name
        namespace[cls.__name__] = cls
        names.append(cls.__name__)
    return names


def _make(
    factor_name: str,
    key: str,
    desc: str,
    lookback: int,
    deps: Sequence[str],
    extract: Callable[..., pd.DataFrame],
):
    class _Factor(BaseFactor):
        name = factor_name
        description = desc
        dependencies = list(deps)
        role = "alpha"
        stage = "production"
        lookback_days = int(lookback)
        _jq_key = key

        def calculate(
            self, data: pd.DataFrame, start_date: Optional[str] = None
        ) -> pd.DataFrame:
            return extract(data, self._jq_key, start_date=start_date)

    safe = "".join(part.title() for part in factor_name.split("_"))
    _Factor.__name__ = f"{safe}Factor"
    _Factor.__qualname__ = _Factor.__name__
    return _Factor
