#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网页控件 schema，以及请求参数校验。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str  # int | float | bool | enum
    label: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None
    scope: str = "metric"  # shared | metric

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "label": self.label,
            "default": self.default,
            "scope": self.scope,
        }
        if self.min is not None:
            payload["min"] = self.min
        if self.max is not None:
            payload["max"] = self.max
        if self.choices is not None:
            payload["choices"] = list(self.choices)
        return payload

    def coerce(self, value: Any) -> Any:
        if self.type == "int":
            try:
                coerced = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"参数 {self.name} 需要整数，收到 {value!r}") from exc
            self._check_range(coerced)
            return coerced
        if self.type == "float":
            try:
                coerced = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"参数 {self.name} 需要浮点数，收到 {value!r}") from exc
            self._check_range(coerced)
            return coerced
        if self.type == "bool":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "y"}:
                    return True
                if lowered in {"0", "false", "no", "n"}:
                    return False
                raise ValueError(f"参数 {self.name} 需要布尔值，收到 {value!r}")
            return bool(value)
        if self.type == "enum":
            choices = self.choices or ()
            if value not in choices:
                raise ValueError(f"参数 {self.name}={value!r} 不在 {list(choices)} 中")
            return value
        raise ValueError(f"不支持的参数类型: {self.type}")

    def _check_range(self, coerced: float) -> None:
        if self.min is not None and coerced < self.min:
            raise ValueError(f"参数 {self.name}={coerced} 小于最小值 {self.min}")
        if self.max is not None and coerced > self.max:
            raise ValueError(f"参数 {self.name}={coerced} 大于最大值 {self.max}")


HORIZON_PARAM = ParamSpec(
    name="horizon",
    type="int",
    label="持有交易日 N",
    default=5,
    min=1,
    max=60,
    scope="shared",
)

def universe_param() -> ParamSpec:
    """choices 来自 Universes/ 扫描结果，加新池不必改这里。"""
    from Universes.catalog import names

    return ParamSpec(
        name="universe",
        type="enum",
        label="股票池",
        default="all",
        choices=tuple(names()),
        scope="shared",
    )


UNIVERSE_PARAM = ParamSpec(
    name="universe",
    type="enum",
    label="股票池",
    default="all",
    choices=("all", "hs300", "zz500", "zz1000"),
    scope="shared",
)
