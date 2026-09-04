from .base import Universe


class AllAShare(Universe):
    name = "all"
    description = "A 股可交易（PIT 非 ST / 次新 / 未退市 / 当日有开盘）"
