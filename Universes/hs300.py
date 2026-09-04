from .base import Universe


class HS300(Universe):
    name = "hs300"
    description = "沪深300 历史成分"
    index_codes = ("000300.SH",)
