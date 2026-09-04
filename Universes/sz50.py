from .base import Universe


class SZ50(Universe):
    name = "sz50"
    description = "上证50 历史成分"
    index_codes = ("000016.SH",)
