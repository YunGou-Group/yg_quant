from .base import Universe


class CYB(Universe):
    name = "cyb"
    description = "创业板指 历史成分"
    index_codes = ("399006.SZ",)
