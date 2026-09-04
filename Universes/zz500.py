from .base import Universe


class ZZ500(Universe):
    name = "zz500"
    description = "中证500 历史成分"
    index_codes = ("000905.SH",)
