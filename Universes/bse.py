from .base import Universe
from .rules import BOARD_BSE


class BSE(Universe):
    name = "bse"
    description = "北交所（代码 8 / 4）"
    boards = (BOARD_BSE,)
