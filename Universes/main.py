from .base import Universe
from .rules import BOARD_MAIN


class MainBoard(Universe):
    name = "main"
    description = "沪深主板（代码 60 / 00）"
    boards = (BOARD_MAIN,)
