from .base import Universe
from .rules import BOARD_GEM


class GEM(Universe):
    name = "gem"
    description = "创业板全部（代码 300 / 301）"
    boards = (BOARD_GEM,)
