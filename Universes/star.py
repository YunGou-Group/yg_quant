from .base import Universe
from .rules import BOARD_STAR


class STAR(Universe):
    name = "star"
    description = "科创板全部（代码 688）"
    boards = (BOARD_STAR,)
