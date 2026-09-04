"""网页进程：FastAPI + 静态前端。评估实现仍在 FactorEvaluates。"""

from .server import app, create_app, main

__all__ = ["app", "create_app", "main"]
