#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 yg_quant 仓库根加入 sys.path，供脚本、spawn 子进程和测试使用。

库内代码应写 ``from DailyUpdates...`` / ``from FactorEvaluates...`` / ``from StrategyEngine...`` / ``from Strategies...`` / ``from Universes...`` / ``from App...``，
不要把 DailyUpdates 目录本身塞进 sys.path（那会变成第二套顶层包名）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def repo_root(start: Optional[PathLike] = None) -> Path:
    here = Path(start or __file__).resolve()
    if here.is_file():
        here = here.parent
    for cand in [here, *here.parents]:
        if (cand / "DailyUpdates").is_dir() and (cand / "FactorEvaluates").is_dir():
            return cand
    raise RuntimeError(
        "找不到 yg_quant 仓库根目录（需要同时包含 DailyUpdates 与 FactorEvaluates）。"
        "请在仓库内运行，或把仓库根加入 PYTHONPATH。"
    )


def ensure_repo_root(start: Optional[PathLike] = None) -> Path:
    root = repo_root(start)
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


def _env_path(name: str) -> Optional[Path]:
    text = os.getenv(name, "").strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def default_data_dir() -> Path:
    override = _env_path("YG_QUANT_DATA_DIR")
    if override is not None:
        return override
    return repo_root() / "data"


def default_db_path() -> Path:
    override = _env_path("YG_QUANT_DB_PATH")
    if override is not None:
        return override
    preferred = default_data_dir() / "yg_quant.db"
    legacy = default_data_dir() / "qmt.db"
    if not preferred.is_file() and legacy.is_file():
        return legacy
    return preferred


def default_factor_dir() -> Path:
    override = _env_path("YG_QUANT_FACTOR_DIR")
    if override is not None:
        return override
    return default_db_path().parent / "factors"


def default_eval_dir() -> Path:
    override = _env_path("YG_QUANT_EVAL_DIR")
    if override is not None:
        return override
    factor = _env_path("YG_QUANT_FACTOR_DIR")
    if factor is not None:
        return factor.parent / "factor_eval"
    return default_db_path().parent / "factor_eval"


def default_style_crowding_dir() -> Path:
    override = _env_path("YG_QUANT_STYLE_CROWDING_DIR")
    if override is not None:
        return override
    return default_data_dir() / "style_crowding"


def relaunch_as_module(module_name: str) -> None:
    """若当前文件被当成松散脚本启动，则改以 ``python -m module_name`` 再跑一遍。

    相对导入（``from .Foo import Bar``）在 F5 / ``python path/to/file.py`` 时会失败；
    入口文件在插入仓库根后立刻调用本函数即可。
    """
    import inspect
    import runpy

    caller = inspect.currentframe()
    if caller is None or caller.f_back is None:
        return
    globals_ = caller.f_back.f_globals
    if globals_.get("__name__") == "__main__" and not globals_.get("__package__"):
        runpy.run_module(module_name, run_name="__main__")
        raise SystemExit(0)


# 被直接 import 时也保证可用（python -m、IDE 从仓库根启动）
ensure_repo_root()
