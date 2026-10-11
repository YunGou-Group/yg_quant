#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中证1000指数增强：历史指数权重底仓 + LGBM 主动袖套。

组合权重：
    w = (1 - active_share) * w_benchmark + active_share * w_alpha

``w_benchmark`` 使用决策日可见的最近一期中证1000历史成分权重；
``w_alpha`` 在成分内按 LGBM 分数选 Top N 等权。LGBM 特征先对
Size+NLSIZE 残差化，标签仍是原始持有期收益。

策略只做多，不把指数风格中性化为零；它控制的是相对指数的主动偏离。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from DailyUpdates.storage import SQLiteStorage
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import cli_field, n_field, rebalance_field
from Strategies.lgbm import DEFAULT_HORIZON, DEFAULT_LOOKBACK, DEFAULT_REFIT
from Strategies.lgbm_sn import DEFAULT_BLEND, DEFAULT_SMOOTH, LgbmSizeNlTopK
from yg_quant_repo import default_db_path

logger = logging.getLogger("Strategies.lgbm_index_enhance")

DEFAULT_N = 100
DEFAULT_FACTOR = (
    "holding_ret_adj,-emotion_ARBR,growth_total_profit_growth_rate,alpha044"
)
DEFAULT_ACTIVE_SHARE = 0.20
BENCHMARK_INDEX_CODE = "000852.SH"
REQUIRED_UNIVERSE = "zz1000"


class LgbmIndexEnhanceTopK(LgbmSizeNlTopK):
    """中证1000基准权重为核心、LGBM TopN 为主动袖套。"""

    name = "lgbm_index_enhance"

    @classmethod
    def from_cli(cls, args, allocator):
        if not getattr(args, "factor", None):
            args.factor = DEFAULT_FACTOR
        if getattr(args, "n", None) is None:
            args.n = DEFAULT_N
        if getattr(args, "blend", None) is None:
            args.blend = DEFAULT_BLEND
        if getattr(args, "smooth", None) is None:
            args.smooth = DEFAULT_SMOOTH
        universe = str(getattr(args, "universe", "") or "").strip().lower()
        if universe != REQUIRED_UNIVERSE:
            raise SystemExit(
                "lgbm_index_enhance 必须使用 --universe zz1000，"
                "否则基准权重与股票池不一致"
            )
        inst = super().from_cli(args, allocator)
        raw = getattr(args, "active_share", None)
        inst.active_share = float(
            DEFAULT_ACTIVE_SHARE if raw is None else raw
        )
        if not 0.0 <= inst.active_share <= 1.0:
            raise SystemExit("lgbm_index_enhance --active-share 须在 0 与 1 之间")
        return inst

    @classmethod
    def cli_fields(cls):
        return [
            cli_field(
                "factor",
                "因子列表，默认指增核",
                "str",
                default=DEFAULT_FACTOR,
                placeholder=DEFAULT_FACTOR,
            ),
            n_field(DEFAULT_N, label="主动袖套持股数"),
            rebalance_field("20", placeholder="daily / weekly / 20"),
            cli_field("lookback", "LGBM回看天数", "int", default=DEFAULT_LOOKBACK),
            cli_field("horizon", "LGBM持有期", "int", default=DEFAULT_HORIZON),
            cli_field("refit", "同一模型沿用交易日数", "int", default=DEFAULT_REFIT),
            cli_field("blend", "树分权重", "float", default=DEFAULT_BLEND),
            cli_field("smooth", "上期分数平滑", "float", default=DEFAULT_SMOOTH),
            cli_field(
                "active_share",
                "LGBM主动袖套权重，余下为指数权重",
                "float",
                default=DEFAULT_ACTIVE_SHARE,
            ),
        ]

    @classmethod
    def run_tag(cls, args) -> str:
        if not getattr(args, "factor", None):
            args.factor = DEFAULT_FACTOR
        if getattr(args, "n", None) is None:
            args.n = DEFAULT_N
        if getattr(args, "blend", None) is None:
            args.blend = DEFAULT_BLEND
        if getattr(args, "smooth", None) is None:
            args.smooth = DEFAULT_SMOOTH
        tag = LgbmSizeNlTopK.run_tag(args)
        if tag.startswith("lgbmsn_"):
            tag = "lgbm_index_enhance_" + tag[len("lgbmsn_") :]
        else:
            tag = "lgbm_index_enhance_" + tag
        active = getattr(args, "active_share", None)
        active = DEFAULT_ACTIVE_SHARE if active is None else float(active)
        return tag + f"active{int(round(active * 100))}"

    def __init__(
        self,
        *args,
        active_share: float = DEFAULT_ACTIVE_SHARE,
        **kwargs,
    ):
        kwargs.setdefault("n", DEFAULT_N)
        super().__init__(*args, **kwargs)
        self.active_share = float(np.clip(active_share, 0.0, 1.0))
        self._benchmark_weights: Optional[np.ndarray] = None
        self._benchmark_dates: Optional[np.ndarray] = None

    def _load_benchmark_weights(self, ctx: DayContext) -> None:
        """把月度历史指数权重按 asof 向前填充到回测日；不会读取未来快照。"""
        storage = SQLiteStorage(str(default_db_path()))
        frame = storage.read_index_constituents(
            index_code=BENCHMARK_INDEX_CODE,
            start_date=str(ctx._store.dates[0]),
            end_date=str(ctx._store.dates[-1]),
        )
        if frame.empty:
            raise ValueError(
                f"缺少 {BENCHMARK_INDEX_CODE} 历史权重，请先更新 index_weight"
            )
        frame = frame.copy()
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
        frame = frame.dropna(subset=["trade_date", "symbol", "weight"])
        frame = frame[frame["weight"] > 0]
        if frame.empty:
            raise ValueError(f"{BENCHMARK_INDEX_CODE} 历史权重全为空")

        symbols = [str(s) for s in ctx.symbols]
        columns = {symbol: i for i, symbol in enumerate(symbols)}
        snapshot_days = np.asarray(
            sorted(frame["trade_date"].unique()), dtype="U10"
        )
        snapshots = np.zeros(
            (len(snapshot_days), len(symbols)), dtype=np.float64
        )
        day_pos = {str(day): i for i, day in enumerate(snapshot_days)}
        for row in frame.itertuples(index=False):
            col = columns.get(str(row.symbol))
            if col is not None:
                snapshots[day_pos[str(row.trade_date)], col] = float(row.weight)

        dates = np.asarray([str(d) for d in ctx._store.dates], dtype="U10")
        index = np.searchsorted(snapshot_days, dates, side="right") - 1
        weights = np.zeros((len(dates), len(symbols)), dtype=np.float64)
        valid = index >= 0
        weights[valid] = snapshots[index[valid]]
        self._benchmark_weights = weights
        self._benchmark_dates = dates
        logger.info(
            "中证1000历史权重已缓存：%s 个快照，%s~%s",
            len(snapshot_days),
            snapshot_days[0],
            snapshot_days[-1],
        )

    def _benchmark_weight(self, ctx: DayContext, tradable: np.ndarray) -> np.ndarray:
        if self._benchmark_weights is None:
            self._load_benchmark_weights(ctx)
        assert self._benchmark_weights is not None
        raw = np.asarray(
            self._benchmark_weights[int(ctx._t)], dtype=np.float64
        ).copy()
        raw = np.where(np.asarray(tradable, dtype=bool), raw, 0.0)
        total = float(raw.sum())
        if total <= 0:
            raise ValueError(
                f"{ctx.asof} 没有可用的 {BENCHMARK_INDEX_CODE} asof 权重"
            )
        return raw / total

    def score(self, ctx: DayContext) -> TargetHoldings:
        held = self._gate.skip(ctx)
        if held is not None:
            return held

        values = self._lgbm_score(ctx)
        universe = np.asarray(ctx.universe(), dtype=bool)
        benchmark = self._benchmark_weight(ctx, universe)

        eligible = universe & np.isfinite(values)
        k = int(min(self.n, int(eligible.sum())))
        alpha = np.zeros(len(ctx.symbols), dtype=np.float64)
        if k > 0 and self.active_share > 0:
            ranked = np.where(eligible, values, -np.inf)
            pick = np.argpartition(ranked, -k)[-k:]
            pick = pick[eligible[pick]]
            if pick.size:
                alpha[pick] = 1.0 / float(pick.size)

        active = self.active_share if alpha.sum() > 0 else 0.0
        weights = (1.0 - active) * benchmark + active * alpha
        weights = np.where(universe, weights, 0.0)
        total = float(weights.sum())
        if total > 0:
            weights /= total
        return self._gate.remember(
            TargetHoldings.from_array(
                ctx.asof, ctx.execute_on, ctx.symbols, weights
            )
        )
