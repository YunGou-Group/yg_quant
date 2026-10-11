#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LGBM 变体：因子先对 Size+NLSIZE 截面中性化，再 LambdaRank 合成。

用法：--strategy lgbm_sn --factor a,b,-c
不改 lgbm.py。默认 --blend 1（纯树分）、--smooth 0.8。
标签仍是已实现持有期收益；树看到的是剥掉线性 Size/NLSIZE 后的因子残差。
"""

from __future__ import annotations

import numpy as np

from StrategyEngine.context import DayContext
from StrategyEngine.strategy import cli_field, n_field, rebalance_field
from Strategies._factor_combo import parse_factor_list
from Strategies._style_neutralize import (
    neutralize_panel_on_expo,
    size_nlsize_expo_on_store,
)
from Strategies.lgbm import (
    DEFAULT_HORIZON,
    DEFAULT_LOOKBACK,
    DEFAULT_N,
    DEFAULT_REFIT,
    LgbmWeightedTopK,
    combine_scores_lgbm,
)

DEFAULT_BLEND = 1.0
DEFAULT_SMOOTH = 0.8


class LgbmSizeNlTopK(LgbmWeightedTopK):
    name = "lgbm_sn"

    @classmethod
    def from_cli(cls, args, allocator):
        if getattr(args, "blend", None) is None:
            args.blend = DEFAULT_BLEND
        if getattr(args, "smooth", None) is None:
            args.smooth = DEFAULT_SMOOTH
        return super().from_cli(args, allocator)

    @classmethod
    def cli_fields(cls):
        return [
            cli_field("factor", "因子列表", "str", required=True, placeholder="a,b,-c"),
            n_field(DEFAULT_N),
            rebalance_field("20", placeholder="daily / weekly / 20"),
            cli_field("lookback", "LGBM回看天数", "int", default=DEFAULT_LOOKBACK),
            cli_field("horizon", "LGBM持有期", "int", default=DEFAULT_HORIZON),
            cli_field("refit", "同一模型沿用交易日数", "int", default=DEFAULT_REFIT),
            cli_field("blend", "树分权重", "float", default=DEFAULT_BLEND),
            cli_field("smooth", "上期分数平滑", "float", default=DEFAULT_SMOOTH),
        ]

    @classmethod
    def panel_kwargs(cls, args) -> dict:
        legs = parse_factor_list(args.factor)
        names = [name for name, _sign in legs]
        if "style_size" not in names:
            names = names + ["style_size"]
        return {"factors": names}

    @classmethod
    def run_tag(cls, args) -> str:
        blend = getattr(args, "blend", None)
        if blend is None:
            args.blend = DEFAULT_BLEND
        smooth = getattr(args, "smooth", None)
        if smooth is None:
            args.smooth = DEFAULT_SMOOTH
        tag = LgbmWeightedTopK.run_tag(args)
        if tag.startswith("lgbm_"):
            tag = "lgbmsn_" + tag[len("lgbm_") :]
        else:
            tag = "lgbmsn_" + tag
        s = DEFAULT_SMOOTH if getattr(args, "smooth", None) is None else float(args.smooth)
        return tag + f"sm{int(round(s * 100))}"

    def __init__(self, *args, blend: float = DEFAULT_BLEND, smooth: float = DEFAULT_SMOOTH, **kwargs):
        super().__init__(*args, blend=blend, smooth=smooth, **kwargs)

    def _lgbm_score(self, ctx: DayContext) -> np.ndarray:
        need = self.lookback + self.horizon
        close = ctx.history("close", need)
        t1 = int(ctx._t) + 1
        t0 = max(0, t1 - need)
        mask = np.asarray(ctx._store.mask[t0:t1], dtype=bool)
        if mask.shape[0] < need:
            pad = np.zeros((need - mask.shape[0], mask.shape[1]), dtype=bool)
            mask = np.vstack([pad, mask])
        signs = [sign for _, sign in self.legs]
        factor_hist = [ctx.history(name, need) for name, _ in self.legs]
        expo = size_nlsize_expo_on_store(ctx._store)[t0:t1]
        if expo.shape[0] < need:
            pad = np.full(
                (need - expo.shape[0], expo.shape[1], expo.shape[2]),
                np.nan,
                dtype=np.float64,
            )
            expo = np.concatenate([pad, expo], axis=0)
        factor_hist = [
            neutralize_panel_on_expo(panel, expo, mask) for panel in factor_hist
        ]
        reuse = None if self._need_refit(ctx._t) else self.last_model
        score, fitted = combine_scores_lgbm(
            factor_hist,
            signs,
            close,
            mask,
            lookback=self.lookback,
            horizon=self.horizon,
            min_rows=self.min_rows,
            blend=self.blend,
            model=reuse,
        )
        if self._need_refit(ctx._t):
            if fitted is not None:
                self.last_model = fitted
                self._fitted_t = int(ctx._t)
            elif self.last_model is not None:
                score, _ = combine_scores_lgbm(
                    factor_hist,
                    signs,
                    close,
                    mask,
                    lookback=self.lookback,
                    horizon=self.horizon,
                    min_rows=self.min_rows,
                    blend=self.blend,
                    model=self.last_model,
                )
        else:
            self.last_model = fitted if fitted is not None else self.last_model
        if self._prev_score is not None and self._prev_score.shape == score.shape and self.smooth > 0:
            both = np.isfinite(score) & np.isfinite(self._prev_score)
            mixed = score.copy()
            mixed[both] = (1.0 - self.smooth) * score[both] + self.smooth * self._prev_score[both]
            score = mixed
        self._prev_score = np.asarray(score, dtype=np.float64)
        return score
