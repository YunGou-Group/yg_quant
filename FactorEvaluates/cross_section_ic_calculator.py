#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""截面 RankIC 计算者：按日相关，不 shift 行情。"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CrossSectionICCalculator:
    def daily_rank_ic(
        self,
        factor: pd.DataFrame,
        fwd_ret: pd.DataFrame,
        min_obs: int = 20,
    ) -> pd.Series:
        x = factor.to_numpy(dtype=np.float64, copy=False)
        y = fwd_ret.to_numpy(dtype=np.float64, copy=False)
        # 先取交集再排名，与 matrix_utils.spearman_pairwise 保持同一口径：
        # 各自在全集上排名会让缺失样本挤占名次，系统性压低 |IC|。
        valid = np.isfinite(x) & np.isfinite(y)
        x = np.where(valid, x, np.nan)
        y = np.where(valid, y, np.nan)
        ic = self._pearson_rows(self._rank_rows(x), self._rank_rows(y), min_obs=min_obs)
        return pd.Series(ic, index=factor.index, name="daily_rank_ic")

    def daily_pearson_ic(
        self,
        factor: pd.DataFrame,
        fwd_ret: pd.DataFrame,
        min_obs: int = 20,
    ) -> pd.Series:
        x = factor.to_numpy(dtype=np.float64, copy=False)
        y = fwd_ret.to_numpy(dtype=np.float64, copy=False)
        ic = self._pearson_rows(x, y, min_obs=min_obs)
        return pd.Series(ic, index=factor.index, name="daily_ic")

    def summary(self, daily_ic: pd.Series) -> dict:
        clean = daily_ic.astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        n = int(len(clean))
        if n == 0:
            return {
                "mean": np.nan,
                "std": np.nan,
                "icir": np.nan,
                "positive_ratio": np.nan,
                "n_days": 0,
            }
        values = clean.to_numpy(dtype=np.float64, copy=False)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else float("nan")
        icir = (
            float(mean / std) if n > 1 and std > 0 and np.isfinite(std) else float("nan")
        )
        return {
            "mean": mean,
            "std": std,
            "icir": icir,
            "positive_ratio": float(np.mean(values > 0)),
            "n_days": n,
        }

    @staticmethod
    def _rank_rows(values: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, np.nan, dtype=np.float64)
        for i in range(values.shape[0]):
            row = values[i]
            valid = np.isfinite(row)
            n = int(valid.sum())
            if n < 3:
                continue
            vals = row[valid]
            order = np.argsort(vals, kind="mergesort")
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(1, n + 1, dtype=np.float64)
            sorted_vals = vals[order]
            j = 0
            while j < n:
                k = j + 1
                while k < n and sorted_vals[k] == sorted_vals[j]:
                    k += 1
                if k - j > 1:
                    ranks[order[j:k]] = 0.5 * ((j + 1) + k)
                j = k
            out[i, valid] = ranks
        return out

    @staticmethod
    def _pearson_rows(
        left: np.ndarray, right: np.ndarray, min_obs: int = 20
    ) -> np.ndarray:
        valid = np.isfinite(left) & np.isfinite(right)
        count = valid.sum(axis=1)
        out = np.full(left.shape[0], np.nan, dtype=np.float64)
        ok_rows = count >= min_obs
        if not ok_rows.any():
            return out
        left_c = np.where(valid, left, np.nan)
        right_c = np.where(valid, right, np.nan)
        left_mean = np.zeros((left.shape[0], 1), dtype=np.float64)
        right_mean = np.zeros((left.shape[0], 1), dtype=np.float64)
        left_mean[ok_rows] = np.nanmean(left_c[ok_rows], axis=1, keepdims=True)
        right_mean[ok_rows] = np.nanmean(right_c[ok_rows], axis=1, keepdims=True)
        ld = np.where(valid, left_c - left_mean, 0.0)
        rd = np.where(valid, right_c - right_mean, 0.0)
        num = (ld * rd).sum(axis=1)
        den = np.sqrt((ld * ld).sum(axis=1) * (rd * rd).sum(axis=1))
        ok = ok_rows & (den > 0)
        out[ok] = num[ok] / den[ok]
        return out
