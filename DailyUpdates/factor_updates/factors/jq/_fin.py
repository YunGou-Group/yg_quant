#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚宽质量 / 基础 / 成长 / 每股：财务按 ann_date 前向填充到交易日。"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

from yg_quant_repo import default_db_path

from . import _panel as panel

_CACHE: Dict[int, "FinEngine"] = {}
_LOCK = threading.Lock()

PERCENT_FIELDS: Set[str] = {
    "roe",
    "roa",
    "roic",
    "npta",
    "grossprofit_margin",
    "netprofit_margin",
    "cogs_of_sales",
    "op_of_gr",
    "ebit_of_gr",
    "saleexp_to_gr",
    "adminexp_of_gr",
    "finaexp_of_gr",
    "ocf_to_or",
    "salescash_to_or",
    "ocf_to_opincome",
    "debt_to_assets",
    "debt_to_eqt",
    "assets_to_eqt",
    "longdeb_to_debt",
    "currentdebt_to_debt",
    "ca_to_assets",
    "nca_to_assets",
    "tbassets_to_totalassets",
    "ocf_to_debt",
    "ocf_to_netdebt",
    "ocf_to_shortdebt",
    "longdebt_to_workingcapital",
    "dtprofit_to_profit",
    "opincome_of_ebt",
    "n_op_profit_of_ebt",
    "investincome_of_ebt",
    "netprofit_yoy",
    "dt_netprofit_yoy",
    "or_yoy",
    "equity_yoy",
    "op_yoy",
    "ebt_yoy",
    "assets_yoy",
    "ocf_yoy",
    "basic_eps_yoy",
    "tr_yoy",
    "impai_ttm",
}


def scale_percent_ratio(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sample = values.to_numpy(dtype=np.float64, copy=False)
    med = np.nanmedian(np.abs(sample)) if sample.size else np.nan
    if np.isfinite(med) and med > 2.0:
        return values / 100.0
    return values


def _events_to_wide(
    events: pd.DataFrame, index: pd.Index, columns: pd.Index
) -> pd.DataFrame:
    empty = pd.DataFrame(np.nan, index=index, columns=columns, dtype=np.float64)
    if events is None or events.empty:
        return empty
    piece = events.dropna(subset=["value"]).copy()
    if piece.empty:
        return empty
    piece["ann_date"] = pd.to_datetime(piece["ann_date"]).dt.strftime("%Y-%m-%d")
    piece = piece.drop_duplicates(["ts_code", "ann_date"], keep="last")
    announced = piece.pivot(index="ann_date", columns="ts_code", values="value")
    calendar = pd.Index(index.astype(str))
    combined = calendar.union(announced.index).sort_values()
    wide = announced.reindex(combined).ffill().reindex(calendar)
    return wide.reindex(columns=columns).astype(np.float64)


class FinEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        financial: Optional[pd.DataFrame] = None,
    ):
        frame = panel.stock_frame(data)
        if "close" in frame.columns:
            self.template = panel.to_wide(frame, "close")
        elif "total_mv" in frame.columns:
            self.template = panel.to_wide(frame, "total_mv")
        else:
            raise ValueError("财务因子需要 close 或 total_mv 作为日历")
        self.total_mv = (
            panel.to_wide(frame, "total_mv") if "total_mv" in frame.columns else None
        )
        self.circ_mv = (
            panel.to_wide(frame, "circ_mv") if "circ_mv" in frame.columns else None
        )
        self.pe_ttm = (
            panel.to_wide(frame, "pe_ttm") if "pe_ttm" in frame.columns else None
        )
        self.ps_ttm = (
            panel.to_wide(frame, "ps_ttm") if "ps_ttm" in frame.columns else None
        )
        self.pb = panel.to_wide(frame, "pb") if "pb" in frame.columns else None
        self.total_share = (
            panel.to_wide(frame, "total_share")
            if "total_share" in frame.columns
            else None
        )
        self._fin = self._load_financial(financial)
        self._pit: Dict[str, pd.DataFrame] = {}
        self._derived: Dict[str, pd.DataFrame] = {}

    def _load_financial(self, override: Optional[pd.DataFrame]) -> pd.DataFrame:
        if override is not None:
            frame = override.copy()
        else:
            from DailyUpdates.storage.sqlite_storage import SQLiteStorage

            frame = SQLiteStorage(str(default_db_path())).read_financial_indicator()
        if frame is None or frame.empty:
            return pd.DataFrame()
        if "ts_code" not in frame.columns and "symbol" in frame.columns:
            frame = frame.rename(columns={"symbol": "ts_code"})
        frame["ts_code"] = frame["ts_code"].astype(str)
        if "update_flag" in frame.columns:
            frame = frame.sort_values(["ts_code", "ann_date", "update_flag"])
        else:
            frame = frame.sort_values(["ts_code", "ann_date"])
        for name in PERCENT_FIELDS:
            if name in frame.columns:
                frame[name] = scale_percent_ratio(frame[name])
        return frame

    def pit(self, field: str) -> pd.DataFrame:
        if field in self._pit:
            return self._pit[field]
        if self._fin.empty or field not in self._fin.columns:
            wide = pd.DataFrame(
                np.nan, index=self.template.index, columns=self.template.columns
            )
        else:
            events = self._fin[["ts_code", "ann_date", field]].rename(
                columns={field: "value"}
            )
            events["value"] = pd.to_numeric(events["value"], errors="coerce")
            if "update_flag" in self._fin.columns:
                events = events.assign(update_flag=self._fin["update_flag"])
                events = events.sort_values(["ts_code", "ann_date", "update_flag"])
            wide = _events_to_wide(events, self.template.index, self.template.columns)
        self._pit[field] = wide
        return wide

    def compute(self, key: str) -> pd.DataFrame:
        if key.startswith("pit:"):
            return self.pit(key.split(":", 1)[1])
        fn = getattr(self, f"_{key}", None)
        if fn is None:
            raise KeyError(f"未知财务因子: {key}")
        return fn()

    def _operating_cycle(self) -> pd.DataFrame:
        return self.pit("invturn_days") + self.pit("arturn_days")

    def _equity_to_asset(self) -> pd.DataFrame:
        debt = self.pit("debt_to_assets")
        return 1.0 - debt

    def _market_cap(self) -> pd.DataFrame:
        if self.total_mv is None:
            return self.template * np.nan
        return self.total_mv.reindex_like(self.template)

    def _circ_cap(self) -> pd.DataFrame:
        if self.circ_mv is None:
            return self.template * np.nan
        return self.circ_mv.reindex_like(self.template)

    def _sales_to_price(self) -> pd.DataFrame:
        if self.ps_ttm is None:
            return self.template * np.nan
        ps = self.ps_ttm.reindex_like(self.template)
        return (1.0 / ps).where(ps > 0)

    def _cash_flow_to_price(self) -> pd.DataFrame:
        ocfps = self.pit("ocfps")
        if self.total_mv is None or self.total_share is None:
            return self.template * np.nan
        mv = self.total_mv.reindex_like(self.template)
        shares = self.total_share.reindex_like(self.template)
        return (ocfps * shares / mv).where((mv > 0) & (shares > 0))

    def _peg(self) -> pd.DataFrame:
        if self.pe_ttm is None:
            return self.template * np.nan
        pe = self.pe_ttm.reindex_like(self.template)
        growth = self.pit("netprofit_yoy")
        # 聚宽：PE / (增长率 * 100)；增长率为小数。PE 或增长为负记缺失。
        return (pe / (growth * 100.0)).where((pe > 0) & (growth > 0))

    def _empty(self) -> pd.DataFrame:
        return self.template * np.nan

    def _memo(self, key: str, fn) -> pd.DataFrame:
        if key not in self._derived:
            self._derived[key] = fn()
        return self._derived[key]

    def _shares(self) -> Optional[pd.DataFrame]:
        if self.total_share is None:
            return None
        return self.total_share.reindex_like(self.template)

    def _mv(self) -> Optional[pd.DataFrame]:
        if self.total_mv is None:
            return None
        return self.total_mv.reindex_like(self.template)

    def _book_equity(self) -> pd.DataFrame:
        return self._memo(
            "book_equity",
            lambda: self._scale_ps("bps"),
        )

    def _total_assets(self) -> pd.DataFrame:
        def _calc() -> pd.DataFrame:
            eq = self._book_equity()
            ratio = 1.0 - self.pit("debt_to_assets")
            return (eq / ratio).where(ratio > 0)

        return self._memo("total_assets", _calc)

    def _total_liability(self) -> pd.DataFrame:
        return self._memo(
            "total_liability",
            lambda: self._total_assets() * self.pit("debt_to_assets"),
        )

    def _scale_ps(self, field: str) -> pd.DataFrame:
        shares = self._shares()
        if shares is None:
            return self._empty()
        return self.pit(field) * shares

    def _revenue(self) -> pd.DataFrame:
        return self._memo("revenue", lambda: self._scale_ps("revenue_ps"))

    def _total_revenue(self) -> pd.DataFrame:
        def _calc() -> pd.DataFrame:
            out = self._scale_ps("total_revenue_ps")
            return out.fillna(self._revenue())

        return self._memo("total_revenue", _calc)

    def _ocf(self) -> pd.DataFrame:
        return self._memo("ocf", lambda: self._scale_ps("ocfps"))

    def _times_rev(self, field: str, use_total: bool = True) -> pd.DataFrame:
        base = self._total_revenue() if use_total else self._revenue()
        return self.pit(field) * base

    def _prep_fin_field(self, field: str) -> pd.DataFrame:
        if self._fin.empty or field not in self._fin.columns:
            return pd.DataFrame()
        ev = self._fin[["ts_code", "ann_date", "end_date", field]].copy()
        ev["value"] = pd.to_numeric(ev[field], errors="coerce")
        ev["ann_date"] = ev["ann_date"].astype(str)
        ev["end_date"] = ev["end_date"].astype(str)
        if "update_flag" in self._fin.columns:
            ev = ev.assign(update_flag=self._fin["update_flag"])
            ev = ev.sort_values(["ts_code", "end_date", "ann_date", "update_flag"])
        else:
            ev = ev.sort_values(["ts_code", "end_date", "ann_date"])
        ev = ev.dropna(subset=["value"])
        ev = ev.drop_duplicates(["ts_code", "end_date"], keep="last")
        return ev[["ts_code", "ann_date", "end_date", "value"]]

    @staticmethod
    def _prior_end(end: str) -> str:
        raw = str(end).strip()
        stamp = raw.replace("-", "").replace("/", "")[:8]
        if len(stamp) < 8 or not stamp.isdigit():
            return ""
        try:
            year = int(stamp[:4]) - 1
        except ValueError:
            return ""
        ymd = f"{year:04d}{stamp[4:8]}"
        if "-" in raw:
            return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
        return ymd

    def _yoy_wide(
        self,
        field: str,
        how: str = "chg",
        annual_only: bool = False,
        events: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        ev = self._prep_fin_field(field) if events is None else events.copy()
        if ev.empty:
            return self._empty()
        if annual_only:
            ev = ev[ev["end_date"].str.endswith("1231")]
        if ev.empty:
            return self._empty()
        ev = ev.copy()
        ev["prior_end"] = ev["end_date"].map(self._prior_end)
        prior = ev.rename(columns={"value": "prior", "end_date": "pe"})
        merged = ev.merge(
            prior[["ts_code", "pe", "prior"]],
            left_on=["ts_code", "prior_end"],
            right_on=["ts_code", "pe"],
            how="left",
        )
        curr = merged["value"]
        prev = merged["prior"]
        if how == "chg":
            merged["value"] = curr / prev - 1.0
        elif how == "ratio":
            merged["value"] = curr / prev
        elif how == "inv_ratio":
            merged["value"] = prev / curr
        else:
            raise KeyError(f"未知同比方式: {how}")
        return _events_to_wide(
            merged[["ts_code", "ann_date", "value"]],
            self.template.index,
            self.template.columns,
        )

    def _annual_rows(self, field: str) -> pd.DataFrame:
        ev = self._prep_fin_field(field)
        if ev.empty:
            return ev
        return ev[ev["end_date"].str.endswith("1231")].copy()

    def _roll_annual(
        self, field: str, years: int, min_years: int, reducer
    ) -> pd.DataFrame:
        ev = self._annual_rows(field)
        if ev.empty:
            return self._empty()
        rows = []
        for code, group in ev.groupby("ts_code", sort=False):
            group = group.sort_values("end_date")
            values = group["value"].to_numpy(dtype=np.float64)
            anns = group["ann_date"].to_list()
            for i in range(len(group)):
                start = max(0, i - years + 1)
                stat = reducer(values[start : i + 1])
                if stat is None or not np.isfinite(stat):
                    continue
                rows.append((code, anns[i], float(stat)))
        if not rows:
            return self._empty()
        events = pd.DataFrame(rows, columns=["ts_code", "ann_date", "value"])
        return _events_to_wide(events, self.template.index, self.template.columns)

    @staticmethod
    def _geom_one_plus(window: np.ndarray, min_years: int = 4) -> float:
        sample = window[np.isfinite(window)]
        if sample.size < min_years:
            return np.nan
        plus = 1.0 + sample
        plus = plus[plus > 0]
        if plus.size < min_years:
            return np.nan
        return float(np.exp(np.mean(np.log(plus))) - 1.0)

    def _net_profit_to_total_operate_revenue_ttm(self) -> pd.DataFrame:
        den = self.pit("total_revenue_ps")
        return (self.pit("eps") / den).where(den != 0)

    def _cfo_to_ev(self) -> pd.DataFrame:
        mv = self._mv()
        if mv is None:
            return self._empty()
        ev = mv + self.pit("netdebt")
        return (self._ocf() / ev).where(ev != 0)

    def _fixed_asset_ratio(self) -> pd.DataFrame:
        assets = self._total_assets()
        return (self.pit("fixed_assets") / assets).where(assets > 0)

    def _DEGM(self) -> pd.DataFrame:
        return self._memo("DEGM", lambda: self._yoy_wide("grossprofit_margin", "chg"))

    def _net_operating_cash_flow_coverage(self) -> pd.DataFrame:
        eps = self.pit("eps")
        return (self.pit("ocfps") / eps).where(eps != 0)

    def _intangible_asset_ratio(self) -> pd.DataFrame:
        return 1.0 - self.pit("tbassets_to_totalassets")

    def _MLEV(self) -> pd.DataFrame:
        mv = self._mv()
        if mv is None:
            return self._empty()
        ltd = self._total_liability() * self.pit("longdeb_to_debt")
        den = ltd + mv
        return (ltd / den).where(den > 0)

    def _net_operate_cash_flow_to_asset(self) -> pd.DataFrame:
        assets = self._total_assets()
        return (self._ocf() / assets).where(assets > 0)

    def _long_debt_to_asset_ratio(self) -> pd.DataFrame:
        return self.pit("longdeb_to_debt") * self.pit("debt_to_assets")

    def _long_term_debt_to_asset_ratio(self) -> pd.DataFrame:
        return self._long_debt_to_asset_ratio()

    def _debt_to_tangible_equity_ratio(self) -> pd.DataFrame:
        da = self.pit("debt_to_assets")
        tba = self.pit("tbassets_to_totalassets")
        den = tba - da
        return (da / den).where(den > 0)

    def _ROAEBITTTM(self) -> pd.DataFrame:
        assets = self._total_assets()
        return (self.pit("ebit") / assets).where(assets > 0)

    def _equity_to_fixed_asset_ratio(self) -> pd.DataFrame:
        fa = self.pit("fixed_assets")
        return (self._book_equity() / fa).where(fa > 0)

    def _ACCA(self) -> pd.DataFrame:
        return self._net_operate_cash_flow_to_asset() - self.pit("roa")

    def _equity_turnover_rate(self) -> pd.DataFrame:
        bps = self.pit("bps")
        return (self.pit("revenue_ps") / bps).where(bps != 0)

    def _total_profit_to_cost_ratio(self) -> pd.DataFrame:
        cost = (
            self.pit("cogs_of_sales")
            + self.pit("saleexp_to_gr")
            + self.pit("adminexp_of_gr")
            + self.pit("finaexp_of_gr")
        )
        return (self.pit("ebit_of_gr") / cost).where(cost > 0)

    def _LVGI(self) -> pd.DataFrame:
        return self._yoy_wide("debt_to_assets", "ratio", annual_only=True)

    def _SGI(self) -> pd.DataFrame:
        return self._yoy_wide("revenue_ps", "ratio", annual_only=True)

    def _GMI(self) -> pd.DataFrame:
        return self._yoy_wide("grossprofit_margin", "inv_ratio", annual_only=True)

    def _DSRI(self) -> pd.DataFrame:
        return self._yoy_wide("arturn_days", "ratio", annual_only=True)

    def _SGAI(self) -> pd.DataFrame:
        sale = self._prep_fin_field("saleexp_to_gr")
        admin = self._prep_fin_field("adminexp_of_gr")
        if sale.empty or admin.empty:
            return self._empty()
        merged = sale.merge(
            admin,
            on=["ts_code", "end_date"],
            how="inner",
            suffixes=("_s", "_a"),
        )
        merged["value"] = merged["value_s"] + merged["value_a"]
        merged["ann_date"] = merged["ann_date_s"]
        events = merged[["ts_code", "ann_date", "end_date", "value"]]
        return self._yoy_wide(
            "saleexp_to_gr", "ratio", annual_only=True, events=events
        )

    def _rnoa_ttm(self) -> pd.DataFrame:
        return self.pit("op_of_gr") * self.pit("assets_turn")

    def _roe_ttm_8y(self) -> pd.DataFrame:
        return self._roll_annual(
            "roe", 8, 4, lambda w: self._geom_one_plus(w, 4)
        )

    def _roa_ttm_8y(self) -> pd.DataFrame:
        return self._roll_annual(
            "roa", 8, 4, lambda w: self._geom_one_plus(w, 4)
        )

    def _DEGM_8y(self) -> pd.DataFrame:
        def _calc() -> pd.DataFrame:
            ev = self._annual_rows("grossprofit_margin")
            if ev.empty:
                return self._empty()
            ev = ev.copy()
            ev["prior_end"] = ev["end_date"].map(self._prior_end)
            prior = ev.rename(columns={"value": "prior", "end_date": "pe"})
            merged = ev.merge(
                prior[["ts_code", "pe", "prior"]],
                left_on=["ts_code", "prior_end"],
                right_on=["ts_code", "pe"],
                how="left",
            )
            merged["value"] = merged["value"] / merged["prior"] - 1.0
            degm = merged.dropna(subset=["value"])[
                ["ts_code", "ann_date", "end_date", "value"]
            ]
            if degm.empty:
                return self._empty()
            rows = []
            for code, group in degm.groupby("ts_code", sort=False):
                group = group.sort_values("end_date")
                values = group["value"].to_numpy(dtype=np.float64)
                anns = group["ann_date"].to_list()
                for i in range(len(group)):
                    start = max(0, i - 7)
                    stat = self._geom_one_plus(values[start : i + 1], 4)
                    if not np.isfinite(stat):
                        continue
                    rows.append((code, anns[i], float(stat)))
            if not rows:
                return self._empty()
            events = pd.DataFrame(rows, columns=["ts_code", "ann_date", "value"])
            return _events_to_wide(
                events, self.template.index, self.template.columns
            )

        return self._memo("DEGM_8y", _calc)

    def _margin_stability(self) -> pd.DataFrame:
        def _reducer(window: np.ndarray) -> float:
            sample = window[np.isfinite(window)]
            if sample.size < 4:
                return np.nan
            std = float(np.std(sample, ddof=1))
            if std == 0:
                return np.nan
            return float(np.mean(sample) / std)

        return self._memo(
            "margin_stability",
            lambda: self._roll_annual("grossprofit_margin", 8, 4, _reducer),
        )

    def _maximum_margin(self) -> pd.DataFrame:
        left = self._margin_stability()
        right = self._DEGM_8y()
        return left.where(left >= right, right)

    def _total_operating_revenue_ttm(self) -> pd.DataFrame:
        return self._total_revenue()

    def _operating_revenue_ttm(self) -> pd.DataFrame:
        return self._revenue()

    def _operating_profit_ttm(self) -> pd.DataFrame:
        return self._times_rev("op_of_gr")

    def _net_operate_cash_flow_ttm(self) -> pd.DataFrame:
        return self._ocf()

    def _interest_carry_current_liability(self) -> pd.DataFrame:
        current = self._total_liability() * self.pit("currentdebt_to_debt")
        return current - self.pit("current_exint")

    def _sale_expense_ttm(self) -> pd.DataFrame:
        return self._times_rev("saleexp_to_gr")

    def _gross_profit_ttm(self) -> pd.DataFrame:
        return self._times_rev("grossprofit_margin", use_total=False)

    def _total_operating_cost_ttm(self) -> pd.DataFrame:
        return (
            self._times_rev("cogs_of_sales")
            + self._sale_expense_ttm()
            + self._administration_expense_ttm()
            + self._financial_expense_ttm()
        )

    def _administration_expense_ttm(self) -> pd.DataFrame:
        return self._times_rev("adminexp_of_gr")

    def _financial_expense_ttm(self) -> pd.DataFrame:
        return self._times_rev("finaexp_of_gr")

    def _total_profit_ttm(self) -> pd.DataFrame:
        return self.pit("ebit")

    def _net_profit_ttm(self) -> pd.DataFrame:
        return self._scale_ps("eps")

    def _OperateNetIncome(self) -> pd.DataFrame:
        return self.pit("opincome_of_ebt") * self.pit("ebit")

    def _asset_impairment_loss_ttm(self) -> pd.DataFrame:
        return self._times_rev("impai_ttm")

    def _np_parent_company_owners_ttm(self) -> pd.DataFrame:
        return self._scale_ps("eps")

    def _operating_cost_ttm(self) -> pd.DataFrame:
        return self._times_rev("cogs_of_sales", use_total=False)

    def _goods_sale_and_service_render_cash_ttm(self) -> pd.DataFrame:
        return self._times_rev("salescash_to_or", use_total=False)

    def _operating_liability(self) -> pd.DataFrame:
        return self._total_liability() - self.pit("interestdebt")

    def _operating_profit_per_share_ttm(self) -> pd.DataFrame:
        return self.pit("op_of_gr") * self.pit("total_revenue_ps").fillna(
            self.pit("revenue_ps")
        )

    def _operating_profit_per_share(self) -> pd.DataFrame:
        return self._operating_profit_per_share_ttm()


def get_fin_engine(
    data: pd.DataFrame, financial: Optional[pd.DataFrame] = None
) -> FinEngine:
    key = id(data) if financial is None else (id(data), id(financial))
    with _LOCK:
        engine = _CACHE.get(key)
        if engine is None:
            engine = FinEngine(data, financial=financial)
            if financial is None:
                _CACHE.clear()
                _CACHE[key] = engine
            else:
                _CACHE[key] = engine
        return engine


def extract_fin(
    data: pd.DataFrame,
    key: str,
    start_date: Optional[str] = None,
    financial: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    wide = get_fin_engine(data, financial=financial).compute(key)
    return panel.wide_to_long(wide, start_date=start_date)
