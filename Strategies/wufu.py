#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""五福 ETF 日频：25 日加权对数动量 × R²，走弱期切全球池，无标的时买银华日利。

相对聚宽原版的日频替代：
- 全日成交额 / 成交量，不再用分钟量外推
- 去掉盘中趋势确认、分钟止损、13:10 定时
- asof 收盘给目标，引擎 T+1 开盘成交
- Tushare amount 为千元，流动性门槛 ×1000 换成元
- ETF 份额折算按次日 pre_close 做前复权，避免折算日净值腰斩

用法：
  python -m StrategyEngine --mode backtest --strategy wufu --start 2023-01-01
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from FactorEvaluates.market_panel_loader import MarketPanelLoader
from StrategyEngine.allocators import Allocator, EqualWeight
from StrategyEngine.context import DayContext
from StrategyEngine.holdings import TargetHoldings
from StrategyEngine.strategy import Strategy, n_field, rebalance_field
from Strategies.rebalance_schedule import RebalanceGate, RebalanceSpec, parse_rebalance, spec_from_cli

logger = logging.getLogger("Strategies")

DEFAULT_N = 1
DEFAULT_REBALANCE = "daily"
LOOKBACK_DAYS = 25
MIN_SCORE = 0.0
MAX_SCORE = 5.0
SCORE_RATIO = 0.9
R2_THRESHOLD = 0.4
MA_LOOKBACK = 10
MA_THRESHOLD = 1.0
VOLUME_LOOKBACK = 5
VOLUME_THRESHOLD = 1.8
LOSS_FLOOR = 0.97
WEAK_MA = 10
MAX_WEAK_DAYS = 20
LIQ_DAYS = 3
LIQ_DIVISOR = 20000.0
LIQ_FALLBACK = 10_000_000.0
AMOUNT_YUAN_SCALE = 1000.0  # Tushare fund_daily.amount = 千元
DYNAMIC_TOP = 300
DEFENSIVE = "SH511880"
WARMUP_DAYS = 45

WEAK_INDEXES: Tuple[Tuple[str, str], ...] = (
    ("大盘", "index_SH000300"),
    ("小盘", "index_SZ399101"),
    ("创业板", "index_SZ399006"),
    ("中证A500", "index_SH000510"),
)


def jq_to_repo(code: str) -> str:
    """聚宽 518880.XSHG / Tushare 518880.SH → SH518880。"""
    text = str(code).strip().upper()
    if not text:
        return text
    text = text.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")
    if text.startswith("INDEX_"):
        return text
    if "." in text:
        num, market = text.split(".", 1)
        if market in {"SH", "SZ", "BJ"}:
            return f"{market}{num}"
    if text.startswith(("SH", "SZ", "BJ")) and len(text) >= 8 and "." not in text:
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    code6 = digits[-6:].zfill(6) if digits else ""
    if not code6:
        return text
    if code6.startswith(("5", "6", "9")):
        return f"SH{code6}"
    return f"SZ{code6}"


_GLOBAL_JQ = (
    "518880.XSHG", "501018.XSHG", "161226.XSHE", "159985.XSHE", "159980.XSHE",
    "513310.XSHG", "159518.XSHE", "159509.XSHE", "513100.XSHG", "513520.XSHG",
    "513500.XSHG", "159502.XSHE", "513400.XSHG", "513030.XSHG", "513290.XSHG",
    "520830.XSHG", "159529.XSHE",
)
_CHINA_JQ = (
    "513090.XSHG", "513120.XSHG", "513180.XSHG", "513330.XSHG", "513750.XSHG",
    "159892.XSHE", "513190.XSHG", "159605.XSHE", "513630.XSHG", "159323.XSHE",
    "510900.XSHG", "513920.XSHG", "513970.XSHG",
    "511380.XSHG", "512050.XSHG", "510500.XSHG", "159915.XSHE", "510300.XSHG",
    "512100.XSHG", "159949.XSHE", "588080.XSHG", "159967.XSHE", "588220.XSHG",
    "563300.XSHG", "510760.XSHG",
    "588200.XSHG", "515880.XSHG", "159981.XSHE", "512880.XSHG", "513350.XSHG",
    "159326.XSHE", "159516.XSHE", "159206.XSHE", "512480.XSHG", "159363.XSHE",
    "159870.XSHE", "512400.XSHG", "159755.XSHE", "588170.XSHG", "159992.XSHE",
    "159995.XSHE", "512890.XSHG", "515220.XSHG", "159566.XSHE", "159819.XSHE",
    "512800.XSHG", "512690.XSHG", "515050.XSHG", "562500.XSHG", "512170.XSHG",
    "517520.XSHG", "159869.XSHE", "512070.XSHG", "159611.XSHE", "562800.XSHG",
    "515120.XSHG", "512010.XSHG", "510880.XSHG", "515790.XSHG", "515980.XSHG",
    "512660.XSHG", "159928.XSHE", "512710.XSHG", "560860.XSHG", "515030.XSHG",
    "159766.XSHE", "159218.XSHE", "159852.XSHE", "516160.XSHG", "516150.XSHG",
    "159227.XSHE", "159583.XSHE", "588790.XSHG", "159865.XSHE", "512980.XSHG",
    "159851.XSHE", "561360.XSHG", "561980.XSHG", "562590.XSHG", "512200.XSHG",
    "159732.XSHE", "159667.XSHE", "516510.XSHG", "159840.XSHE", "159998.XSHE",
    "159825.XSHE", "512670.XSHG", "159883.XSHE", "515210.XSHG", "515400.XSHG",
    "159256.XSHE", "561330.XSHG", "515170.XSHG", "159638.XSHE", "516520.XSHG",
    "513360.XSHG", "516190.XSHG",
)
GLOBAL_POOL: Tuple[str, ...] = tuple(jq_to_repo(c) for c in _GLOBAL_JQ)
CHINA_POOL: Tuple[str, ...] = tuple(jq_to_repo(c) for c in _CHINA_JQ)
FIXED_POOL: Tuple[str, ...] = GLOBAL_POOL + CHINA_POOL

FUND_COMPANIES: Tuple[str, ...] = tuple(
    sorted(
        {
            "易方达", "广发", "华夏", "华安", "嘉实", "富国", "招商", "鹏华", "南方", "汇添富",
            "国泰", "平安", "银华", "天弘", "建信", "工银", "华泰柏瑞", "博时", "景顺长城", "景顺",
            "华宝", "申万菱信", "万家", "中欧", "兴证全球", "浙商", "诺安", "前海开源", "泰康",
            "泰达宏利", "农银汇理", "交银", "东方红", "财通", "华商", "国联", "永赢", "金鹰",
            "德邦", "创金合信", "西部利得", "圆信永丰", "泓德", "汇安", "诺德", "恒生前海",
            "华润元大", "大成", "海富通", "摩根", "华泰", "中信", "中银", "兴全", "国信", "长城",
            "中金", "浙商证券", "东海", "东吴", "浦银安盛", "信达澳亚", "中加", "中航", "中融",
            "中邮", "中庚", "中信保诚", "中信建投", "中银国际", "中银证券", "九泰", "交银施罗德",
            "光大保德信", "兴银", "农银", "国投瑞银", "国海富兰克林", "国联安", "国金", "太平",
            "方正富邦", "民生加银", "汇丰晋信", "银河", "长信", "长安", "长盛", "长江证券", "鹏扬",
        },
        key=len,
        reverse=True,
    )
)
NOISE_WORDS: Tuple[str, ...] = tuple(
    sorted(
        {
            "6666", "8888", "9999", "A类", "AH", "B", "BS", "C", "C类", "CS", "DB", "E", "E类",
            "ETF", "ETF基金", "ETF联接", "FG", "G60", "GF", "GT", "HGS", "LOF", "LOF基金", "LOF联接",
            "SG", "SZ", "TF", "TK", "WJ", "YH", "ZS", "ZZ", "板块", "策略", "产业", "场内", "场外",
            "低波", "基本面", "基金", "精选", "联接", "联接基金", "量化", "龙头", "民企", "民营",
            "国企", "央企", "智能", "全指", "上市开放式", "指基", "指增", "指数", "指数A", "指数C",
            "指数ETF", "指数基金", "主题", "增强", "上海", "黄", "30", "50", "100", "300", "500",
            "1000", "2000", "大", "新", "四川", "浙江", "湖北",
        },
        key=len,
        reverse=True,
    )
)
SPECIAL_GROUPS: Tuple[Dict[str, object], ...] = (
    {
        "name": "香港组",
        "keywords": ("恒生", "恒指", "港股", "港股通", "H股", "香港", "港", "HKC", "HK", "HGS", "H", "中概", "HS科技"),
        "remove_words": ("恒生", "恒指", "港股", "港股通", "H股", "香港", "港", "HKC", "HK", "HGS", "H", "中概", "HS"),
    },
    {
        "name": "科创组",
        "keywords": ("科创", "科创板", "科综", "KC", "K C", "双创", "科创创业", "创创"),
        "remove_words": (
            "科创", "科创板", "科综", "KC", "K C", "双创", "科创创业", "创创",
            "债券", "债汇", "债指", "债沪", "债易", "债基", "债兴", "债摩", "债", "AAA",
        ),
    },
    {
        "name": "创业组",
        "keywords": ("创业板", "创业", "创板", "创成长"),
        "remove_words": ("创业板", "创业", "创板", "创成长"),
    },
    {
        "name": "美指组",
        "keywords": ("标普", "纳指", "纳斯达克"),
        "remove_words": ("标普", "纳指", "纳斯达克"),
    },
)
EXCLUDE_KEYWORDS: Tuple[str, ...] = tuple(
    sorted(
        {
            "300", "500", "1000", "2000", "800", "30", "50", "100", "180", "200",
            "沪深", "中证", "上证", "深证", "深成", "A50", "A100", "A500", "深100",
            "短融", "可转债", "转债", "双债", "利率债", "国债", "地债", "政金债", "国开债",
            "基准国债", "新综债", "信用债", "企业债", "公司债", "城投债", "城投", "美元债",
            "沪公司债", "科创债", "科债", "科创AAA", "自由现金流", "现金流", "现金流E",
            "现金流基", "现金流TF", "现金流全", "300现金流", "800现金流", "货币", "现金",
            "快线", "快钱", "中银现金", "500现金", "800现金", "现金800", "现金自由",
            "现金指数", "全指现金", "现金全指", "ESG", "MSCI", "MS", "债",
        },
        key=len,
        reverse=True,
    )
)


@dataclass
class WeakState:
    is_weak: bool = False
    start: Optional[str] = None


@dataclass
class EtfMetrics:
    symbol: str
    name: str
    score: float
    r2: float
    volume_ratio: Optional[float]
    passed_momentum: bool
    passed_r2: bool
    passed_ma: bool
    passed_volume: bool
    passed_loss: bool


def momentum_score(price_series: np.ndarray, lookback_days: int = LOOKBACK_DAYS):
    """加权对数价格回归：年化收益 × R²。样本不足返回 (None, None, None)。"""
    series = np.asarray(price_series, dtype=np.float64)
    series = series[np.isfinite(series) & (series > 0)]
    if series.size < lookback_days + 1:
        return None, None, None
    y = np.log(series[-(lookback_days + 1) :])
    x = np.arange(len(y), dtype=np.float64)
    weights = np.linspace(1.0, 2.0, len(y))
    w2 = weights ** 2
    w_sum = float(np.sum(w2))
    x_bar = float(np.sum(w2 * x) / w_sum)
    y_bar = float(np.sum(w2 * y) / w_sum)
    dx = x - x_bar
    dy = y - y_bar
    variance_x = float(np.sum(w2 * dx ** 2))
    if variance_x == 0:
        return 0.0, 0.0, 0.0
    slope = float(np.sum(w2 * dx * dy) / variance_x)
    intercept = y_bar - slope * x_bar
    annualized = math.exp(slope * 250.0) - 1.0
    y_pred = slope * x + intercept
    ss_res = float(np.sum(weights * (y - y_pred) ** 2))
    ss_tot = float(np.sum(weights * (y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return annualized * r2, annualized, r2


def volume_ratio_daily(
    hist_volumes: np.ndarray,
    today_vol: float,
    lookback: int = VOLUME_LOOKBACK,
) -> Optional[float]:
    """全日量 / 近 lookback 日均量。不再按盘中分钟外推。"""
    hist = np.asarray(hist_volumes, dtype=np.float64)
    if hist.size < lookback:
        return None
    past = hist[-lookback:]
    if np.any(~np.isfinite(past)) or np.any(past <= 0):
        return None
    avg = float(np.mean(past))
    if avg <= 0 or not np.isfinite(today_vol) or today_vol < 0:
        return None
    return float(today_vol / avg)


def passed_loss_filter(price_series: np.ndarray, floor: float = LOSS_FLOOR) -> bool:
    px = np.asarray(price_series, dtype=np.float64)
    px = px[np.isfinite(px) & (px > 0)]
    if px.size < 4:
        return True
    ratios = px[-3:] / px[-4:-1]
    return bool(np.min(ratios) >= floor)


def passed_ma_filter(price_series: np.ndarray, ma_n: int = MA_LOOKBACK, thresh: float = MA_THRESHOLD) -> bool:
    px = np.asarray(price_series, dtype=np.float64)
    px = px[np.isfinite(px) & (px > 0)]
    if px.size < ma_n:
        return False
    current = float(px[-1])
    ma = float(np.mean(px[-ma_n:]))
    return current > ma * thresh


def apply_filters(rows: Sequence[EtfMetrics], *, is_weak: bool) -> List[EtfMetrics]:
    out = list(rows)
    steps = [
        (lambda m: m.passed_momentum, True),
        (lambda m: m.passed_r2, not is_weak),
        (lambda m: m.passed_ma, is_weak),
        (lambda m: m.passed_volume, True),
        (lambda m: m.passed_loss, True),
    ]
    for cond, enabled in steps:
        if enabled:
            out = [m for m in out if cond(m)]
    return out


def liquidity_threshold(amount_hist: np.ndarray, divisor: float = LIQ_DIVISOR) -> float:
    """近 3 日全市场 ETF 成交额均值 / divisor。amount 为千元。"""
    yuan = np.asarray(amount_hist, dtype=np.float64) * AMOUNT_YUAN_SCALE
    yuan = np.where(np.isfinite(yuan) & (yuan > 0.0), yuan, 0.0)
    daily = yuan.sum(axis=1)
    had = np.isfinite(np.asarray(amount_hist, dtype=np.float64)).any(axis=1) & (daily > 0)
    if int(had.sum()) < LIQ_DAYS:
        return LIQ_FALLBACK
    return float(daily[had][-LIQ_DAYS:].mean()) / float(divisor)


def avg_daily_yuan(amount_hist: np.ndarray) -> np.ndarray:
    yuan = np.asarray(amount_hist, dtype=np.float64) * AMOUNT_YUAN_SCALE
    yuan = np.where(np.isfinite(yuan), yuan, 0.0)
    n = max(1, int(yuan.shape[0]))
    return yuan.sum(axis=0) / float(n)


def weak_vote_need(n_valid: int, required: int = 3) -> int:
    """原文 4 只指数里至少 3 只表决。缺历史时按有效只数，至少 2 只才开票。"""
    valid = int(n_valid)
    need = int(required)
    if valid < 2:
        return need
    return min(need, valid)


def step_weak_period(
    state: WeakState,
    asof: str,
    dates: Sequence[str],
    below_count: int,
    above_count: int,
    max_days: int = MAX_WEAK_DAYS,
    need: int = 3,
) -> WeakState:
    days_count = 0
    if state.is_weak and state.start:
        days_count = sum(1 for day in dates if state.start <= day <= asof)
    exceeded = days_count >= int(max_days)
    threshold = int(need)
    if state.is_weak:
        if exceeded or above_count >= threshold:
            return WeakState(False, None)
        if below_count >= threshold:
            return WeakState(True, asof)
        return WeakState(True, state.start)
    if below_count >= threshold:
        return WeakState(True, asof)
    return WeakState(False, None)


def index_vs_ma(series: Optional[pd.Series], asof: str, ma_n: int = WEAK_MA) -> Optional[bool]:
    """True 站上均线，False 低于，None 不足或持平。"""
    if series is None or series.empty:
        return None
    work = pd.to_numeric(series, errors="coerce").copy()
    work.index = pd.to_datetime(work.index).strftime("%Y-%m-%d")
    work = work[~work.index.duplicated(keep="last")].sort_index()
    work = work[work.index <= asof].dropna()
    if len(work) < int(ma_n):
        return None
    last = float(work.iloc[-1])
    ma = float(work.iloc[-int(ma_n) :].mean())
    if last > ma:
        return True
    if last < ma:
        return False
    return None


def match_special_group(name: str) -> Optional[str]:
    text = str(name or "")
    for group in SPECIAL_GROUPS:
        for kw in group["keywords"]:
            if str(kw) in text:
                return str(group["name"])
    return None


def is_excluded_name(name: str) -> bool:
    text = str(name or "")
    return any(k in text for k in EXCLUDE_KEYWORDS)


def clean_etf_name(name: str, *, group: Optional[str] = None) -> str:
    cleaned = str(name or "")
    for company in FUND_COMPANIES:
        cleaned = cleaned.replace(company, "")
    if group:
        for item in SPECIAL_GROUPS:
            if item["name"] == group:
                for word in item["remove_words"]:
                    cleaned = cleaned.replace(str(word), "")
                break
    for noise in NOISE_WORDS:
        cleaned = cleaned.replace(noise, "")
    return cleaned.strip()


def dynamic_sector_pool(
    names: Mapping[str, str],
    avg_money: Mapping[str, float],
    threshold: float,
    top_n: int = DYNAMIC_TOP,
) -> List[str]:
    """名称聚类后每组取成交额最大的一只，再截 top_n。"""
    normal: Dict[str, List[Tuple[str, float]]] = {}
    special: Dict[str, List[Tuple[str, float]]] = {}
    for code, raw_name in names.items():
        money = float(avg_money.get(code, 0.0) or 0.0)
        if money <= threshold:
            continue
        if is_excluded_name(raw_name):
            continue
        group = match_special_group(raw_name)
        cleaned = clean_etf_name(raw_name, group=group)
        if not cleaned:
            continue
        key = cleaned[:2] if len(cleaned) >= 2 else cleaned
        if group:
            bucket = special.setdefault(f"{group}_{key}", [])
        else:
            bucket = normal.setdefault(key, [])
        bucket.append((code, money))
    picked: List[Tuple[str, float]] = []
    for bucket in list(normal.values()) + list(special.values()):
        bucket.sort(key=lambda item: item[1], reverse=True)
        picked.append(bucket[0])
    picked.sort(key=lambda item: item[1], reverse=True)
    return [code for code, _ in picked[: int(top_n)]]


def select_targets(
    filtered: Sequence[EtfMetrics],
    holdings: Sequence[str],
    hold_n: int,
    score_ratio: float,
) -> List[str]:
    """过滤后按动量排序，保留仍在候选池里的旧仓，再补到 hold_n。"""
    ranked = sorted(filtered, key=lambda m: m.score, reverse=True)
    top10 = ranked[:10]
    if not top10:
        return []
    n = max(1, int(hold_n))
    if len(top10) >= n:
        reference = top10[n - 1].score
        floor = reference * float(score_ratio)
        candidates = [m for m in top10 if m.score >= floor]
    else:
        candidates = list(top10)
    by_code = {m.symbol: m for m in candidates}
    retained = [by_code[code] for code in holdings if code in by_code]
    if len(retained) >= n:
        retained.sort(key=lambda m: m.score, reverse=True)
        return [m.symbol for m in retained[:n]]
    kept = {m.symbol for m in retained}
    extra = [m for m in candidates if m.symbol not in kept][: n - len(retained)]
    return [m.symbol for m in retained + extra]


def _load_etf_names() -> Dict[str, str]:
    from DailyUpdates.storage import SQLiteStorage
    from yg_quant_repo import default_db_path

    basic = SQLiteStorage(str(default_db_path())).read_etf_basic()
    if basic is None or basic.empty:
        return {}
    out: Dict[str, str] = {}
    for row in basic.itertuples(index=False):
        out[str(row.symbol)] = str(getattr(row, "name", "") or "")
    return out


class WufuEtf(Strategy):
    name = "wufu"

    @classmethod
    def from_cli(cls, args, allocator):
        n = args.n if getattr(args, "n", None) is not None else DEFAULT_N
        spec = spec_from_cli(getattr(args, "rebalance", None) or DEFAULT_REBALANCE)
        return cls(n=n, allocator=allocator, rebalance=spec)

    @classmethod
    def cli_fields(cls):
        return [n_field(DEFAULT_N), rebalance_field(DEFAULT_REBALANCE)]

    @classmethod
    def panel_kwargs(cls, args) -> dict:
        del args
        return {
            "asset_class": "etf",
            "market_fields": ("close", "vol", "amount"),
            "factors": (),
        }

    @classmethod
    def cost_kwargs(cls, args=None) -> dict:
        """场内基金：沿用全局万一佣金 / 滑点 / 最低 5 元，印花税为 0。"""
        del args
        return {
            "commission": 0.0001,
            "stamp": 0.0,
            "slippage": 0.00258,
            "min_commission": 5.0,
        }

    @classmethod
    def warmup(cls, args) -> int:
        del args
        return WARMUP_DAYS

    @classmethod
    def run_tag(cls, args) -> str:
        n = args.n if getattr(args, "n", None) is not None else DEFAULT_N
        spec = parse_rebalance(getattr(args, "rebalance", None) or DEFAULT_REBALANCE)
        return f"wufu_h{int(n)}{spec.tag_suffix()}"

    def __init__(
        self,
        n: int = DEFAULT_N,
        allocator: Allocator | None = None,
        rebalance: str | RebalanceSpec = DEFAULT_REBALANCE,
        names: Optional[Mapping[str, str]] = None,
        index_close: Optional[Mapping[str, pd.Series]] = None,
    ):
        self.n = max(1, int(n))
        self.allocator = allocator or EqualWeight()
        self.rebalance = (
            parse_rebalance(rebalance) if not isinstance(rebalance, RebalanceSpec) else rebalance
        )
        self._gate = RebalanceGate(self.rebalance)
        self._names: Dict[str, str] = dict(names) if names is not None else {}
        self._index_close: Dict[str, pd.Series] = dict(index_close) if index_close is not None else {}
        self._injected = names is not None
        self._index_injected = index_close is not None
        self._ready = False
        self._weak = WeakState()
        self._index: Dict[str, int] = {}
        self._weak_index_warned = False

    def score(self, ctx: DayContext) -> TargetHoldings:
        held = self._gate.skip(ctx)
        if held is not None:
            return held
        self._ensure(ctx)
        universe = ctx.universe()
        closes = np.asarray(ctx.history("close", LOOKBACK_DAYS + 1), dtype=np.float64)
        vols = np.asarray(ctx.history("vol", VOLUME_LOOKBACK + 1), dtype=np.float64)
        amounts = np.asarray(ctx.history("amount", LIQ_DAYS), dtype=np.float64)
        threshold = liquidity_threshold(amounts)
        avg_money = {ctx.symbols[i]: float(v) for i, v in enumerate(avg_daily_yuan(amounts))}
        below, above, n_valid = self._weak_counts(ctx.asof)
        need = weak_vote_need(n_valid)
        self._weak = step_weak_period(
            self._weak, ctx.asof, ctx._store.dates, below, above, need=need
        )
        pool = self._pool(ctx.symbols, universe, avg_money, threshold)
        rows = self._metrics(ctx.symbols, pool, universe, closes, vols)
        filtered = apply_filters(rows, is_weak=self._weak.is_weak)
        ratio = 1.0 if self._weak.is_weak else SCORE_RATIO
        holdings = [ctx.symbols[i] for i, w in enumerate(ctx.position) if w > 1e-9]
        targets = select_targets(filtered, holdings, self.n, ratio)
        if not targets:
            loc = self._index.get(DEFENSIVE)
            if loc is not None and universe[loc] and np.isfinite(closes[-1, loc]):
                targets = [DEFENSIVE]
                logger.info("调仓 %s 无标的，防御 %s", ctx.asof, DEFENSIVE)
            else:
                logger.info("调仓 %s 无标的且防御不可用，空仓", ctx.asof)
                return self._gate.remember(
                    TargetHoldings.from_array(
                        ctx.asof, ctx.execute_on, ctx.symbols, np.zeros(len(ctx.symbols))
                    )
                )
        picks = np.asarray([self._index[s] for s in targets if s in self._index], dtype=np.int64)
        if picks.size == 0:
            return self._gate.remember(
                TargetHoldings.from_array(
                    ctx.asof, ctx.execute_on, ctx.symbols, np.zeros(len(ctx.symbols))
                )
            )
        weights = self.allocator.size(ctx, picks)
        logger.info(
            "调仓 %s %s 池=%s 过滤=%s 目标=%s",
            ctx.asof,
            "走弱" if self._weak.is_weak else "正常",
            int(np.asarray(pool, dtype=bool).sum()),
            len(filtered),
            [ctx.symbols[i] for i in picks],
        )
        return self._gate.remember(
            TargetHoldings.from_array(ctx.asof, ctx.execute_on, ctx.symbols, weights)
        )

    def _ensure(self, ctx: DayContext) -> None:
        if self._ready:
            return
        self._index = {str(sym): i for i, sym in enumerate(ctx.symbols)}
        if not self._injected:
            try:
                self._names = _load_etf_names()
            except Exception:
                logger.warning("etf_basic 名称加载失败，动态池将为空")
                self._names = {}
        if not self._index_injected:
            loader = MarketPanelLoader()
            loaded: Dict[str, pd.Series] = {}
            for label, symbol in WEAK_INDEXES:
                try:
                    loaded[label] = loader.load_index_close(symbol)
                except Exception:
                    loaded[label] = pd.Series(dtype="float64")
            self._index_close = loaded
        self._ready = True

    def _weak_counts(self, asof: str) -> Tuple[int, int, int]:
        above = 0
        below = 0
        ready = 0
        missing: List[str] = []
        for label, _symbol in WEAK_INDEXES:
            series = self._index_close.get(label)
            flag = index_vs_ma(series, asof, WEAK_MA)
            if flag is True:
                above += 1
                ready += 1
            elif flag is False:
                below += 1
                ready += 1
            elif series is None or getattr(series, "empty", True):
                missing.append(label)
            else:
                work = pd.to_numeric(series, errors="coerce")
                work.index = pd.to_datetime(work.index).strftime("%Y-%m-%d")
                work = work[work.index <= asof].dropna()
                if len(work) < WEAK_MA:
                    missing.append(label)
                else:
                    ready += 1
        if missing and not self._weak_index_warned:
            self._weak_index_warned = True
            logger.warning(
                "走弱指数历史不足 %s，有效 %s/4，表决门槛 %s（原文需 3/4）",
                missing,
                ready,
                weak_vote_need(ready),
            )
        return below, above, ready

    def _pool(
        self,
        symbols: Sequence[str],
        universe: np.ndarray,
        avg_money: Mapping[str, float],
        threshold: float,
    ) -> np.ndarray:
        listed = {str(s) for s, ok in zip(symbols, universe) if ok}
        def liquid(codes: Iterable[str]) -> List[str]:
            return [
                c
                for c in codes
                if c in listed and float(avg_money.get(c, 0.0) or 0.0) > threshold
            ]

        if self._weak.is_weak:
            picked = liquid(GLOBAL_POOL) or [c for c in GLOBAL_POOL if c in listed]
        else:
            fixed = liquid(FIXED_POOL)
            names = {code: self._names.get(code, "") for code in listed if self._names.get(code)}
            dynamic = dynamic_sector_pool(names, avg_money, threshold)
            picked = sorted(set(fixed) | set(dynamic))
        out = np.zeros(len(symbols), dtype=bool)
        for code in picked:
            loc = self._index.get(code)
            if loc is not None:
                out[loc] = True
        return out

    def _metrics(
        self,
        symbols: Sequence[str],
        pool: np.ndarray,
        universe: np.ndarray,
        closes: np.ndarray,
        vols: np.ndarray,
    ) -> List[EtfMetrics]:
        rows: List[EtfMetrics] = []
        today_close = closes[-1]
        today_vol = vols[-1]
        past_vol = vols[:-1]
        for i, in_pool in enumerate(pool):
            if not in_pool or not universe[i]:
                continue
            px = closes[:, i]
            if not np.isfinite(today_close[i]) or today_close[i] <= 0:
                continue
            score, _ann, r2 = momentum_score(px, LOOKBACK_DAYS)
            if score is None or r2 is None:
                continue
            vol_ratio = volume_ratio_daily(past_vol[:, i], float(today_vol[i]))
            rows.append(
                EtfMetrics(
                    symbol=str(symbols[i]),
                    name=self._names.get(str(symbols[i]), str(symbols[i])),
                    score=float(score),
                    r2=float(r2),
                    volume_ratio=vol_ratio,
                    passed_momentum=MIN_SCORE <= float(score) <= MAX_SCORE,
                    passed_r2=float(r2) > R2_THRESHOLD,
                    passed_ma=passed_ma_filter(px),
                    passed_volume=vol_ratio is not None and vol_ratio < VOLUME_THRESHOLD,
                    passed_loss=passed_loss_filter(px),
                )
            )
        return rows
