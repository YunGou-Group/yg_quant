export function asXY(item) {
  if (!item) return { x: [], y: [] };
  if (Array.isArray(item.dates) || Array.isArray(item.x)) {
    return { x: item.dates || item.x || [], y: item.values || item.y || [] };
  }
  return { x: [], y: [] };
}

export function hasPoints(item) {
  return asXY(item).x.length > 0;
}

export function lineTrace(item, name, extra = {}) {
  const xy = asXY(item);
  if (!xy.x.length) return null;
  return { ...xy, name, mode: extra.mode || "lines", ...extra };
}

export function pickSeries(series, names, extras = {}) {
  const out = [];
  for (const name of names) {
    const trace = lineTrace(series[name], extras[name] || name);
    if (trace) out.push(trace);
  }
  return out;
}

export function prefixSeries(series, prefix, extra = {}) {
  return Object.keys(series || {})
    .filter((key) => key.startsWith(prefix))
    .sort()
    .map((key) => lineTrace(series[key], extra.rename ? extra.rename(key) : key, extra.trace || {}))
    .filter(Boolean);
}

export function matchSeries(series, pred, extra = {}) {
  return Object.keys(series || {})
    .filter(pred)
    .sort()
    .map((key) => lineTrace(series[key], extra.rename ? extra.rename(key) : key, extra.trace || {}))
    .filter(Boolean);
}

export function meanOf(item) {
  const y = asXY(item).y.filter((value) => value != null && Number.isFinite(Number(value)));
  if (!y.length) return null;
  return y.reduce((sum, value) => sum + Number(value), 0) / y.length;
}

export function flattenEvalResult(data) {
  const out = {};
  if (!data) return out;
  const metrics = data.metrics || {};
  for (const [metricName, payload] of Object.entries(metrics)) {
    const series = payload?.series || {};
    for (const [key, xy] of Object.entries(series)) {
      const item = {
        dates: xy?.dates || [],
        values: xy?.values || [],
        metric: key,
      };
      if (!item.dates.length) continue;
      out[`${metricName}__${key}`] = item;
      if (!out[key]) out[key] = item;
    }
  }
  const aliases = [
    ["daily_rank_ic", "rank_ic"],
    ["daily_ic", "ic"],
    ["daily_pure_rank_ic", "pure_ic"],
    ["consistency", "long_short_term_consistency"],
  ];
  for (const [from, to] of aliases) {
    if (out[from] && !out[to]) out[to] = { ...out[from], metric: to };
  }
  if (data.market?.series) {
    const market = data.market.series;
    if (market.cum_ret) out.market_cum = market.cum_ret;
    if (market.daily_ret) out.market_daily = market.daily_ret;
    if (market.drawdown) out.market_dd = market.drawdown;
    out.market_label = { dates: [], values: [], metric: data.market.label || "沪深300" };
  }
  return out;
}

/** 与 FactorEvaluates.exposure_engine.STYLE_LABELS 对齐（含 NLSIZE）。 */
export const BARRA_STYLE_KEYS = [
  "style_size",
  "nlsize",
  "style_beta",
  "style_momentum",
  "style_resvol",
  "style_liquidity",
  "style_btop",
  "style_ey",
  "style_growth",
  "style_leverage",
];

export const BARRA_STYLE_LABELS = {
  style_size: "Size",
  nlsize: "NLSIZE",
  style_beta: "Beta",
  style_momentum: "Momentum",
  style_resvol: "ResVol",
  style_liquidity: "Liquidity",
  style_btop: "BTOP",
  style_ey: "EY",
  style_growth: "Growth",
  style_leverage: "Leverage",
};

export function styleKeyOf(raw) {
  return String(raw || "")
    .replace(/^ma_beta_/, "")
    .replace(/^beta_/, "")
    .replace(/^ma_f_/, "")
    .replace(/^f_/, "")
    .replace(/^cum_attr_/, "")
    .replace(/^attr_/, "")
    .replace(/^factor_style_correlation_/, "");
}

export function styleLabel(key) {
  const bare = styleKeyOf(key);
  return BARRA_STYLE_LABELS[bare] || bare.replace(/^style_/, "");
}

/** 按 Barra 固定顺序收集序列里出现的风格键（跳过行业）。 */
export function orderedStyleKeys(series, prefix) {
  const found = new Set();
  for (const key of Object.keys(series || {})) {
    if (!key.startsWith(prefix)) continue;
    const styleKey = key.slice(prefix.length);
    if (!styleKey || styleKey.startsWith("ind_")) continue;
    found.add(styleKey);
  }
  const ordered = BARRA_STYLE_KEYS.filter((k) => found.has(k));
  const rest = [...found].filter((k) => !BARRA_STYLE_KEYS.includes(k)).sort();
  return ordered.concat(rest);
}
