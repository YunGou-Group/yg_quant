export const SUMMARY_LABELS = {
  factor_name: "因子",
  rank_ic_mean: "RankIC",
  rank_ic_ir: "RankIC IR",
  ic_mean: "Pearson IC",
  ic_ir: "Pearson ICIR",
  coverage_rate_mean: "覆盖率",
  quantile_spread_mean: "分层价差",
  long_short_term_consistency_mean: "长短窗一致性",
  weighted_pnl_mean: "加权 PnL",
  rolling_ic_mean: "滚动 IC",
  pure_ic_mean: "纯化 RankIC",
  factor_returns_mean: "因子收益",
  long_only_return_mean: "等权多头",
};

export const FILTER_COLUMNS = [
  "rank_ic_ir",
  "rank_ic_mean",
  "ic_mean",
  "coverage_rate_mean",
  "quantile_spread_mean",
  "long_short_term_consistency_mean",
  "weighted_pnl_mean",
  "rolling_ic_mean",
  "pure_ic_mean",
];

export function colLabel(key) {
  return SUMMARY_LABELS[key] || key;
}

export function familyOf(name) {
  const text = String(name || "").trim();
  if (!text) return "other";
  if (/^alpha\d+/i.test(text)) return "alpha101";
  if (text.startsWith("style_")) return "barra";
  for (const sep of ["_", "-", "."]) {
    if (text.includes(sep)) return text.split(sep)[0] || "other";
  }
  return text;
}
