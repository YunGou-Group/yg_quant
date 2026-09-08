<script setup>
import { computed } from "vue";
import ChartGallery from "./charts/ChartGallery.vue";
import {
  BARRA_STYLE_KEYS,
  BARRA_STYLE_LABELS,
  flattenEvalResult,
} from "./charts/seriesUtils.js";
import { icirValue, fmt } from "../format.js";

const props = defineProps({
  data: { type: Object, default: null },
  schema: { type: Object, default: null },
});

function fieldHint(metricName, fieldName) {
  const metric = (props.schema?.metrics || []).find((item) => item.name === metricName);
  const field = (metric?.fields || []).find((item) => item.name === fieldName);
  if (!field) return "";
  return `${field.dimension}：${field.meaning}`;
}

const series = computed(() => flattenEvalResult(props.data));

function exposureStyleItems(scalars) {
  if (!scalars) return [];
  return BARRA_STYLE_KEYS.map((key) => {
    const value = scalars[`ma_abs_${key}`] ?? scalars[`mean_abs_${key}`];
    const label = BARRA_STYLE_LABELS[key] || key;
    return {
      label: `|maβ| ${label}`,
      value,
      title: fieldHint("exposure", `ma_abs_${key}`),
    };
  });
}

function attributionStyleItems(scalars) {
  if (!scalars) return [];
  return BARRA_STYLE_KEYS.map((key) => {
    const value = scalars[`cum_attr_${key}`];
    const label = BARRA_STYLE_LABELS[key] || key;
    return {
      label: `累计 ${label}`,
      value,
      title: fieldHint("attribution", `cum_attr_${key}`),
    };
  });
}

const kpis = computed(() => {
  const data = props.data;
  if (!data) return [];
  const metrics = data.metrics || {};
  const ic = metrics.rank_ic || {};
  const icir = metrics.icir || {};
  const trend = metrics.long_short_term_consistency || {};
  const quantile = metrics.quantile || {};
  const coverage = metrics.coverage_rate || {};
  const exp = metrics.exposure?.scalars || {};
  const attr = metrics.attribution?.scalars || {};
  const market = data.market || {};
  const marketLabel = market.label || "沪深300";
  const marketKpis = data.market
    ? [
        {
          label: `${marketLabel}区间`,
          value: market.scalars?.cum_last,
          title: "评估窗内指数收盘累计收益，只表示市场状态，不是超额、也不是因子远期收益",
        },
        {
          label: "大盘最大回撤",
          value: market.scalars?.max_drawdown,
          title: "评估窗内指数从高点回撤的最小值",
        },
      ]
    : [];
  const groups = [
    {
      title: "预测力",
      items: [
        { label: `RankIC均值(${data.horizon})`, value: ic.scalars?.mean, title: fieldHint("rank_ic", "mean") },
        { label: "ICIR", value: icirValue(ic, icir), title: fieldHint("icir", "icir") },
        { label: "纯化IC均值", value: metrics.pure_ic?.scalars?.mean, title: fieldHint("pure_ic", "mean") },
        { label: "纯化ICIR", value: metrics.pure_ic?.scalars?.icir, title: fieldHint("pure_ic", "icir") },
        { label: "分层价差", value: quantile.scalars?.spread, title: fieldHint("quantile", "spread") },
        { label: "单调性", value: quantile.scalars?.monotonicity, title: fieldHint("quantile", "monotonicity") },
      ],
    },
    { title: "市场", items: marketKpis },
    {
      title: "风格暴露",
      items: [
        ...exposureStyleItems(exp),
        { label: "行业哑变量", value: exp.n_industries, title: fieldHint("exposure", "n_industries") },
        { label: "|maβ| 行业", value: exp.ma_abs_industry, title: fieldHint("exposure", "ma_abs_industry") },
        { label: "主行业", value: exp.top_industry, title: fieldHint("exposure", "top_industry") },
        { label: "|maβ| 主行业", value: exp.ma_abs_top_industry, title: fieldHint("exposure", "ma_abs_top_industry") },
      ],
    },
    {
      title: "风格归因",
      items: [
        { label: "累计因子收益", value: attr.cum_factor, title: fieldHint("attribution", "cum_factor") },
        { label: "累计风格归因", value: attr.cum_explained, title: fieldHint("attribution", "cum_explained") },
        { label: "累计行业归因", value: attr.cum_attr_industry, title: fieldHint("attribution", "cum_attr_industry") },
        { label: "累计残差", value: attr.cum_residual, title: fieldHint("attribution", "cum_residual") },
        { label: "残差占比", value: attr.residual_share, title: fieldHint("attribution", "residual_share") },
        ...attributionStyleItems(attr),
      ],
    },
    {
      title: "时效 / 覆盖",
      items: [
        { label: "RankIC正值比例", value: ic.scalars?.positive_ratio, title: fieldHint("rank_ic", "positive_ratio") },
        { label: "长短窗一致性", value: trend.scalars?.last ?? trend.scalars?.mean, title: fieldHint("long_short_term_consistency", "last") },
        { label: "覆盖率", value: coverage.scalars?.mean ?? coverage.scalars?.coverage_mean, title: fieldHint("coverage_rate", "mean") },
      ],
    },
  ];
  return groups
    .map((group) => ({
      ...group,
      items: group.items.filter((item) => {
        if (item.value == null || item.value === "") return false;
        if (typeof item.value === "string") return true;
        return !Number.isNaN(item.value);
      }),
    }))
    .filter((group) => group.items.length);
});
</script>

<template>
  <div>
    <div class="kpi-groups">
      <section v-for="group in kpis" :key="group.title" class="kpi-group">
        <div class="kpi-title">{{ group.title }}</div>
        <div class="kpis">
          <div v-for="item in group.items" :key="item.label" class="kpi" :title="item.title">
            <div class="n">{{ fmt(item.value) }}</div>
            <div class="l">{{ item.label }}</div>
          </div>
        </div>
      </section>
    </div>
    <ChartGallery v-if="data" :series="series" />
  </div>
</template>

<style scoped>
.kpi-groups {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin: 8px 0 16px;
}

.kpi-group {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px 12px;
}

.kpi-title {
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
}

.kpis {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.kpi {
  background: #243044;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 14px;
  min-width: 120px;
  cursor: help;
}

.kpi .n {
  font-size: 18px;
  font-variant-numeric: tabular-nums;
}

.kpi .l {
  color: var(--muted);
  font-size: 12px;
}
</style>
