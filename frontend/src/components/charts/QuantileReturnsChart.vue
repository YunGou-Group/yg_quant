<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import {
  cumNavFromReturns,
  lineTrace,
  matchSeries,
  prefixQuantileTraces,
  shortQuantileName,
  sortQuantileKeys,
} from "./seriesUtils.js";

const props = defineProps({
  series: { type: Object, default: () => ({}) },
  horizon: { type: [Number, String], default: 1 },
});

const returns = computed(() => {
  const fromDaily = prefixQuantileTraces(props.series, "quantile_returns_Q", {
    toItem: (item) => cumNavFromReturns(item, props.horizon),
  });
  if (fromDaily.length) return fromDaily;
  const navKeys = sortQuantileKeys(
    Object.keys(props.series || {}).filter((key) => /^Q\d+$/.test(key))
  );
  return navKeys
    .map((key) => lineTrace(props.series[key], shortQuantileName(key)))
    .filter(Boolean);
});

const afterCost = computed(() =>
  prefixQuantileTraces(props.series, "quantile_returns_after_cost_", {
    toItem: (item) => cumNavFromReturns(item, props.horizon),
  })
);

const factorMean = computed(() => prefixQuantileTraces(props.series, "quantile_factor_mean_"));

const extra = computed(() => {
  const rename = (key) =>
    ({
      quantile_ic: "组内 Pearson IC",
      quantile_rank_ic: "组内 RankIC",
      quantile_return_hit: "收益命中",
      best_in_best_ratio: "顶组命中顶收益",
    })[key] ||
    key
      .replace(/^quantile_ic_/, "Pearson ")
      .replace(/^quantile_rank_ic_/, "RankIC ")
      .replace(/^quantile_top_hit_/, "顶收益命中 ")
      .replace(/^quantile_bot_hit_/, "底收益命中 ");
  const fromPrefix = [
    ...prefixQuantileTraces(props.series, "quantile_ic_Q", { rename }),
    ...prefixQuantileTraces(props.series, "quantile_rank_ic_Q", { rename }),
    ...prefixQuantileTraces(props.series, "quantile_top_hit_Q", { rename }),
    ...prefixQuantileTraces(props.series, "quantile_bot_hit_Q", { rename }),
  ];
  if (fromPrefix.length) return fromPrefix;
  return matchSeries(
    props.series,
    (key) => ["quantile_ic", "quantile_rank_ic", "quantile_return_hit", "best_in_best_ratio"].includes(key),
    { rename }
  );
});
</script>

<template>
  <div class="stack">
    <ChartPanel title="分位净值" :traces="returns" />
    <ChartPanel v-if="afterCost.length" title="分位费后净值" :traces="afterCost" />
    <ChartPanel v-if="factorMean.length" title="分位因子均值" :traces="factorMean" />
    <ChartPanel title="分位 IC / 命中" :traces="extra" />
  </div>
</template>

<style scoped>
.stack { display: contents; }
</style>
