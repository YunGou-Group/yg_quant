<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { matchSeries, prefixSeries } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

const returns = computed(() => {
  const nav = matchSeries(props.series, (key) => /^Q\d+$/.test(key));
  if (nav.length) return nav;
  return prefixSeries(props.series, "quantile_returns_");
});

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
    ...prefixSeries(props.series, "quantile_ic_Q", { rename }),
    ...prefixSeries(props.series, "quantile_rank_ic_Q", { rename }),
    ...prefixSeries(props.series, "quantile_top_hit_Q", { rename }),
    ...prefixSeries(props.series, "quantile_bot_hit_Q", { rename }),
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
    <ChartPanel title="分位 IC / 命中" :traces="extra" />
  </div>
</template>

<style scoped>
.stack { display: contents; }
</style>
