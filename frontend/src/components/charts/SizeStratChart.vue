<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { cumNavFromReturns, lineTrace, matchSeries, pickSeries } from "./seriesUtils.js";

const props = defineProps({
  series: { type: Object, default: () => ({}) },
  horizon: { type: [Number, String], default: 1 },
});

const LABELS = {
  size_ls: "市值分层多空",
  size_long: "市值分层多头",
  size_short: "市值分层空头",
  size_rank_ls: "排名分层多空",
  size_rank_long: "排名分层多头",
  size_rank_short: "排名分层空头",
};

const main = computed(() =>
  pickSeries(props.series, Object.keys(LABELS), LABELS)
);

const layers = computed(() =>
  matchSeries(props.series, (key) => /^size_ls_S\d+$/.test(key) || /^size_rank_ls_R\d+$/.test(key))
);

/** 与分位净值同口径：N 日层内多空收益 → 折日 → 累计净值。 */
const cum = computed(() => {
  const keys = ["size_ls", "size_rank_ls"];
  const out = [];
  for (const key of keys) {
    const item = props.series?.[key];
    if (!item) continue;
    const nav = cumNavFromReturns(item, props.horizon);
    const trace = lineTrace(nav, `${LABELS[key] || key}（累计）`);
    if (trace) out.push(trace);
  }
  return out;
});
</script>

<template>
  <div class="stack">
    <ChartPanel title="市值分层多空" :traces="main" />
    <ChartPanel v-if="cum.length" title="市值分层多空累计" :traces="cum" />
    <ChartPanel title="市值分层（各层）" :traces="layers" />
  </div>
</template>

<style scoped>
.stack { display: contents; }
</style>
