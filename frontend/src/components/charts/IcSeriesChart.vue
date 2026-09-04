<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { asXY, lineTrace } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

const traces = computed(() => {
  const s = props.series || {};
  const wanted = [
    ["rank_ic", "RankIC"],
    ["daily_rank_ic", "RankIC"],
    ["ic", "Pearson IC"],
    ["daily_ic", "Pearson IC"],
    ["pure_ic", "纯化 RankIC"],
    ["daily_pure_rank_ic", "纯化 RankIC"],
    ["rolling_ic", "滚动 IC"],
    ["weighted_ic", "加权 IC"],
    ["nonlinear_ic", "非线性 IC"],
    ["mi_ic", "互信息 IC"],
  ];
  const seen = new Set();
  const out = [];
  for (const [key, label] of wanted) {
    if (seen.has(label)) continue;
    const trace = lineTrace(s[key], label);
    if (trace) {
      seen.add(label);
      out.push(trace);
    }
  }
  const cum = lineTrace(s.rank_ic__cumsum || s.cumsum, "累计 RankIC", { yaxis: "y2" });
  if (cum) out.push(cum);
  return out.filter((t) => asXY(t).x.length);
});
</script>

<template>
  <ChartPanel
    title="IC 序列"
    :traces="traces"
    :layout="{ yaxis2: { overlaying: 'y', side: 'right', gridcolor: '#2c3a4f', showgrid: false } }"
  />
</template>
