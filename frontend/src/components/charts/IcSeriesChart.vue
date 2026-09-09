<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { asXY, lineTrace } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

/** 全库 run 只落日度 RankIC；累计在前端 skipna 累加。 */
function cumFromDaily(item) {
  const { x, y } = asXY(item);
  if (!x.length) return null;
  let sum = 0;
  const values = y.map((v) => {
    const n = Number(v);
    if (v == null || !Number.isFinite(n)) return null;
    sum += n;
    return sum;
  });
  return { dates: x, values };
}

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
    ["nonlinear_ic_ir", "非线性 IC IR"],
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
  let cum = lineTrace(s.rank_ic__cumsum || s.cumsum, "累计 RankIC", { yaxis: "y2" });
  if (!cum) {
    cum = lineTrace(
      cumFromDaily(s.rank_ic || s.daily_rank_ic),
      "累计 RankIC",
      { yaxis: "y2" }
    );
  }
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
