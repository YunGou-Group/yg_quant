<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { asXY } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

const traces = computed(() => {
  const s = props.series || {};
  const out = [];
  const daily = asXY(s.market_daily);
  const dd = asXY(s.market_dd);
  const cum = asXY(s.market_cum);
  if (daily.x.length) {
    out.push({ ...daily, type: "bar", name: "日收益", yaxis: "y2", marker: { color: "#5b8ff9" }, opacity: 0.35 });
  }
  if (dd.x.length) {
    out.push({ ...dd, name: "回撤", mode: "lines", fill: "tozeroy", line: { color: "#e8684a", width: 1 }, opacity: 0.35 });
  }
  if (cum.x.length) {
    out.push({ ...cum, name: "区间累计", mode: "lines", line: { color: "#e8eef7", width: 2 } });
  }
  return out;
});

const title = computed(() => {
  const label = props.series?.market_label?.metric;
  return `${label || "市场"}（评估区间，收盘口径）`;
});
</script>

<template>
  <ChartPanel
    :title="title"
    :traces="traces"
    :layout="{
      yaxis: { tickformat: '.1%', gridcolor: '#2c3a4f' },
      yaxis2: { overlaying: 'y', side: 'right', tickformat: '.1%', showgrid: false },
    }"
  />
</template>
