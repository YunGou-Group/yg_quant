<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { asXY } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

const traces = computed(() => {
  const src = props.series.rank_ic || props.series.daily_rank_ic || props.series.ic || props.series.daily_ic;
  const y = asXY(src).y.filter((v) => v != null && Number.isFinite(Number(v)));
  if (!y.length) return [];
  return [{ x: y, type: "histogram", name: "IC 分布" }];
});
</script>

<template>
  <ChartPanel title="IC 直方图" :traces="traces" />
</template>
