<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { meanOf } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

const traces = computed(() => {
  const keys = Object.keys(props.series || {})
    .filter((key) => /^quantile_returns_Q\d+$/.test(key) || /^Q\d+$/.test(key))
    .filter((key) => key.startsWith("quantile_returns_") || !Object.keys(props.series).some((k) => k === `quantile_returns_${key}`))
    .sort((a, b) => Number(a.replace(/\D/g, "")) - Number(b.replace(/\D/g, "")));
  if (!keys.length) return [];
  const x = keys.map((key) => key.replace("quantile_returns_", ""));
  const y = keys.map((key) => meanOf(props.series[key]));
  if (y.every((v) => v == null)) return [];
  return [{ x, y, type: "bar", name: "组均收益" }];
});
</script>

<template>
  <ChartPanel title="单调性（分组均收益）" :traces="traces" />
</template>
