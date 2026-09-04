<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { meanOf, prefixSeries } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

const lines = computed(() => prefixSeries(props.series, "ic_decay_"));

const bars = computed(() => {
  const keys = Object.keys(props.series || {})
    .filter((key) => key.startsWith("ic_decay_"))
    .sort((a, b) => Number(a.replace(/\D/g, "")) - Number(b.replace(/\D/g, "")));
  if (!keys.length) return [];
  const x = keys.map((key) => key.replace("ic_decay_", ""));
  const y = keys.map((key) => meanOf(props.series[key]));
  if (y.every((v) => v == null)) return [];
  return [{ x, y, type: "bar", name: "均值 RankIC" }];
});
</script>

<template>
  <div class="stack">
    <ChartPanel title="IC 衰减（时序）" :traces="lines" />
    <ChartPanel title="IC 衰减（各 horizon 均值）" :traces="bars" />
  </div>
</template>

<style scoped>
.stack { display: contents; }
</style>
