<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { prefixSeries } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });
const traces = computed(() => prefixSeries(props.series, "quantile_turnover_"));
</script>

<template>
  <ChartPanel title="分位换手" :traces="traces" />
</template>
