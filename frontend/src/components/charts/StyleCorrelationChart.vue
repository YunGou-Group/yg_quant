<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { prefixSeries, styleLabel } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });
const traces = computed(() =>
  prefixSeries(props.series, "factor_style_correlation_", {
    rename: (key) => styleLabel(key),
  })
);
</script>

<template>
  <ChartPanel title="风格秩相关" :traces="traces" />
</template>
