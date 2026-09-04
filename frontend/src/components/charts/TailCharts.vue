<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { prefixSeries } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });
const traces = computed(() => prefixSeries(props.series, "tail_"));
</script>

<template>
  <ChartPanel title="尾部" :traces="traces" />
</template>
