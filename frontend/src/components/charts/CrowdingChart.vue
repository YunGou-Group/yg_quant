<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { prefixSeries } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });
const traces = computed(() =>
  prefixSeries(props.series, "crowding_", {
    rename: (key) =>
      ({
        crowding_valuation_premium: "估值拥挤溢价",
        crowding_endpoint_dispersion: "端点收益离散",
      })[key] || key,
  })
);
</script>

<template>
  <ChartPanel title="拥挤度" :traces="traces" />
</template>
