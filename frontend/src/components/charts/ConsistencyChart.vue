<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { pickSeries } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });
const traces = computed(() =>
  pickSeries(props.series, ["long_short_term_consistency", "consistency", "ma_fast", "ma_slow"], {
    long_short_term_consistency: "长短窗一致性",
    consistency: "一致性",
    ma_fast: "短窗 RankIC",
    ma_slow: "长窗 RankIC",
  })
);
</script>

<template>
  <ChartPanel title="长短窗一致性" :traces="traces" />
</template>
