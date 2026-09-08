<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { PLOT_COLORS } from "./plotTheme.js";
import { asXY, orderedStyleKeys, styleLabel } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

/** 只画均线，避免 10 条日频叠在一起看不清。 */
const traces = computed(() => {
  const s = props.series || {};
  const keys = orderedStyleKeys(s, "ma_beta_").length
    ? orderedStyleKeys(s, "ma_beta_")
    : orderedStyleKeys(s, "beta_");
  return keys
    .map((styleKey, index) => {
      const ma = asXY(s[`ma_beta_${styleKey}`]);
      const daily = asXY(s[`beta_${styleKey}`]);
      const xy = ma.x.length ? ma : daily;
      if (!xy.x.length) return null;
      return {
        ...xy,
        name: styleLabel(styleKey),
        mode: "lines",
        line: { color: PLOT_COLORS[index % PLOT_COLORS.length], width: 2 },
      };
    })
    .filter(Boolean);
});
</script>

<template>
  <ChartPanel title="Barra 暴露 β（均线）" :traces="traces" />
</template>
