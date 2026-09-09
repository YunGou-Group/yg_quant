<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { PLOT_COLORS } from "./plotTheme.js";
import { asXY, orderedStyleKeys, prefixSeries, styleLabel } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

const traces = computed(() => {
  const s = props.series || {};
  const keys = orderedStyleKeys(s, "factor_style_correlation_");
  return keys
    .map((styleKey, index) => {
      const xy = asXY(s[`factor_style_correlation_${styleKey}`]);
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

const lsExposure = computed(() => [
  ...prefixSeries(props.series, "long_short_barra_exposure_"),
  ...prefixSeries(props.series, "barra_exposure_Q"),
]);

const lsCorr = computed(() => [
  ...prefixSeries(props.series, "long_short_barra_corr_"),
  ...prefixSeries(props.series, "barra_corr_Q"),
]);
</script>

<template>
  <div class="stack">
    <ChartPanel title="风格秩相关" :traces="traces" />
    <ChartPanel v-if="lsExposure.length" title="分位/多空 Barra 暴露" :traces="lsExposure" />
    <ChartPanel v-if="lsCorr.length" title="分位/多空 Barra 相关" :traces="lsCorr" />
  </div>
</template>

<style scoped>
.stack { display: contents; }
</style>
