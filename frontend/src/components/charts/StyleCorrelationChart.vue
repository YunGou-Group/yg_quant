<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { PLOT_COLORS } from "./plotTheme.js";
import { asXY, orderedStyleKeys, styleLabel } from "./seriesUtils.js";

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
</script>

<template>
  <ChartPanel title="风格秩相关" :traces="traces" />
</template>
