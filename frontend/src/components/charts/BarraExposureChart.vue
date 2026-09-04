<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { PLOT_COLORS } from "./plotTheme.js";
import { asXY, styleLabel } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

const traces = computed(() => {
  const s = props.series || {};
  const fromMa = Object.keys(s)
    .filter((key) => key.startsWith("ma_beta_"))
    .map((key) => key.slice("ma_beta_".length));
  const fromBeta = Object.keys(s)
    .filter((key) => key.startsWith("beta_") && !key.startsWith("beta_ind_"))
    .map((key) => key.slice("beta_".length));
  const keys = (fromMa.length ? fromMa : fromBeta).filter((key) => !key.startsWith("ind_"));
  const out = [];
  keys.forEach((styleKey, index) => {
    const color = PLOT_COLORS[index % PLOT_COLORS.length];
    const label = styleLabel(styleKey);
    const daily = asXY(s[`beta_${styleKey}`]);
    const ma = asXY(s[`ma_beta_${styleKey}`]);
    if (daily.x.length) {
      out.push({
        ...daily,
        name: `${label} 日频`,
        legendgroup: label,
        showlegend: false,
        mode: "lines",
        line: { color, width: 1 },
        opacity: 0.25,
      });
    }
    if (ma.x.length) {
      out.push({
        ...ma,
        name: `${label} 均线`,
        legendgroup: label,
        mode: "lines",
        line: { color, width: 2 },
      });
    } else if (daily.x.length) {
      out.push({ ...daily, name: label, mode: "lines", line: { color, width: 2 } });
    }
  });
  return out;
});
</script>

<template>
  <ChartPanel title="Barra 暴露 β" :traces="traces" />
</template>
