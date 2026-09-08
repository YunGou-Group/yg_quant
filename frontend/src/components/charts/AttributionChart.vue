<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { PLOT_COLORS } from "./plotTheme.js";
import {
  asXY,
  orderedStyleKeys,
  pickSeries,
  styleLabel,
} from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

const cum = computed(() => {
  const s = props.series || {};
  const totals = pickSeries(
    s,
    ["cum_factor", "cum_explained", "cum_attr_industry", "cum_residual"],
    {
      cum_factor: "累计因子收益 λ",
      cum_explained: "累计风格+行业",
      cum_attr_industry: "累计行业归因",
      cum_residual: "累计残差",
    }
  );
  const styles = orderedStyleKeys(s, "cum_attr_").map((styleKey, index) => {
    const xy = asXY(s[`cum_attr_${styleKey}`]);
    if (!xy.x.length) return null;
    return {
      ...xy,
      name: `累计 ${styleLabel(styleKey)}`,
      mode: "lines",
      opacity: 0.85,
      line: {
        color: PLOT_COLORS[(index + 4) % PLOT_COLORS.length],
        width: 1.5,
      },
    };
  });
  return totals.concat(styles.filter(Boolean));
});

/** 风格收益只画均线，完整 10 腿。 */
const styleRet = computed(() => {
  const s = props.series || {};
  const keys = orderedStyleKeys(s, "ma_f_").length
    ? orderedStyleKeys(s, "ma_f_")
    : orderedStyleKeys(s, "f_");
  return keys
    .map((styleKey, index) => {
      const ma = asXY(s[`ma_f_${styleKey}`]);
      const daily = asXY(s[`f_${styleKey}`]);
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
  <div class="stack">
    <ChartPanel
      title="风格收益 f_t（均线）"
      :traces="styleRet"
      :layout="{ yaxis: { tickformat: '.2%', gridcolor: '#2c3a4f' } }"
    />
    <ChartPanel
      title="因子归因（全风格）"
      :traces="cum"
      :layout="{ yaxis: { tickformat: '.2%', gridcolor: '#2c3a4f' } }"
    />
  </div>
</template>

<style scoped>
.stack { display: contents; }
</style>
