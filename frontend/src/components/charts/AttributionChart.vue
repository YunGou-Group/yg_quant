<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { PLOT_COLORS } from "./plotTheme.js";
import { asXY, pickSeries, styleLabel } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

const cum = computed(() =>
  pickSeries(props.series, ["cum_factor", "cum_explained", "cum_attr_industry", "cum_residual"], {
    cum_factor: "累计因子收益 λ",
    cum_explained: "累计风格+行业",
    cum_attr_industry: "累计行业归因",
    cum_residual: "累计残差",
  }).concat(
    ["style_size", "style_momentum", "style_beta", "style_liquidity"].map((key) => {
      const item = props.series[`cum_attr_${key}`];
      const xy = asXY(item);
      if (!xy.x.length) return null;
      return { ...xy, name: `累计 ${styleLabel(key)}`, mode: "lines", opacity: 0.7 };
    }).filter(Boolean)
  )
);

const styleRet = computed(() => {
  const s = props.series || {};
  const keys = Object.keys(s)
    .filter((key) => key.startsWith("ma_f_"))
    .map((key) => key.slice("ma_f_".length))
    .filter((key) => !key.startsWith("ind_"));
  const out = [];
  keys.forEach((styleKey, index) => {
    const color = PLOT_COLORS[index % PLOT_COLORS.length];
    const label = styleLabel(styleKey);
    const daily = asXY(s[`f_${styleKey}`]);
    const ma = asXY(s[`ma_f_${styleKey}`]);
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
    }
  });
  return out;
});
</script>

<template>
  <div class="stack">
    <ChartPanel
      title="风格收益 f_t"
      :traces="styleRet"
      :layout="{ yaxis: { tickformat: '.2%', gridcolor: '#2c3a4f' } }"
    />
    <ChartPanel
      title="因子归因"
      :traces="cum"
      :layout="{ yaxis: { tickformat: '.2%', gridcolor: '#2c3a4f' } }"
    />
  </div>
</template>

<style scoped>
.stack { display: contents; }
</style>
