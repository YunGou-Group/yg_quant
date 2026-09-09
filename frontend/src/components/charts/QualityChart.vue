<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { pickSeries, prefixSeries } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

const quality = computed(() =>
  pickSeries(
    props.series,
    [
      "coverage_rate",
      "missing_rate",
      "missing_samples",
      "total_samples",
      "unique_rate",
      "extreme_value_ratio",
      "iqr",
    ],
    {
      coverage_rate: "覆盖率",
      missing_rate: "缺失率",
      missing_samples: "缺失样本数",
      total_samples: "样本总数",
      unique_rate: "唯一值比例",
      extreme_value_ratio: "极端值比例",
      iqr: "IQR",
    }
  )
);

const dist = computed(() =>
  pickSeries(
    props.series,
    [
      "factor_mean",
      "factor_std",
      "factor_median",
      "factor_min",
      "factor_max",
      "factor_skewness",
      "factor_kurtosis",
    ],
    {
      factor_mean: "截面均值",
      factor_std: "截面波动",
      factor_median: "截面中位数",
      factor_min: "截面最小",
      factor_max: "截面最大",
      factor_skewness: "偏度",
      factor_kurtosis: "峰度",
    }
  )
);

const autocorr = computed(() => prefixSeries(props.series, "factor_autocorr_"));
</script>

<template>
  <div class="stack">
    <ChartPanel title="质量" :traces="quality" />
    <ChartPanel title="截面分布" :traces="dist" />
    <ChartPanel title="因子自相关" :traces="autocorr" />
  </div>
</template>

<style scoped>
.stack { display: contents; }
</style>
