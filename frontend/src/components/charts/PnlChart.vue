<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { pickSeries } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });
const traces = computed(() =>
  pickSeries(
    props.series,
    ["weighted_pnl", "weighted_long_pnl", "weighted_short_pnl", "long_only_return", "short_only_return", "factor_returns"],
    {
      weighted_pnl: "加权 PnL",
      weighted_long_pnl: "加权多头",
      weighted_short_pnl: "加权空头",
      long_only_return: "等权多头",
      short_only_return: "等权空头",
      factor_returns: "因子收益 λ",
    }
  )
);
</script>

<template>
  <ChartPanel title="PnL / 多头" :traces="traces" />
</template>
