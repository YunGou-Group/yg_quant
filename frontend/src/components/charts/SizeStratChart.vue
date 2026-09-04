<script setup>
import { computed } from "vue";
import ChartPanel from "./ChartPanel.vue";
import { matchSeries, pickSeries } from "./seriesUtils.js";

const props = defineProps({ series: { type: Object, default: () => ({}) } });

const main = computed(() =>
  pickSeries(props.series, ["size_ls", "size_long", "size_short", "size_rank_ls", "size_rank_long", "size_rank_short"], {
    size_ls: "市值分层多空",
    size_long: "市值分层多头",
    size_short: "市值分层空头",
    size_rank_ls: "排名分层多空",
    size_rank_long: "排名分层多头",
    size_rank_short: "排名分层空头",
  })
);

const layers = computed(() =>
  matchSeries(props.series, (key) => /^size_ls_S\d+$/.test(key) || /^size_rank_ls_R\d+$/.test(key))
);
</script>

<template>
  <div class="stack">
    <ChartPanel title="市值分层多空" :traces="main" />
    <ChartPanel title="市值分层（各层）" :traces="layers" />
  </div>
</template>

<style scoped>
.stack { display: contents; }
</style>
