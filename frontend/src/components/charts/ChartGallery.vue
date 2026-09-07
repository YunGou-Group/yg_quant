<script setup>
import AttributionChart from "./AttributionChart.vue";
import BarraExposureChart from "./BarraExposureChart.vue";
import ConsistencyChart from "./ConsistencyChart.vue";
import CrowdingChart from "./CrowdingChart.vue";
import DistributionChart from "./DistributionChart.vue";
import IcDecayChart from "./IcDecayChart.vue";
import IcSeriesChart from "./IcSeriesChart.vue";
import MarketChart from "./MarketChart.vue";
import MonotonicityChart from "./MonotonicityChart.vue";
import PnlChart from "./PnlChart.vue";
import QualityChart from "./QualityChart.vue";
import QuantileReturnsChart from "./QuantileReturnsChart.vue";
import SizeStratChart from "./SizeStratChart.vue";
import StyleCorrelationChart from "./StyleCorrelationChart.vue";
import TailCharts from "./TailCharts.vue";
import TurnoverChart from "./TurnoverChart.vue";

const props = defineProps({
  series: { type: Object, default: () => ({}) },
  visibleIds: { type: Object, default: null },
});

function show(id) {
  return !props.visibleIds || props.visibleIds.has(id);
}

function showAny(ids) {
  return ids.some((id) => show(id));
}
</script>

<template>
  <div class="gallery">
    <MarketChart v-if="show('market')" :series="series" />
    <IcSeriesChart v-if="show('ic_series')" :series="series" />
    <DistributionChart v-if="show('ic_hist')" :series="series" />
    <IcDecayChart v-if="show('ic_decay')" :series="series" />
    <QuantileReturnsChart v-if="show('quantile_returns')" :series="series" />
    <TurnoverChart v-if="show('turnover')" :series="series" />
    <MonotonicityChart v-if="show('monotonicity')" :series="series" />
    <ConsistencyChart v-if="show('consistency')" :series="series" />
    <PnlChart v-if="show('pnl')" :series="series" />
    <SizeStratChart v-if="show('size_strat')" :series="series" />
    <CrowdingChart v-if="show('crowding')" :series="series" />
    <TailCharts v-if="show('tail')" :series="series" />
    <QualityChart v-if="showAny(['quality', 'factor_stats', 'factor_autocorr'])" :series="series" />
    <StyleCorrelationChart v-if="show('style_corr')" :series="series" />
    <BarraExposureChart v-if="show('barra_exposure')" :series="series" />
    <AttributionChart v-if="show('attribution')" :series="series" />
  </div>
</template>

<style scoped>
.gallery {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
@media (min-width: 1400px) {
  .gallery {
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
}
@media (max-width: 1100px) {
  .gallery {
    grid-template-columns: 1fr;
  }
}
</style>
