<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { plotLayout } from "./plotTheme.js";

const props = defineProps({
  title: { type: String, default: "" },
  traces: { type: Array, default: () => [] },
  height: { type: Number, default: 280 },
});

const el = ref(null);
const has = computed(() => (props.traces || []).length > 0);
let Plotly = null;

async function ensurePlotly() {
  if (!Plotly) {
    Plotly = (await import("plotly.js-dist-min")).default;
  }
  return Plotly;
}

async function render() {
  if (!el.value || !has.value) return;
  const plot = await ensurePlotly();
  if (!el.value) return;
  plot.react(el.value, props.traces, plotLayout(props.title, { height: props.height }), { responsive: true });
}

onMounted(render);
watch(() => props.traces, render, { deep: true });
watch(() => props.title, render);
</script>

<template>
  <div v-show="has" ref="el" class="chart"></div>
</template>

<style scoped>
.chart {
  width: 100%;
  min-height: 240px;
}
</style>
