<script setup>
import { onMounted, ref, watch } from "vue";
import { plotLayout } from "./plotTheme.js";

const props = defineProps({
  title: { type: String, default: "" },
  traces: { type: Array, default: () => [] },
  layout: { type: Object, default: () => ({}) },
  height: { type: Number, default: 300 },
});

const el = ref(null);
let Plotly = null;

async function ensurePlotly() {
  if (!Plotly) {
    Plotly = (await import("plotly.js-dist-min")).default;
  }
  return Plotly;
}

async function render() {
  if (!el.value || !props.traces?.length) return;
  const plot = await ensurePlotly();
  if (!el.value) return;
  plot.react(
    el.value,
    props.traces,
    plotLayout(props.title, { height: props.height, ...props.layout }),
    { responsive: true }
  );
}

onMounted(render);
watch(() => props.traces, render, { deep: true });
watch(() => props.title, render);
watch(() => props.layout, render, { deep: true });
</script>

<template>
  <div v-show="traces.length" ref="el" class="chart"></div>
</template>

<style scoped>
.chart {
  width: 100%;
  min-height: 240px;
}
</style>
