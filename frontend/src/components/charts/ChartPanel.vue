<script setup>
import BasePlot from "./BasePlot.vue";

defineProps({
  title: { type: String, default: "" },
  traces: { type: Array, default: () => [] },
  layout: { type: Object, default: () => ({}) },
  height: { type: Number, default: 300 },
});
</script>

<template>
  <section v-if="traces.length" class="panel">
    <BasePlot :title="title" :traces="traces" :layout="layout" :height="height" />
  </section>
</template>

<style scoped>
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 8px;
  min-width: 0;
}
</style>
