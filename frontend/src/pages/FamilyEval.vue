<script setup>
import { onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { fetchFamilyEval } from "../api.js";
import { colLabel } from "../columns.js";
import { fmt } from "../format.js";
import { useRunWorkspace } from "../runWorkspace.js";

const router = useRouter();
const { runId, runLink } = useRunWorkspace();
const error = ref("");
const items = ref([]);
const start = ref("");
const end = ref("");
const mode = ref("mean");

const modes = [
  { id: "mean", label: "方向无关 mean" },
  { id: "abs", label: "带符号 abs_mean" },
  { id: "dir", label: "择腿 @dir" },
];

function keys(row) {
  const all = Object.keys(row).filter((k) => k !== "family" && k !== "n" && k !== "best_member");
  if (mode.value === "abs") return all.filter((k) => k.endsWith("_abs_mean"));
  if (mode.value === "dir") return all.filter((k) => k.includes("@dir"));
  return all.filter((k) => k.endsWith("_mean") && !k.endsWith("_abs_mean") && !k.includes("@dir"));
}

function keyLabel(key) {
  return colLabel(key.replace(/_abs_mean$/, "_mean").replace(/@dir.*/, "_mean")) || key;
}

async function load() {
  error.value = "";
  if (!runId.value) {
    items.value = [];
    return;
  }
  try {
    const data = await fetchFamilyEval(runId.value, {
      start: start.value || undefined,
      end: end.value || undefined,
    });
    items.value = data.items || [];
  } catch (err) {
    error.value = String(err.message || err);
  }
}

function openBest(row) {
  if (!row.best_member) return;
  router.push(runLink(`/detail/${encodeURIComponent(row.best_member)}`));
}

onMounted(load);
watch([runId, start, end], load);
</script>

<template>
  <div class="page">
    <h1>家族评估</h1>
    <p class="meta">方向无关取 mean；带符号取 abs_mean；两腿收益按 rank_ic 符号择腿。自定义时段从 daily 重算。</p>
    <div class="toolbar">
      <input v-model="start" type="date" />
      <input v-model="end" type="date" />
      <div class="tabs">
        <button v-for="item in modes" :key="item.id" :class="{ active: mode === item.id }" @click="mode = item.id">
          {{ item.label }}
        </button>
      </div>
    </div>
    <p v-if="error" class="status error">{{ error }}</p>
    <div class="cards">
      <article v-for="row in items" :key="row.family" class="card" @click="openBest(row)">
        <header>
          <strong>{{ row.family }}</strong>
          <span class="meta">n={{ row.n }} best={{ row.best_member || "—" }}</span>
        </header>
        <div class="kv" v-for="key in keys(row).slice(0, 12)" :key="key">
          <span>{{ keyLabel(key) }}</span>
          <b>{{ fmt(row[key]) }}</b>
        </div>
      </article>
    </div>
  </div>
</template>

<style scoped>
.page { padding: 20px 24px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 12px 0; }
.tabs { display: flex; gap: 4px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; cursor: pointer; }
header { display: flex; justify-content: space-between; margin-bottom: 8px; }
.kv { display: flex; justify-content: space-between; font-size: 12px; padding: 2px 0; color: var(--muted); }
.kv b { color: var(--text); font-weight: 500; }
</style>
