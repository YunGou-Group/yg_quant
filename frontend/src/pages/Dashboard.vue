<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { useRouter, RouterLink } from "vue-router";
import ChartPanel from "../components/charts/ChartPanel.vue";
import { fetchFactorTable } from "../api.js";
import { colLabel } from "../columns.js";
import { fmt } from "../format.js";
import { useRunWorkspace } from "../runWorkspace.js";

const router = useRouter();
const { runId, current, runLink } = useRunWorkspace();
const error = ref("");
const items = ref([]);
const columns = ref([]);
const coverageMin = ref(0);
const search = ref("");
const sortKey = ref("rank_ic_ir");
const sortDir = ref(-1);

const previewCols = computed(() => {
  const preferred = [
    "factor_name",
    "rank_ic_mean",
    "rank_ic_ir",
    "ic_mean",
    "coverage_rate_mean",
    "quantile_spread_mean",
    "long_short_term_consistency_mean",
  ];
  const have = new Set(columns.value);
  const cols = preferred.filter((c) => have.has(c));
  if (!cols.includes("factor_name") && have.has("factor_name")) cols.unshift("factor_name");
  return cols.length ? cols : columns.value.slice(0, 8);
});

const shown = computed(() => {
  const min = Number(coverageMin.value) || 0;
  const q = search.value.trim().toLowerCase();
  let rows = items.value;
  if (min) rows = rows.filter((row) => Number(row.coverage_rate_mean || 0) >= min);
  if (q) rows = rows.filter((row) => String(row.factor_name || "").toLowerCase().includes(q));
  const key = sortKey.value;
  const dir = sortDir.value;
  return [...rows].sort((a, b) => {
    const av = a[key];
    const bv = b[key];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "string" || typeof bv === "string") {
      return dir * String(av).localeCompare(String(bv), "zh");
    }
    return dir * (Number(av) - Number(bv));
  });
});

const scatter = computed(() => {
  const rows = shown.value.filter((row) => row.rank_ic_mean != null);
  if (!rows.length) return [];
  return [
    {
      x: rows.map((row) => row.coverage_rate_mean),
      y: rows.map((row) => row.rank_ic_mean),
      text: rows.map((row) => row.factor_name),
      mode: "markers",
      type: "scatter",
      name: "因子",
    },
  ];
});

const topIr = computed(() => {
  const rows = [...shown.value]
    .filter((row) => row.rank_ic_ir != null)
    .sort((a, b) => Number(b.rank_ic_ir) - Number(a.rank_ic_ir))
    .slice(0, 15);
  if (!rows.length) return [];
  return [
    {
      x: rows.map((row) => row.factor_name),
      y: rows.map((row) => row.rank_ic_ir),
      type: "bar",
      name: "RankIC IR",
    },
  ];
});

async function load() {
  error.value = "";
  if (!runId.value) {
    items.value = [];
    columns.value = [];
    return;
  }
  try {
    const data = await fetchFactorTable(runId.value);
    items.value = data.items || [];
    columns.value = data.columns || [];
    if (!columns.value.includes(sortKey.value) && columns.value.includes("rank_ic_mean")) {
      sortKey.value = "rank_ic_mean";
    }
  } catch (err) {
    error.value = String(err.message || err);
  }
}

function toggleSort(col) {
  if (sortKey.value === col) sortDir.value *= -1;
  else {
    sortKey.value = col;
    sortDir.value = col === "factor_name" ? 1 : -1;
  }
}

function openDetail(row) {
  router.push(runLink(`/detail/${encodeURIComponent(row.factor_name)}`));
}

function evalLink(row) {
  return runLink("/eval", {
    factor: row.factor_name,
    start: current.value?.start,
    end: current.value?.end,
    horizon: current.value?.horizon,
    universe: current.value?.universe,
  });
}

onMounted(load);
watch(runId, load);
</script>

<template>
  <div class="page">
    <div class="hero">
      <div>
        <h1>因子总览</h1>
        <p class="meta">读当前全库 run 的 summary。没有数据时请先到「全库评估」提交任务。</p>
      </div>
      <div class="toolbar">
        <label class="field">
          搜索
          <input v-model="search" placeholder="因子名" />
        </label>
        <label class="field">
          覆盖率过滤
          <input v-model.number="coverageMin" type="number" min="0" max="1" step="0.05" />
        </label>
      </div>
    </div>
    <div class="kpis">
      <div class="card"><div class="label">显示</div><div class="value">{{ shown.length }}</div></div>
      <div class="card"><div class="label">全部</div><div class="value">{{ items.length }}</div></div>
    </div>
    <p v-if="error" class="status error">{{ error }}</p>
    <div class="overview">
      <ChartPanel title="覆盖率 vs RankIC" :traces="scatter" />
      <ChartPanel title="RankIC IR Top15" :traces="topIr" />
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th v-for="col in previewCols" :key="col" @click="toggleSort(col)">
              {{ colLabel(col) }}
              <span v-if="sortKey === col">{{ sortDir > 0 ? "↑" : "↓" }}</span>
            </th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in shown" :key="row.factor_name" @click="openDetail(row)">
            <td v-for="col in previewCols" :key="col">
              {{ col === "factor_name" ? row[col] : fmt(row[col]) }}
            </td>
            <td>
              <RouterLink class="eval" :to="evalLink(row)" @click.stop>重算</RouterLink>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.page { padding: 20px 24px 40px; }
.hero { display: flex; justify-content: space-between; gap: 16px; align-items: start; }
h1 { margin: 0 0 6px; font-size: 22px; }
.toolbar { display: flex; gap: 12px; }
.kpis { display: flex; gap: 12px; margin: 16px 0; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; min-width: 120px; }
.label { color: var(--muted); font-size: 12px; }
.value { font-size: 22px; font-weight: 600; }
.overview { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
.table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
th { cursor: pointer; user-select: none; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: #243044; }
.eval { color: var(--accent); text-decoration: none; font-size: 12px; }
@media (max-width: 1100px) {
  .overview { grid-template-columns: 1fr; }
}
</style>
