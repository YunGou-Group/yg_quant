<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { fetchFactorTable, previewSelect } from "../api.js";
import { FILTER_COLUMNS, colLabel } from "../columns.js";
import { fmt } from "../format.js";
import { useRunWorkspace } from "../runWorkspace.js";

const router = useRouter();
const { runId, runLink } = useRunWorkspace();
const columns = ref([]);
const metric = ref("rank_ic_ir");
const op = ref(">=");
const value = ref(0.5);
const rankBy = ref("rank_ic_ir");
const rules = ref([]);
const result = ref(null);
const error = ref("");

const filterCols = computed(() => {
  const have = new Set(columns.value);
  const preferred = FILTER_COLUMNS.filter((col) => have.has(col));
  return preferred.length ? preferred : columns.value.slice(0, 8);
});

const resultCols = computed(() => {
  const preferred = [
    "factor_name",
    rankBy.value,
    "rank_ic_mean",
    "rank_ic_ir",
    "coverage_rate_mean",
    "quantile_spread_mean",
  ];
  const sample = result.value?.items?.[0];
  if (!sample) return preferred.slice(0, 2);
  const have = new Set(Object.keys(sample));
  return [...new Set(preferred.filter((col) => have.has(col)))];
});

async function loadColumns() {
  error.value = "";
  result.value = null;
  if (!runId.value) {
    columns.value = [];
    return;
  }
  try {
    const data = await fetchFactorTable(runId.value);
    columns.value = (data.columns || []).filter((c) => c !== "factor_name");
    const cols = filterCols.value;
    if (!cols.includes(metric.value) && cols[0]) metric.value = cols[0];
    if (!cols.includes(rankBy.value) && cols[0]) rankBy.value = cols[0];
  } catch (err) {
    error.value = String(err.message || err);
  }
}

function addRule() {
  rules.value.push({ metric: metric.value, op: op.value, value: Number(value.value) });
}

async function run() {
  error.value = "";
  try {
    result.value = await previewSelect({
      run_id: runId.value || undefined,
      rules: rules.value,
      rank_by: rankBy.value,
      ascending: false,
      limit: 80,
    });
  } catch (err) {
    error.value = String(err.message || err);
  }
}

function openDetail(row) {
  router.push(runLink(`/detail/${encodeURIComponent(row.factor_name)}`));
}

onMounted(loadColumns);
watch(runId, loadColumns);
</script>

<template>
  <div class="page">
    <h1>因子筛选</h1>
    <p class="meta">规则作用在当前 run 的 summary 上。点行进入深度报告。</p>
    <div class="toolbar">
      <select v-model="metric">
        <option v-for="col in filterCols" :key="col" :value="col">{{ colLabel(col) }}</option>
      </select>
      <select v-model="op">
        <option>&gt;=</option>
        <option>&gt;</option>
        <option>&lt;=</option>
        <option>&lt;</option>
      </select>
      <input v-model.number="value" type="number" step="0.01" />
      <button @click="addRule">添加规则</button>
      <label class="field">排序
        <select v-model="rankBy">
          <option v-for="col in filterCols" :key="col" :value="col">{{ colLabel(col) }}</option>
        </select>
      </label>
      <button class="primary" @click="run">预览</button>
    </div>
    <div class="rules">
      <span v-for="(rule, i) in rules" :key="i" class="pill">
        {{ colLabel(rule.metric) }} {{ rule.op }} {{ rule.value }}
        <button @click="rules.splice(i, 1)">×</button>
      </span>
    </div>
    <p v-if="error" class="status error">{{ error }}</p>
    <p v-if="result" class="meta">通过 {{ result.n_passed }} / {{ result.n_total }}</p>
    <table v-if="result">
      <thead>
        <tr>
          <th v-for="col in resultCols" :key="col">{{ colLabel(col) }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in result.items" :key="row.factor_name" @click="openDetail(row)">
          <td v-for="col in resultCols" :key="col">{{ col === "factor_name" ? row[col] : fmt(row[col]) }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.page { padding: 20px 24px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: end; }
.rules { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
.pill { background: var(--panel); border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 8px; border-bottom: 1px solid var(--line); text-align: left; }
tbody tr { cursor: pointer; }
</style>
