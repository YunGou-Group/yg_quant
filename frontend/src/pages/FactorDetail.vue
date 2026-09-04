<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter, RouterLink } from "vue-router";
import ChartGallery from "../components/charts/ChartGallery.vue";
import { fetchDatasetCoverage, fetchFactorDetail, fetchFactorSeriesBundle, fetchFactorTable } from "../api.js";
import { fmt } from "../format.js";
import { useRunWorkspace } from "../runWorkspace.js";

const route = useRoute();
const router = useRouter();
const { runId, current, runLink } = useRunWorkspace();
const factor = computed(() => decodeURIComponent(route.params.factor || ""));
const names = ref([]);
const summary = ref({});
const error = ref("");
const seriesMap = ref({});
const loading = ref(false);
const coverage = ref(null);
const openDims = ref(new Set(["预测力"]));
const loadedKeys = ref(new Set());

const DIM_ORDER = ["预测力", "有效期", "暴露度", "风格暴露", "风格归因", "尾部", "市场"];

const cards = computed(() => {
  const s = summary.value || {};
  return [
    ["RankIC", s.rank_ic_mean],
    ["RankIC IR", s.rank_ic_ir],
    ["Pearson IC", s.ic_mean],
    ["覆盖率", s.coverage_rate_mean],
    ["分层价差", s.quantile_spread_mean],
    ["一致性", s.long_short_term_consistency_mean],
    ["加权 PnL", s.weighted_pnl_mean],
  ].filter(([, value]) => value != null && value !== "");
});

const missingCharts = computed(() =>
  (coverage.value?.charts || []).filter((item) => item.status !== "ok")
);

const dimGroups = computed(() => {
  const charts = coverage.value?.charts || [];
  const buckets = new Map();
  for (const chart of charts) {
    const dim = chart.dimension || "其他";
    if (!buckets.has(dim)) buckets.set(dim, []);
    buckets.get(dim).push(chart);
  }
  const ordered = DIM_ORDER.filter((key) => buckets.has(key)).map((key) => ({
    dimension: key,
    charts: buckets.get(key),
    ok: buckets.get(key).filter((item) => item.status === "ok").length,
  }));
  for (const [key, items] of buckets) {
    if (!DIM_ORDER.includes(key)) {
      ordered.push({
        dimension: key,
        charts: items,
        ok: items.filter((item) => item.status === "ok").length,
      });
    }
  }
  return ordered;
});

const visibleIds = computed(() => {
  const ids = new Set();
  for (const group of dimGroups.value) {
    if (!openDims.value.has(group.dimension)) continue;
    for (const chart of group.charts) {
      if (chart.status === "ok") ids.add(chart.id);
    }
  }
  return ids;
});

const evalTo = computed(() =>
  runLink("/eval", {
    factor: factor.value,
    start: current.value?.start,
    end: current.value?.end,
    horizon: current.value?.horizon,
    universe: current.value?.universe,
  })
);

function keysForDims(dims) {
  const keys = [];
  const seen = new Set();
  for (const chart of coverage.value?.charts || []) {
    if (!dims.has(chart.dimension)) continue;
    if (chart.status !== "ok") continue;
    for (const key of chart.matched_keys || []) {
      if (seen.has(key) || loadedKeys.value.has(key)) continue;
      seen.add(key);
      keys.push(key);
    }
  }
  return keys;
}

async function fetchKeys(keys) {
  if (!keys.length || !factor.value || !runId.value) return;
  const bundle = await fetchFactorSeriesBundle(factor.value, runId.value, keys);
  seriesMap.value = { ...seriesMap.value, ...(bundle.series || {}) };
  const next = new Set(loadedKeys.value);
  for (const key of keys) next.add(key);
  loadedKeys.value = next;
}

async function load() {
  error.value = "";
  seriesMap.value = {};
  coverage.value = null;
  loadedKeys.value = new Set();
  openDims.value = new Set(["预测力"]);
  loading.value = true;
  try {
    if (!runId.value) return;
    const table = await fetchFactorTable(runId.value);
    names.value = (table.items || []).map((item) => item.factor_name);
    if (!factor.value) return;
    const [detail, cov] = await Promise.all([
      fetchFactorDetail(factor.value, runId.value),
      fetchDatasetCoverage(runId.value, factor.value).catch(() => null),
    ]);
    summary.value = detail.summary || {};
    coverage.value = cov;
    const first = keysForDims(openDims.value);
    await fetchKeys(first);
  } catch (err) {
    error.value = String(err.message || err);
  } finally {
    loading.value = false;
  }
}

async function toggleDim(dimension) {
  const next = new Set(openDims.value);
  if (next.has(dimension)) next.delete(dimension);
  else next.add(dimension);
  openDims.value = next;
  if (next.has(dimension)) {
    try {
      await fetchKeys(keysForDims(new Set([dimension])));
    } catch (err) {
      error.value = String(err.message || err);
    }
  }
}

function goFactor(name) {
  if (!name) return;
  router.push(runLink(`/detail/${encodeURIComponent(name)}`));
}

onMounted(load);
watch(() => route.params.factor, load);
watch(runId, load);
</script>

<template>
  <div class="page">
    <div class="hero">
      <div>
        <h1>深度报告</h1>
        <p class="meta">按维度展开图表并按需拉序列。缺图会列出原因。</p>
      </div>
      <div class="toolbar">
        <input
          :value="factor"
          list="factor-names"
          placeholder="选择或输入因子"
          @change="goFactor($event.target.value)"
        />
        <datalist id="factor-names">
          <option v-for="name in names" :key="name" :value="name" />
        </datalist>
        <RouterLink v-if="factor" class="eval" :to="evalTo">按自定义参数重算</RouterLink>
      </div>
    </div>
    <p v-if="error" class="status error">{{ error }}</p>
    <p v-if="!factor" class="meta">先选择一个因子。</p>
    <p v-else-if="loading" class="meta">加载序列…</p>
    <template v-else>
      <div class="kpis">
        <div v-for="[label, value] in cards" :key="label" class="card">
          <div class="label">{{ label }}</div>
          <div class="value">{{ fmt(value) }}</div>
        </div>
      </div>
      <section v-if="missingCharts.length" class="missing">
        <h2>缺图 {{ missingCharts.length }}/{{ coverage?.n_charts || missingCharts.length }}</h2>
        <ul>
          <li v-for="item in missingCharts" :key="item.id">
            <strong>{{ item.title }}</strong>
           　{{ item.reason_text }}
            <span v-if="item.detail" class="meta"> — {{ item.detail }}</span>
          </li>
        </ul>
      </section>
      <div class="dims">
        <button
          v-for="group in dimGroups"
          :key="group.dimension"
          type="button"
          :class="{ active: openDims.has(group.dimension) }"
          @click="toggleDim(group.dimension)"
        >
          {{ group.dimension }} {{ group.ok }}/{{ group.charts.length }}
        </button>
      </div>
      <ChartGallery :series="seriesMap" :visible-ids="visibleIds" />
    </template>
  </div>
</template>

<style scoped>
.page { padding: 20px 24px 40px; }
.hero { display: flex; justify-content: space-between; gap: 16px; }
h1 { margin: 0 0 6px; }
.toolbar { display: flex; gap: 8px; align-items: center; }
.eval { color: var(--accent); text-decoration: none; white-space: nowrap; }
.kpis { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; min-width: 120px; }
.label { color: var(--muted); font-size: 12px; }
.value { font-size: 18px; font-weight: 600; }
.missing { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; margin: 0 0 16px; }
.missing h2 { margin: 0 0 8px; font-size: 15px; }
.missing ul { margin: 0; padding-left: 18px; }
.missing li { margin: 4px 0; }
.dims { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 12px; }
.dims button.active { background: var(--accent); color: #081018; border-color: var(--accent); }
</style>
