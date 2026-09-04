<script setup>
import { computed, onMounted, reactive, ref, watch } from "vue";
import { useRoute } from "vue-router";
import FactorSidebar from "../components/FactorSidebar.vue";
import ParamControl from "../components/ParamControl.vue";
import MetricDocs from "../components/MetricDocs.vue";
import EvalCharts from "../components/EvalCharts.vue";
import ResultTable from "../components/ResultTable.vue";
import { fetchMetaRetrying, fetchResults, postEval } from "../api.js";

const route = useRoute();
const meta = ref(null);
const factor = ref(null);
const search = ref("");
const tab = ref("eval");
const start = ref("");
const end = ref("");
const selectedMetrics = ref([]);
const sharedValues = reactive({});
const metricValues = reactive({});
const status = ref("选好因子和日期等参数后，点击「开始计算」。");
const statusError = ref(false);
const evalResult = ref(null);
const results = ref([]);
const sortKey = ref("icir");
const sortDir = ref(-1);
const computing = ref(false);
let evalToken = 0;

const contractText = computed(() => {
  const contract = meta.value?.contract || {};
  return [contract.asof, contract.label_template].filter(Boolean).join(" / ");
});

const factorMetrics = computed(() =>
  (meta.value?.schema?.metrics || []).filter(
    (item) => (item.eval_scope || "single") === "single"
  )
);

const factorSchema = computed(() => {
  const schema = meta.value?.schema;
  if (!schema) return null;
  return { ...schema, metrics: factorMetrics.value };
});

const metricParamSpecs = computed(() => {
  const catalog = factorMetrics.value;
  const selected = metricsWithDeps(selectedMetrics.value, catalog);
  const seen = new Set();
  const specs = [];
  for (const metric of catalog) {
    if (!selected.has(metric.name)) continue;
    for (const spec of metric.params || []) {
      if (seen.has(spec.name)) continue;
      seen.add(spec.name);
      specs.push(spec);
    }
  }
  return specs;
});

function metricsWithDeps(selected, catalog) {
  const byName = new Map(catalog.map((metric) => [metric.name, metric]));
  const producers = {};
  for (const metric of catalog) {
    for (const key of metric.produces || []) producers[key] = metric.name;
  }
  const wanted = new Set(selected);
  const stack = [...selected];
  while (stack.length) {
    const name = stack.pop();
    const metric = byName.get(name);
    if (!metric) continue;
    for (const key of metric.requires || []) {
      const producer = producers[key];
      if (producer && !wanted.has(producer)) {
        wanted.add(producer);
        stack.push(producer);
      }
    }
  }
  return wanted;
}

function setParamDefaults(specs, store) {
  for (const spec of specs) {
    if (store[spec.name] === undefined || store[spec.name] === null) {
      store[spec.name] = spec.default;
    }
  }
}

function collectParams() {
  const params = { ...sharedValues };
  for (const spec of metricParamSpecs.value) {
    const value = metricValues[spec.name];
    if (value !== null && value !== undefined && value !== "") params[spec.name] = value;
  }
  return params;
}

async function runEval() {
  if (!factor.value) {
    statusError.value = true;
    status.value = "请先在左侧点选一个因子，再点「开始计算」。";
    return;
  }
  const token = ++evalToken;
  statusError.value = false;
  computing.value = true;
  status.value = `已提交评估 ${factor.value}，等待后端…`;
  const params = collectParams();
  const body = {
    factor: factor.value,
    start: start.value || null,
    end: end.value || null,
    horizon: params.horizon,
    universe: params.universe,
    metrics: selectedMetrics.value.filter((name) =>
      factorMetrics.value.some((item) => item.name === name)
    ),
    params,
  };
  try {
    const data = await postEval(body, {
      shouldAbort: () => token !== evalToken,
      onTick: (seconds, jobStatus, detail) => {
        if (token !== evalToken) return;
        statusError.value = false;
        const phase =
          jobStatus === "queued"
            ? "排队/预热中"
            : jobStatus === "retry"
              ? "等待后端响应"
              : "正在计算";
        const extra = jobStatus === "retry" && detail ? `（${detail}，继续等待）` : "（请等待，不要重复点击）";
        status.value = `${phase} ${factor.value} … ${seconds}s${extra}`;
      },
    });
    if (token !== evalToken) return;
    if (!data) throw new Error("评估完成但没有返回结果");
    evalResult.value = data;
    status.value = `${data.factor}  ${data.label}  asof=${data.asof}  ${data.start || ""} → ${data.end || ""}`;
  } catch (err) {
    if (err && err.name === "AbortError") return;
    if (token !== evalToken) return;
    statusError.value = true;
    status.value = String(err.message || err);
  } finally {
    if (token === evalToken) computing.value = false;
  }
}

function applySnapshot(snap) {
  if (!snap) return;
  factor.value = snap.factor;
  const params = snap.params || {};
  if (params.horizon != null) sharedValues.horizon = params.horizon;
  if (params.universe) sharedValues.universe = params.universe;
  if (snap.start) start.value = snap.start;
  if (snap.end) end.value = snap.end;
  if (params.metrics?.length) {
    const allowed = new Set(factorMetrics.value.map((item) => item.name));
    selectedMetrics.value = params.metrics.filter((name) => allowed.has(name));
  }
  for (const [key, value] of Object.entries(params)) {
    if (["horizon", "universe", "metrics"].includes(key)) continue;
    metricValues[key] = value;
  }
  tab.value = "eval";
  statusError.value = false;
  status.value = `已载入 ${snap.factor} 的参数，确认后点击「开始计算」。`;
}

async function loadResults() {
  const data = await fetchResults();
  results.value = data.items || [];
}

function onTab(name) {
  tab.value = name;
  if (name === "results") loadResults();
}

watch(selectedMetrics, () => {
  setParamDefaults(metricParamSpecs.value, metricValues);
});
watch(factor, (name) => {
  if (!name) return;
  if (computing.value) {
    evalToken += 1;
    computing.value = false;
  }
  statusError.value = false;
  status.value = `已选 ${name}，调好日期等参数后点击「开始计算」。`;
});

function applyQuery(query) {
  if (!query) return;
  if (query.factor) factor.value = String(query.factor);
  if (query.start) start.value = String(query.start);
  if (query.end) end.value = String(query.end);
  if (query.horizon != null && query.horizon !== "") {
    const n = Number(query.horizon);
    if (Number.isFinite(n)) sharedValues.horizon = n;
  }
  if (query.universe) sharedValues.universe = String(query.universe);
}

watch(
  () => route.query,
  (query) => applyQuery(query)
);

onMounted(async () => {
  try {
    const data = await fetchMetaRetrying((tried, total) => {
      statusError.value = false;
      status.value = `等待后端 http://127.0.0.1:8765 （${tried}/${total}）`;
    });
    meta.value = data;
    setParamDefaults(data.schema?.shared || [], sharedValues);
    start.value = data.calendar?.start || "";
    end.value = data.calendar?.end || "";
    selectedMetrics.value = [...(data.defaults?.metrics || [])];
    setParamDefaults(metricParamSpecs.value, metricValues);
    applyQuery(route.query);
    status.value = factor.value
      ? `已选 ${factor.value}，确认参数后点击「开始计算」。`
      : "选好因子和日期等参数后，点击「开始计算」。";
  } catch (err) {
    statusError.value = true;
    status.value = "加载 meta 失败，请先运行 python -m App。 " + err;
  }
});
</script>

<template>
  <div class="app">
    <header>
      <div class="tabs">
        <button :class="{ active: tab === 'eval' }" @click="onTab('eval')">分析</button>
        <button :class="{ active: tab === 'results' }" @click="onTab('results')">结果</button>
      </div>
      <span class="meta">{{ contractText }}</span>
      <span class="spacer"></span>
      <button class="primary" :disabled="!factor || computing" @click="runEval">
        {{ computing ? "计算中…" : "开始计算" }}
      </button>
    </header>
    <FactorSidebar
      v-model="factor"
      v-model:search="search"
      :factors="meta?.factors || []"
    />
    <main>
      <section v-show="tab === 'eval'">
        <div class="toolbar">
          <ParamControl
            v-for="spec in meta?.schema?.shared || []"
            :key="spec.name"
            :spec="spec"
            v-model="sharedValues[spec.name]"
          />
          <div class="field">
            <label>开始日期</label>
            <input v-model="start" type="date" />
          </div>
          <div class="field">
            <label>结束日期</label>
            <input v-model="end" type="date" />
          </div>
        </div>
        <MetricDocs
          :metrics="factorMetrics"
          v-model:selected="selectedMetrics"
        />
        <div class="toolbar">
          <ParamControl
            v-for="spec in metricParamSpecs"
            :key="spec.name"
            :spec="spec"
            v-model="metricValues[spec.name]"
          />
        </div>
        <div class="status" :class="{ error: statusError }">{{ status }}</div>
        <EvalCharts :data="evalResult" :schema="factorSchema" />
      </section>
      <section v-show="tab === 'results'">
        <ResultTable
          :items="results"
          v-model:sort-key="sortKey"
          v-model:sort-dir="sortDir"
          @select="applySnapshot"
        />
      </section>
    </main>
  </div>
</template>

<style scoped>
.app {
  display: grid;
  grid-template-columns: 240px 1fr;
  grid-template-rows: auto 1fr;
  height: calc(100vh - 64px);
}

header {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 16px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}

.tabs {
  display: flex;
  gap: 4px;
}

.spacer {
  flex: 1;
}

main {
  overflow: auto;
  padding: 12px 16px 24px;
}

.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: end;
  margin-bottom: 12px;
}
</style>
