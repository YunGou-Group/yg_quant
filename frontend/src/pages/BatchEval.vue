<script setup>
import { computed, onMounted, ref, watch } from "vue";
import MetricDocs from "../components/MetricDocs.vue";
import {
  deleteDataset,
  fetchDatasetCompare,
  fetchMeta,
  patchDataset,
  postBatchEval,
} from "../api.js";
import { familyOf } from "../columns.js";
import { useRunWorkspace } from "../runWorkspace.js";

const { datasets, runId, runLabel, setRun, loadDatasets } = useRunWorkspace();

const start = ref("2010-01-01");
const end = ref("");
const universe = ref("all");
const universeChoices = ref(["all"]);
const horizon = ref(5);
const batchSize = ref(32);
const nWorkers = ref(0);
const label = ref("");
const status = ref("可勾选指标和家族以缩小范围；不选家族即全部非 style_* 因子。");
const error = ref(false);
const computing = ref(false);
const progress = ref(null);
const leftRun = ref("");
const rightRun = ref("");
const comparing = ref(false);
const compareError = ref("");
const compareResult = ref(null);
const schemaMetrics = ref([]);
const selectedMetrics = ref([]);
const factorNames = ref([]);
const selectedFamilies = ref([]);
const renaming = ref({});

const families = computed(() => {
  const buckets = new Map();
  for (const name of factorNames.value) {
    const key = familyOf(name);
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(name);
  }
  return [...buckets.entries()]
    .map(([family, names]) => ({ family, n: names.length, names }))
    .sort((a, b) => b.n - a.n);
});

const chosenFactors = computed(() => {
  if (!selectedFamilies.value.length) return factorNames.value;
  const allow = new Set(selectedFamilies.value);
  return factorNames.value.filter((name) => allow.has(familyOf(name)));
});

const estimate = computed(() => {
  const n = chosenFactors.value.length;
  const m = selectedMetrics.value.length;
  const fam = selectedFamilies.value.length
    ? `家族 ${selectedFamilies.value.join(", ")}`
    : "全部非 style 因子";
  return `将评估 ${n} 个因子、${m} 个指标（${fam}）。日度耗时大致随因子数线性增加。`;
});

function toggleFamily(name) {
  const cur = new Set(selectedFamilies.value);
  if (cur.has(name)) cur.delete(name);
  else cur.add(name);
  selectedFamilies.value = [...cur];
}

async function loadMeta() {
  try {
    const data = await fetchMeta();
    const metrics = (data.schema?.metrics || []).filter(
      (item) => (item.eval_scope || "single") === "single"
    );
    schemaMetrics.value = metrics;
    selectedMetrics.value = metrics.map((item) => item.name);
    factorNames.value = (data.factors || []).filter((name) => name && !String(name).startsWith("style_"));
    const spec = (data.schema?.shared || []).find((item) => item.name === "universe");
    if (spec?.choices?.length) universeChoices.value = spec.choices;
    if (!start.value && data.calendar?.start) start.value = data.calendar.start;
  } catch {
    schemaMetrics.value = [];
  }
}

async function run() {
  error.value = false;
  computing.value = true;
  status.value = "已提交全库任务…";
  const allMetrics = schemaMetrics.value.map((item) => item.name);
  const metrics =
    selectedMetrics.value.length && selectedMetrics.value.length < allMetrics.length
      ? selectedMetrics.value
      : null;
  const factors =
    selectedFamilies.value.length && chosenFactors.value.length < factorNames.value.length
      ? chosenFactors.value
      : null;
  try {
    const result = await postBatchEval(
      {
        start: start.value || null,
        end: end.value || null,
        universe: universe.value,
        horizon: horizon.value,
        batch_size: batchSize.value,
        n_workers: nWorkers.value,
        label: label.value || null,
        metrics,
        factors,
      },
      {
        onTick: (seconds, jobStatus, info) => {
          progress.value = info;
          const pct = info?.pct != null ? `${Math.round(info.pct * 100)}%` : jobStatus;
          status.value = `${pct} ${info?.message || jobStatus}  ${seconds}s`;
        },
      }
    );
    const newId = result?.run_id || result?.meta?.run_id;
    status.value = `完成 ${newId}  用时 ${(result?.meta?.elapsed_sec / 60 || 0).toFixed(1)} min`;
    await loadDatasets();
    if (newId) setRun(newId);
  } catch (err) {
    error.value = true;
    status.value = String(err.message || err);
  } finally {
    computing.value = false;
  }
}

async function compare() {
  compareError.value = "";
  compareResult.value = null;
  if (!leftRun.value || !rightRun.value) {
    compareError.value = "请选择两次 run";
    return;
  }
  comparing.value = true;
  try {
    compareResult.value = await fetchDatasetCompare(leftRun.value, rightRun.value);
  } catch (err) {
    compareError.value = String(err.message || err);
  } finally {
    comparing.value = false;
  }
}

function pickForCompare(id, side) {
  if (side === "left") leftRun.value = id;
  else rightRun.value = id;
}

async function saveLabel(item) {
  const text = renaming.value[item.run_id];
  if (text == null) return;
  try {
    await patchDataset(item.run_id, { label: text });
    await loadDatasets();
  } catch (err) {
    compareError.value = String(err.message || err);
  }
}

async function removeRun(item) {
  if (!window.confirm(`删除数据集 ${runLabel(item)}？此操作不可恢复。`)) return;
  try {
    await deleteDataset(item.run_id);
    await loadDatasets();
  } catch (err) {
    error.value = true;
    status.value = String(err.message || err);
  }
}

function fmtDelta(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(4)}`;
}

watch(datasets, (items) => {
  if (!leftRun.value && items[0]) leftRun.value = items[0].run_id;
  if (!rightRun.value && (items[1] || items[0])) rightRun.value = (items[1] || items[0]).run_id;
  for (const item of items) {
    if (renaming.value[item.run_id] == null) renaming.value[item.run_id] = item.label || "";
  }
});

watch(runId, (id) => {
  if (id) leftRun.value = id;
});

onMounted(loadMeta);
</script>

<template>
  <div class="page">
    <h1>全库评估</h1>
    <p class="meta">{{ estimate }}</p>
    <div class="toolbar">
      <label class="field">名称<input v-model="label" placeholder="可选，如 2010-2024 hs300 h5" /></label>
      <label class="field">开始<input v-model="start" type="date" /></label>
      <label class="field">结束<input v-model="end" type="date" /></label>
      <label class="field">股票池
        <select v-model="universe">
          <option v-for="name in universeChoices" :key="name" :value="name">{{ name }}</option>
        </select>
      </label>
      <label class="field">horizon<input v-model.number="horizon" type="number" min="1" max="60" /></label>
      <label class="field">每批因子数<input v-model.number="batchSize" type="number" min="1" /></label>
      <label class="field">进程数<input v-model.number="nWorkers" type="number" min="0" /></label>
      <button class="primary" :disabled="computing || !selectedMetrics.length" @click="run">
        {{ computing ? "计算中…" : "开始全库评估" }}
      </button>
    </div>
    <h3>指标</h3>
    <MetricDocs :metrics="schemaMetrics" v-model:selected="selectedMetrics" />
    <h3>因子家族</h3>
    <p class="meta">不选表示全部。点芯片纳入该家族。</p>
    <div class="chips">
      <button
        v-for="item in families"
        :key="item.family"
        type="button"
        :class="{ active: selectedFamilies.includes(item.family) }"
        @click="toggleFamily(item.family)"
      >
        {{ item.family }} {{ item.n }}
      </button>
    </div>
    <div class="status" :class="{ error }">{{ status }}</div>
    <div v-if="progress" class="bar">
      <div class="fill" :style="{ width: `${Math.round((progress.pct || 0) * 100)}%` }"></div>
    </div>

    <h2>已有数据集</h2>
    <ul class="runs">
      <li v-for="item in datasets" :key="item.run_id">
        <span class="grow">{{ runLabel(item) }}　{{ item.run_id }}</span>
        <input
          :value="renaming[item.run_id]"
          placeholder="名称"
          @input="renaming[item.run_id] = $event.target.value"
          @blur="saveLabel(item)"
        />
        <button type="button" @click="pickForCompare(item.run_id, 'left')">左</button>
        <button type="button" @click="pickForCompare(item.run_id, 'right')">右</button>
        <button type="button" @click="setRun(item.run_id)">用这个</button>
        <button type="button" class="danger" @click="removeRun(item)">删除</button>
      </li>
    </ul>

    <h2>对比两次 run</h2>
    <p class="meta">看区间、股票池、指标集合、daily 列和共同因子的 RankIC 差。</p>
    <div class="toolbar">
      <label class="field">左侧
        <select v-model="leftRun">
          <option v-for="item in datasets" :key="item.run_id" :value="item.run_id">
            {{ runLabel(item) }}
          </option>
        </select>
      </label>
      <label class="field">右侧
        <select v-model="rightRun">
          <option v-for="item in datasets" :key="`r-${item.run_id}`" :value="item.run_id">
            {{ runLabel(item) }}
          </option>
        </select>
      </label>
      <button class="primary" :disabled="comparing || !datasets.length" @click="compare">
        {{ comparing ? "对比中…" : "对比" }}
      </button>
    </div>
    <p v-if="compareError" class="status error">{{ compareError }}</p>
    <div v-if="compareResult" class="compare">
      <table>
        <thead>
          <tr>
            <th>字段</th>
            <th>左 {{ compareResult.left?.display_label || compareResult.left?.run_id }}</th>
            <th>右 {{ compareResult.right?.display_label || compareResult.right?.run_id }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in compareResult.meta_diff" :key="row.field">
            <td>{{ row.field }}</td>
            <td>{{ row.left ?? "—" }}</td>
            <td>{{ row.right ?? "—" }}</td>
          </tr>
          <tr v-if="!compareResult.meta_diff?.length">
            <td colspan="3" class="meta">区间 / 股票池 / 规模一致</td>
          </tr>
        </tbody>
      </table>
      <div class="grid3">
        <section>
          <h3>指标</h3>
          <p>共有 {{ compareResult.metrics?.both || 0 }}</p>
          <p v-if="compareResult.metrics?.only_left_n">只在左 {{ compareResult.metrics.only_left_n }}：{{ (compareResult.metrics.only_left || []).join(", ") }}</p>
          <p v-if="compareResult.metrics?.only_right_n">只在右 {{ compareResult.metrics.only_right_n }}：{{ (compareResult.metrics.only_right || []).join(", ") }}</p>
        </section>
        <section>
          <h3>daily 列</h3>
          <p>共有 {{ compareResult.daily_files?.both || 0 }}</p>
          <p v-if="compareResult.daily_files?.only_left_n">只在左 {{ compareResult.daily_files.only_left_n }}：{{ (compareResult.daily_files.only_left || []).join(", ") }}</p>
          <p v-if="compareResult.daily_files?.only_right_n">只在右 {{ compareResult.daily_files.only_right_n }}：{{ (compareResult.daily_files.only_right || []).join(", ") }}</p>
        </section>
        <section>
          <h3>因子</h3>
          <p>共有 {{ compareResult.factors?.both || 0 }}</p>
          <p v-if="compareResult.factors?.only_left_n">只在左 {{ compareResult.factors.only_left_n }}：{{ (compareResult.factors.only_left || []).join(", ") }}</p>
          <p v-if="compareResult.factors?.only_right_n">只在右 {{ compareResult.factors.only_right_n }}：{{ (compareResult.factors.only_right || []).join(", ") }}</p>
        </section>
      </div>
      <section v-if="compareResult.rank_ic">
        <h3>共同因子 RankIC 均值差（右 − 左）</h3>
        <p>n={{ compareResult.rank_ic.n_both }} 平均差 {{ fmtDelta(compareResult.rank_ic.mean_delta) }}</p>
        <div class="grid2">
          <div>
            <h4>右侧更好</h4>
            <ul>
              <li v-for="item in compareResult.rank_ic.top_improved" :key="item.factor">
                {{ item.factor }}　{{ fmtDelta(item.delta) }}
              </li>
            </ul>
          </div>
          <div>
            <h4>右侧更差</h4>
            <ul>
              <li v-for="item in compareResult.rank_ic.top_worsened" :key="item.factor">
                {{ item.factor }}　{{ fmtDelta(item.delta) }}
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.page { padding: 20px 24px; max-width: 1100px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin: 16px 0; }
.bar { height: 8px; background: #243044; border-radius: 99px; overflow: hidden; margin: 12px 0; }
.fill { height: 100%; background: var(--accent); }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 16px; }
.runs { padding-left: 0; list-style: none; color: var(--muted); }
.runs li { display: flex; gap: 8px; align-items: center; margin: 6px 0; flex-wrap: wrap; }
.runs .grow { flex: 1; min-width: 180px; }
.runs button, .chips button { padding: 2px 8px; font-size: 12px; }
.danger { color: var(--bad); }
.compare table { width: 100%; border-collapse: collapse; margin: 12px 0; }
.compare th, .compare td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
.grid3, .grid2 { display: grid; gap: 12px; }
.grid3 { grid-template-columns: 1fr 1fr 1fr; }
.grid2 { grid-template-columns: 1fr 1fr; }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
h3, h4 { margin: 0 0 8px; }
@media (max-width: 900px) {
  .grid3, .grid2 { grid-template-columns: 1fr; }
}
</style>
