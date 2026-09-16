<script setup>
import { computed, ref, watch } from "vue";
import { RouterLink } from "vue-router";
import ChartPanel from "./charts/ChartPanel.vue";
import { fetchLibraryCorr } from "../api.js";
import { fmt } from "../format.js";
import { useRunWorkspace } from "../runWorkspace.js";

const props = defineProps({
  payload: { type: Object, default: null },
  suggestedNames: { type: Array, default: () => [] },
});

const { runId, runLink } = useRunWorkspace();
const tab = ref("factor_corr");
const factorText = ref("");
const subset = ref(null);
const queryError = ref("");
const querying = ref(false);

const tabs = [
  { id: "factor_corr", label: "截面相关" },
  { id: "ic_corr", label: "IC 序列相关" },
];

const loading = computed(() => props.payload == null);
const items = computed(() => props.payload?.items || {});
const parsedNames = computed(() => parseFactorNames(factorText.value));

const missing = computed(() => {
  if (loading.value) return "";
  const ran = props.payload?.library_metrics || [];
  if (items.value.factor_corr || items.value.ic_corr || items.value.family_redundancy) return "";
  if (ran.includes("factor_corr") || ran.includes("ic_corr") || ran.includes("family_redundancy")) {
    return "库级文件缺失，请检查该 run 的 library 目录。";
  }
  return "这次 run 没有算因子相关。全库评估请保持指标全选（或 CLI 不传 --metrics）。";
});

const activeItem = computed(() => items.value[tab.value] || null);

const activeSubset = computed(() => {
  if (subset.value?.metric === tab.value) return subset.value;
  return null;
});

const matrix = computed(() => {
  if (activeSubset.value?.matrix) return activeSubset.value.matrix;
  const raw = activeItem.value?.matrix;
  if (!raw?.index?.length || !raw?.data) return null;
  return raw;
});

const heatmapHint = computed(() => {
  if (missing.value || loading.value) return "";
  if (queryError.value) return "";
  if (activeSubset.value && !activeSubset.value.matrix) {
    return "命中不足两个因子，无法画子矩阵。";
  }
  if (!activeItem.value || !activeItem.value.matrix) return "这次 run 没有这项矩阵。";
  if (!matrix.value) return "没有相关矩阵。";
  return "";
});

const usingSubset = computed(() => Boolean(activeSubset.value?.matrix));
const showTicks = computed(() => usingSubset.value || (matrix.value?.index.length || 0) <= 28);
const heatHeight = computed(() => {
  const n = matrix.value?.index.length || 0;
  if (usingSubset.value) return Math.max(320, Math.min(560, 88 + n * 36));
  return 520;
});

const heatmap = computed(() => {
  const mat = matrix.value;
  if (!mat) return [];
  return [
    {
      type: "heatmap",
      z: mat.data,
      x: mat.columns,
      y: mat.index,
      colorscale: "RdBu",
      zmid: 0,
      zmin: -1,
      zmax: 1,
      hoverongaps: false,
      colorbar: { thickness: 12, len: 0.82, tickfont: { size: 10 } },
      hovertemplate: "%{y} × %{x}: %{z:.2f}<extra></extra>",
    },
  ];
});

const heatLayout = computed(() => {
  const show = showTicks.value;
  const n = matrix.value?.index.length || 0;
  const tickSize = n > 18 ? 8 : 10;
  return {
    showlegend: false,
    margin: { t: 40, r: 48, b: show ? 110 : 32, l: show ? 96 : 32 },
    xaxis: {
      gridcolor: "#2c3a4f",
      tickangle: -60,
      showticklabels: show,
      tickfont: { size: tickSize },
    },
    yaxis: {
      gridcolor: "#2c3a4f",
      autorange: "reversed",
      showticklabels: show,
      tickfont: { size: tickSize },
    },
  };
});

const pairs = computed(() => {
  const mat = matrix.value;
  if (!mat) return [];
  const rows = [];
  for (let i = 0; i < mat.index.length; i += 1) {
    for (let j = i + 1; j < mat.index.length; j += 1) {
      const z = mat.data[i][j];
      if (z == null || Number.isNaN(Number(z))) continue;
      const corr = Number(z);
      rows.push({ a: mat.index[i], b: mat.columns[j], corr, abs: Math.abs(corr) });
    }
  }
  rows.sort((x, y) => y.abs - x.abs);
  return rows.slice(0, usingSubset.value ? 40 : 15);
});

const families = computed(() => {
  const rows = items.value.family_redundancy?.families || [];
  return [...rows].sort((a, b) => Number(b.mean_abs || 0) - Number(a.mean_abs || 0));
});

const familyBars = computed(() => {
  const rows = families.value.filter((row) => row.mean_abs != null && Number.isFinite(Number(row.mean_abs)));
  if (!rows.length) return [];
  return [
    {
      type: "bar",
      x: rows.map((row) => row.family),
      y: rows.map((row) => row.mean_abs),
      name: "家族内 |相关|",
    },
  ];
});

const scalars = computed(() => activeSubset.value?.scalars || activeItem.value?.scalars || {});
const showNumberTable = computed(() => usingSubset.value && (matrix.value?.index.length || 0) <= 24);

function parseFactorNames(text) {
  const seen = new Set();
  const names = [];
  for (const token of String(text || "").split(/[\s,;，、；|]+/)) {
    const name = token.trim();
    if (!name || seen.has(name)) continue;
    seen.add(name);
    names.push(name);
  }
  return names;
}

function sliceLikeBackend(matrix, names) {
  if (!matrix?.index?.length || !matrix?.data) {
    return { names: [], missing: names, matrix: null, scalars: {} };
  }
  const pos = new Map(matrix.index.map((name, i) => [name, i]));
  const lower = new Map(matrix.index.map((name) => [String(name).toLowerCase(), name]));
  const found = [];
  const idxs = [];
  const missing = [];
  const seen = new Set();
  names.forEach((raw) => {
    const name = String(raw || "").trim();
    const canon = pos.has(name) ? name : lower.get(name.toLowerCase());
    if (!canon) {
      missing.push(name);
      return;
    }
    if (seen.has(canon)) return;
    seen.add(canon);
    found.push(canon);
    idxs.push(pos.get(canon));
  });
  const data = idxs.map((i) => idxs.map((j) => matrix.data[i]?.[j] ?? null));
  const vals = [];
  for (let i = 0; i < data.length; i += 1) {
    for (let j = i + 1; j < data.length; j += 1) {
      const z = Number(data[i][j]);
      if (Number.isFinite(z)) vals.push(Math.abs(z));
    }
  }
  return {
    names: found,
    missing,
    matrix: found.length ? { index: found, columns: idxs.map((i) => (matrix.columns || matrix.index)[i]), data } : null,
    scalars: {
      mean_abs: vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null,
      max_abs: vals.length ? Math.max(...vals) : null,
      n_factors: found.length,
    },
  };
}

async function querySubset() {
  queryError.value = "";
  const names = parsedNames.value;
  if (names.length < 2) {
    queryError.value = "请至少输入两个因子名。";
    return;
  }
  if (names.length > 80) {
    queryError.value = "最多 80 个因子，请缩短列表。";
    return;
  }
  querying.value = true;
  try {
    subset.value = await fetchLibraryCorr(runId.value, names, tab.value);
  } catch (err) {
    const local = sliceLikeBackend(activeItem.value?.matrix, names);
    local.metric = tab.value;
    if (!activeItem.value?.matrix) {
      queryError.value = String(err.message || err);
      subset.value = null;
    } else {
      subset.value = local;
    }
  } finally {
    querying.value = false;
  }
}

function clearSubset() {
  subset.value = null;
  queryError.value = "";
}

function fillSuggested() {
  const names = (props.suggestedNames || []).filter(Boolean);
  if (!names.length) return;
  factorText.value = names.join("\n");
}

function detailLink(name) {
  return runLink(`/detail/${encodeURIComponent(name)}`);
}

function cellTone(value) {
  if (value == null || Number.isNaN(Number(value))) return {};
  const z = Number(value);
  const a = Math.min(1, Math.abs(z));
  const bg = z >= 0 ? `rgba(61, 156, 240, ${0.12 + 0.55 * a})` : `rgba(240, 113, 120, ${0.12 + 0.55 * a})`;
  return { background: bg };
}

watch(
  () => [items.value.factor_corr, items.value.ic_corr],
  () => {
    if (!items.value[tab.value] && items.value.ic_corr && tab.value === "factor_corr") {
      tab.value = "ic_corr";
    }
  }
);

watch(tab, () => {
  if (parsedNames.value.length >= 2 && subset.value) querySubset();
});

watch(
  () => props.payload?.run_id,
  () => {
    subset.value = null;
    queryError.value = "";
  }
);
</script>

<template>
  <section class="library">
    <div class="head">
      <h2>因子相关</h2>
      <div class="tabs">
        <button
          v-for="item in tabs"
          :key="item.id"
          type="button"
          :class="{ active: tab === item.id }"
          @click="tab = item.id"
        >
          {{ item.label }}
        </button>
      </div>
    </div>
    <p v-if="loading" class="status">正在读取库级相关…</p>
    <p v-else-if="missing" class="status">{{ missing }}</p>
    <template v-else>
      <div class="query">
        <label class="field grow">
          自选因子
          <textarea
            v-model="factorText"
            rows="4"
            placeholder="输入因子名，逗号 / 空格 / 换行均可，例如：&#10;vol_rel_ma5, chip_low_deposit, pv_ratio"
          />
        </label>
        <div class="actions">
          <button class="primary" type="button" :disabled="querying" @click="querySubset">
            {{ querying ? "查询中…" : "查询子矩阵" }}
          </button>
          <button type="button" :disabled="!suggestedNames.length" @click="fillSuggested">
            填入当前筛选
          </button>
          <button type="button" :disabled="!subset" @click="clearSubset">显示全库</button>
          <span class="meta">已解析 {{ parsedNames.length }} 个</span>
        </div>
      </div>
      <p v-if="queryError" class="status error">{{ queryError }}</p>
      <p v-else-if="activeSubset?.missing?.length" class="status">
        未找到：{{ activeSubset.missing.join("、") }}
      </p>
      <div class="kpis">
        <div class="card">
          <div class="label">平均 |相关|</div>
          <div class="value">{{ fmt(scalars.mean_abs) }}</div>
        </div>
        <div class="card">
          <div class="label">最大 |相关|</div>
          <div class="value">{{ fmt(scalars.max_abs) }}</div>
        </div>
        <div class="card">
          <div class="label">{{ usingSubset ? "子矩阵" : "矩阵阶数" }}</div>
          <div class="value">{{ matrix ? matrix.index.length : "—" }}</div>
        </div>
      </div>
      <p class="hint">
        截面相关看选股是否重叠，IC 相关看收益是否一起动。全库热力图超过 28 个因子时隐藏刻度；子矩阵会显示刻度和数值表。
      </p>
      <p v-if="heatmapHint" class="status">{{ heatmapHint }}</p>
      <div class="grid">
        <ChartPanel
          :title="usingSubset
            ? `自选 ${matrix.index.length}×${matrix.index.length}`
            : (tab === 'factor_corr' ? '截面 Spearman' : 'RankIC 序列相关')"
          :traces="heatmap"
          :layout="heatLayout"
          :height="heatHeight"
        />
        <div class="side">
          <h3>最高 |相关| 配对</h3>
          <p v-if="!pairs.length" class="status">当前没有配对。</p>
          <table v-else>
            <thead>
              <tr>
                <th>因子 A</th>
                <th>因子 B</th>
                <th>ρ</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in pairs" :key="row.a + '|' + row.b">
                <td>
                  <RouterLink class="name" :to="detailLink(row.a)">{{ row.a }}</RouterLink>
                </td>
                <td>
                  <RouterLink class="name" :to="detailLink(row.b)">{{ row.b }}</RouterLink>
                </td>
                <td>{{ fmt(row.corr) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <div v-if="showNumberTable" class="table-wrap">
        <table class="numbers">
          <thead>
            <tr>
              <th></th>
              <th v-for="name in matrix.columns" :key="'c-' + name">{{ name }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, i) in matrix.data" :key="matrix.index[i]">
              <th>{{ matrix.index[i] }}</th>
              <td v-for="(cell, j) in row" :key="matrix.index[i] + '|' + matrix.columns[j]" :style="cellTone(cell)">
                {{ fmt(cell) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-if="families.length && !usingSubset" class="family">
        <ChartPanel title="家族内冗余" :traces="familyBars" :height="280" />
        <div class="side">
          <h3>家族</h3>
          <table>
            <thead>
              <tr>
                <th>家族</th>
                <th>n</th>
                <th>均值|ρ|</th>
                <th>最大|ρ|</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in families" :key="row.family">
                <td>{{ row.family }}</td>
                <td>{{ row.n }}</td>
                <td>{{ fmt(row.mean_abs) }}</td>
                <td>{{ fmt(row.max_abs) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>
  </section>
</template>

<style scoped>
.library { margin-bottom: 24px; }
.head { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
h2 { margin: 0 0 4px; font-size: 18px; }
h3 { margin: 0 0 10px; font-size: 14px; }
.tabs { display: flex; gap: 6px; }
.tabs button {
  background: transparent;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 4px 10px;
}
.tabs button.active { color: #081018; background: var(--accent); border-color: var(--accent); }
.query { display: flex; flex-direction: column; gap: 8px; margin: 12px 0; }
.grow { min-width: 0; }
textarea {
  width: 100%;
  resize: vertical;
  min-height: 88px;
  background: #243044;
  color: var(--text);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px 10px;
  font: inherit;
}
.actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.kpis { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }
.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 12px 16px;
  min-width: 120px;
}
.label { color: var(--muted); font-size: 12px; }
.value { font-size: 22px; font-weight: 600; }
.hint { color: var(--muted); font-size: 12px; margin: 0 0 12px; }
.grid,
.family { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.8fr); gap: 16px; margin-bottom: 16px; }
.side { border: 1px solid var(--line); border-radius: 10px; padding: 12px; background: var(--panel); overflow: auto; }
.table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 8px; margin-bottom: 16px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
.numbers th, .numbers td { text-align: right; font-variant-numeric: tabular-nums; }
.numbers th:first-child, .numbers td:first-child, .numbers tbody th { text-align: left; }
.name { color: var(--accent); text-decoration: none; }
.status { color: var(--muted); }
.status.error { color: var(--bad); }
@media (max-width: 1100px) {
  .grid,
  .family { grid-template-columns: 1fr; }
}
</style>
