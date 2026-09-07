<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import ChartPanel from "../components/charts/ChartPanel.vue";
import ReturnsFlow from "../components/ReturnsFlow.vue";
import {
  fetchAttribution,
  fetchBacktestMeta,
  fetchBacktestRun,
  fetchBacktestRuns,
  postAttribution,
  postBacktest,
} from "../api.js";

const meta = ref(null);
const form = reactive({
  strategy: "small_cap",
  start: "2023-01-01",
  end: "",
  universe: "all",
  factor: "",
  n: null,
  hold: 6,
  anti_tail: false,
  allocator: "equal",
  allocator_lookback: 252,
  max_weight: 1,
  cash: 1000000,
  commission: 0.0003,
  stamp: 0.0005,
  mu_model: "geometric",
  risk_free_rate: 0.02,
  risk_aversion: 1,
  target_return: null,
  target_volatility: null,
  l2_gamma: 0.1,
  tc_rate: 0.001,
  tail_confidence: 0.95,
  benchmark: "",
  no_benchmark: false,
});
const status = ref("选好策略和区间后点击「开始回测」。全市场回测可能要几分钟。");
const statusError = ref(false);
const computing = ref(false);
const attributing = ref(false);
const result = ref(null);
const attribution = ref(null);
const attrStatus = ref("");
const attrError = ref(false);
const runs = ref([]);
let runToken = 0;
let attrToken = 0;

const strategyFields = computed(() => {
  const name = form.strategy;
  const item = (meta.value?.strategies || []).find((row) => row.name === name);
  return item?.fields || [];
});

const universes = computed(() => {
  const items = meta.value?.universes;
  if (items?.length) return items;
  return [{ name: "all", description: "" }];
});

const traces = computed(() => {
  const equity = result.value?.equity;
  if (!equity?.dates?.length) return [];
  const lines = [
    {
      x: equity.dates,
      y: equity.nav,
      type: "scatter",
      mode: "lines",
      name: "策略净值",
    },
  ];
  if (equity.benchmark?.length) {
    lines.push({
      x: equity.dates,
      y: equity.benchmark,
      type: "scatter",
      mode: "lines",
      name: "基准",
    });
  }
  return lines;
});

const kpis = computed(() => {
  const m = result.value?.metrics;
  if (!m) return [];
  return [
    { label: "累计收益", value: pct(m.total_return) },
    { label: "年化", value: pct(m.annualized) },
    { label: "最大回撤", value: pct(m.max_drawdown) },
    { label: "波动", value: pct(m.volatility) },
    { label: "夏普", value: num(m.sharpe) },
    { label: "Calmar", value: num(m.calmar) },
    { label: "日均换手", value: pct(m.turnover_mean) },
    { label: "超额年化", value: pct(m.excess_annualized) },
    { label: "交易日", value: m.n_days == null ? "—" : String(m.n_days) },
  ];
});

const attrKpis = computed(() => {
  const s = attribution.value?.summary;
  if (!s) return [];
  return [
    { label: "组合累计", value: pct(s.portfolio_return) },
    { label: "基准累计", value: pct(s.benchmark_return) },
    { label: "主动收益", value: pct(s.active_return) },
    { label: "交易贡献", value: pct(s.trade_return) },
    { label: "持仓贡献", value: pct(s.holding_return) },
    { label: "配置效应", value: pct(s.allocation_effect) },
    { label: "选择效应", value: pct(s.selection_effect) },
    { label: "费用", value: money(s.fees) },
  ];
});

const factorGroups = computed(() => attribution.value?.factors?.groups || []);
const styleFactorRows = computed(() =>
  (attribution.value?.factors?.rows || []).filter((row) => row.group === "风格偏好")
);
const factorTableRows = computed(() => attribution.value?.factors?.rows || []);
const factorTableSections = computed(() => {
  const sections = [];
  for (const row of factorTableRows.value) {
    const last = sections[sections.length - 1];
    if (!last || last.group !== row.group) {
      sections.push({ group: row.group, rows: [row] });
    } else {
      last.rows.push(row);
    }
  }
  return sections;
});
const treeRows = computed(() => attribution.value?.returns_tree || []);
const treeFlow = computed(() => nestTree(treeRows.value));

const pnlTraces = computed(() => {
  const rows = attribution.value?.pnl || [];
  if (!rows.length) return [];
  const x = rows.map((row) => row.period);
  return [
    { x, y: cum(rows.map((row) => row.trade_return)), type: "scatter", mode: "lines", name: "交易（累计）" },
    { x, y: cum(rows.map((row) => row.holding_return)), type: "scatter", mode: "lines", name: "持仓（累计）" },
  ];
});

const industryTraces = computed(() => {
  const rows = (attribution.value?.industry?.rows || []).slice(0, 12);
  if (!rows.length) return [];
  const x = rows.map((row) => row.industry);
  return [
    { x, y: rows.map((row) => row.allocation), type: "bar", name: "配置" },
    { x, y: rows.map((row) => row.selection), type: "bar", name: "选择" },
  ];
});

const styleTraces = computed(() => {
  const rows = styleFactorRows.value;
  if (!rows.length) return [];
  const ordered = [...rows].sort((a, b) => Number(a.contribution) - Number(b.contribution));
  return [
    {
      y: ordered.map((row) => row.name),
      x: ordered.map((row) => row.contribution),
      type: "bar",
      orientation: "h",
      name: "风格贡献",
      marker: {
        color: ordered.map((row) => (Number(row.contribution) < 0 ? "#d75c66" : "#0b8f79")),
      },
    },
  ];
});

const groupBarWidth = computed(() => {
  const rows = factorGroups.value;
  const peak = Math.max(...rows.map((row) => Math.abs(Number(row.contribution) || 0)), 1e-9);
  return (row) => `${Math.min(100, Math.max(8, (Math.abs(Number(row.contribution) || 0) / peak) * 100))}%`;
});

function fieldOf(key) {
  return strategyFields.value.find((item) => item.key === key);
}

function applyDefaults(data) {
  const d = data?.defaults || {};
  form.strategy = d.strategy || form.strategy;
  form.start = d.start || form.start;
  form.end = d.end || "";
  form.universe = d.universe || "all";
  form.allocator = d.allocator || "equal";
  form.allocator_lookback = d.allocator_lookback ?? 252;
  form.max_weight = d.max_weight ?? 1;
  form.cash = d.cash ?? 1000000;
  form.commission = d.commission ?? 0.0003;
  form.stamp = d.stamp ?? 0.0005;
  form.mu_model = d.mu_model || "geometric";
  form.risk_free_rate = d.risk_free_rate ?? 0.02;
  form.risk_aversion = d.risk_aversion ?? 1;
  form.l2_gamma = d.l2_gamma ?? 0.1;
  form.tc_rate = d.tc_rate ?? 0.001;
  form.tail_confidence = d.tail_confidence ?? 0.95;
  form.benchmark = d.benchmark || "";
  form.hold = d.hold ?? 6;
  form.n = d.n ?? null;
  form.anti_tail = Boolean(d.anti_tail);
}

function body() {
  const payload = {
    strategy: form.strategy,
    start: form.start || null,
    end: form.end || null,
    universe: form.universe || "all",
    allocator: form.allocator,
    allocator_lookback: Number(form.allocator_lookback) || 252,
    max_weight: Number(form.max_weight),
    cash: Number(form.cash),
    commission: Number(form.commission),
    stamp: Number(form.stamp),
    mu_model: form.mu_model,
    risk_free_rate: Number(form.risk_free_rate),
    risk_aversion: Number(form.risk_aversion),
    l2_gamma: Number(form.l2_gamma),
    tc_rate: Number(form.tc_rate),
    tail_confidence: Number(form.tail_confidence),
    anti_tail: Boolean(form.anti_tail),
    no_benchmark: Boolean(form.no_benchmark),
    benchmark: form.no_benchmark ? null : form.benchmark || null,
  };
  if (fieldOf("factor")) payload.factor = form.factor || null;
  if (fieldOf("hold")) payload.hold = form.hold == null || form.hold === "" ? null : Number(form.hold);
  if (fieldOf("n") && form.n != null && form.n !== "") payload.n = Number(form.n);
  if (form.target_return != null && form.target_return !== "") {
    payload.target_return = Number(form.target_return);
  }
  if (form.target_volatility != null && form.target_volatility !== "") {
    payload.target_volatility = Number(form.target_volatility);
  }
  return payload;
}

async function run() {
  if (fieldOf("factor") && !String(form.factor || "").trim()) {
    statusError.value = true;
    status.value = "topk 需要填写因子名。";
    return;
  }
  const current = ++runToken;
  statusError.value = false;
  computing.value = true;
  status.value = "已提交回测，等待后端…";
  try {
    const data = await postBacktest(body(), {
      shouldAbort: () => current !== runToken,
      onTick: (seconds, jobStatus, progress) => {
        if (current !== runToken) return;
        statusError.value = false;
        const phase = progress?.phase;
        const label =
          jobStatus === "queued" || phase === "queued"
            ? "排队中"
            : jobStatus === "retry"
              ? "等待后端响应"
              : phase === "loading_panel"
                ? "正在加载面板"
                : phase === "matching"
                  ? "正在逐日撮合"
                  : "回测进行中";
        status.value = `${label} … ${seconds}s（请等待，不要重复点击）`;
      },
    });
    if (current !== runToken) return;
    result.value = data;
    attribution.value = null;
    attrError.value = false;
    attrStatus.value = "这份回测还没有业绩归因。";
    const m = data.metrics || {};
    status.value = `${data.strategy} / ${data.allocator}  ${m.start || ""} → ${m.end || ""}`;
    await loadRuns();
  } catch (err) {
    if (err && err.name === "AbortError") return;
    if (current !== runToken) return;
    statusError.value = true;
    status.value = String(err.message || err);
  } finally {
    if (current === runToken) computing.value = false;
  }
}

async function loadRuns() {
  try {
    const data = await fetchBacktestRuns();
    runs.value = data.items || [];
  } catch {
    runs.value = [];
  }
}

async function openRun(id) {
  statusError.value = false;
  try {
    result.value = await fetchBacktestRun(id);
    attribution.value = null;
    attrStatus.value = "";
    attrError.value = false;
    const m = result.value.metrics || {};
    status.value = `已载入 ${result.value.strategy} / ${result.value.allocator}  ${m.start || ""} → ${m.end || ""}`;
    await peekAttribution(id);
  } catch (err) {
    statusError.value = true;
    status.value = String(err.message || err);
  }
}

async function peekAttribution(id) {
  attrError.value = false;
  try {
    attribution.value = await fetchAttribution(id);
    attrStatus.value = `已载入归因 ${String(attribution.value.created_at || "").slice(0, 19).replace("T", " ")}`;
  } catch (err) {
    attribution.value = null;
    attrStatus.value = String(err.message || err);
  }
}

async function runAttribution() {
  const id = result.value?.id;
  if (!id) {
    attrError.value = true;
    attrStatus.value = "请先跑完或载入一次回测。";
    return;
  }
  const current = ++attrToken;
  attrError.value = false;
  attributing.value = true;
  attrStatus.value = "已提交归因，等待后端…";
  try {
    const data = await postAttribution(id, {
      shouldAbort: () => current !== attrToken,
      onTick: (seconds, jobStatus, progress) => {
        if (current !== attrToken) return;
        attrError.value = false;
        const phase = progress?.phase;
        const label =
          jobStatus === "queued" || phase === "queued"
            ? "排队中"
            : jobStatus === "retry"
              ? "等待后端响应"
              : "正在还原持仓并归因";
        attrStatus.value = `${label} … ${seconds}s`;
      },
    });
    if (current !== attrToken) return;
    attribution.value = data;
    attrStatus.value = `归因完成（${data.status || "ok"}，收盘盯市）`;
    await loadRuns();
  } catch (err) {
    if (err && err.name === "AbortError") return;
    if (current !== attrToken) return;
    attrError.value = true;
    attrStatus.value = String(err.message || err);
  } finally {
    if (current === attrToken) attributing.value = false;
  }
}

function nestTree(rows) {
  if (!rows?.length) return null;
  const nodes = rows.map((row) => ({
    label: row.label,
    value: row.value,
    level: Number(row.level) || 0,
    tone: row.tone,
    children: [],
  }));
  const root = nodes[0];
  const stack = [root];
  for (const node of nodes.slice(1)) {
    while (stack.length && stack[stack.length - 1].level >= node.level) {
      stack.pop();
    }
    if (!stack.length) break;
    stack[stack.length - 1].children.push(node);
    stack.push(node);
  }
  return root;
}

function pct(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function signClass(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return "";
  return n < 0 ? "neg" : "pos";
}

function num(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function money(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 0 });
}

function cum(values) {
  let total = 0;
  return values.map((item) => {
    const n = Number(item);
    if (!Number.isNaN(n)) total += n;
    return total;
  });
}

function runLabel(item) {
  const m = item.metrics || {};
  const when = String(item.created_at || "").slice(0, 19).replace("T", " ");
  return `${item.strategy || item.tag} / ${item.allocator || "equal"}  ${when}${
    m.annualized == null ? "" : `  年化 ${pct(m.annualized)}`
  }`;
}

onMounted(async () => {
  try {
    const data = await fetchBacktestMeta();
    meta.value = data;
    applyDefaults(data);
    status.value = "选好策略和区间后点击「开始回测」。全市场回测可能要几分钟。";
  } catch (err) {
    statusError.value = true;
    status.value = "加载回测元数据失败，请先运行 python -m App。 " + err;
  }
  await loadRuns();
});
</script>

<template>
  <div class="page">
    <div class="hero">
      <div>
        <h1>策略回测</h1>
        <p class="meta">
          策略只给出目标权重；引擎按 T 收盘决策、T+1 开盘成交。归因是事后评估：收盘盯市拆交易/持仓，再按行业 BF 和本地 CNE5-lite 十风格剥开持仓主动收益。
        </p>
      </div>
      <div class="actions">
        <button class="primary" :disabled="computing || attributing" @click="run">
          {{ computing ? "回测中…" : "开始回测" }}
        </button>
        <button :disabled="computing || attributing || !result?.id" @click="runAttribution">
          {{ attributing ? "归因中…" : "业绩归因" }}
        </button>
      </div>
    </div>

    <div class="toolbar">
      <label class="field">
        策略
        <select v-model="form.strategy">
          <option v-for="item in meta?.strategies || []" :key="item.name" :value="item.name">
            {{ item.name }}
          </option>
        </select>
      </label>
      <label class="field">
        仓位
        <select v-model="form.allocator">
          <option v-for="name in meta?.allocators || []" :key="name" :value="name">{{ name }}</option>
        </select>
      </label>
      <label class="field">
        开始
        <input v-model="form.start" type="date" />
      </label>
      <label class="field">
        结束
        <input v-model="form.end" type="date" />
      </label>
      <label class="field">
        股票池
        <select v-model="form.universe">
          <option v-for="item in universes" :key="item.name" :value="item.name">
            {{ item.name }}
          </option>
        </select>
      </label>
      <label v-if="fieldOf('factor')" class="field">
        因子
        <input v-model="form.factor" placeholder="alpha001" />
      </label>
      <label v-if="fieldOf('hold')" class="field">
        取到第 N 名
        <input v-model.number="form.hold" type="number" min="1" />
      </label>
      <label v-if="fieldOf('n')" class="field">
        {{ fieldOf("n")?.label || "N" }}
        <input v-model.number="form.n" type="number" min="1" placeholder="默认" />
      </label>
      <label v-if="fieldOf('anti_tail')" class="check">
        <input v-model="form.anti_tail" type="checkbox" />
        防尾声
      </label>
    </div>

    <details class="more">
      <summary>费用、仓位上限与优化器参数</summary>
      <div class="toolbar">
        <label class="field">
          本金
          <input v-model.number="form.cash" type="number" min="0" step="10000" />
        </label>
        <label class="field">
          佣金
          <input v-model.number="form.commission" type="number" min="0" step="0.0001" />
        </label>
        <label class="field">
          印花税（卖出）
          <input v-model.number="form.stamp" type="number" min="0" step="0.0001" />
        </label>
        <label class="field">
          单票上限
          <input v-model.number="form.max_weight" type="number" min="0" max="1" step="0.05" />
        </label>
        <label class="field">
          回看天数
          <input v-model.number="form.allocator_lookback" type="number" min="2" />
        </label>
        <label class="field">
          μ 模型
          <select v-model="form.mu_model">
            <option v-for="name in meta?.mu_models || []" :key="name" :value="name">{{ name }}</option>
          </select>
        </label>
        <label class="check">
          <input v-model="form.no_benchmark" type="checkbox" />
          关闭基准
        </label>
      </div>
    </details>

    <p class="status" :class="{ error: statusError }">{{ status }}</p>

    <div v-if="kpis.length" class="kpis">
      <div v-for="item in kpis" :key="item.label" class="card">
        <div class="label">{{ item.label }}</div>
        <div class="value">{{ item.value }}</div>
      </div>
    </div>

    <ChartPanel
      v-if="traces.length"
      title="净值（起始=1）"
      :traces="traces"
      :height="380"
      :layout="{ yaxis: { title: { text: '净值' } } }"
    />

    <section v-if="result?.id" class="attr">
      <h2>业绩归因</h2>
      <p class="status" :class="{ error: attrError }">{{ attrStatus }}</p>
      <p v-if="attribution?.notes?.length" class="meta">{{ attribution.notes.filter(Boolean).join(" ") }}</p>
      <div v-if="attrKpis.length" class="kpis">
        <div v-for="item in attrKpis" :key="item.label" class="card">
          <div class="label">{{ item.label }}</div>
          <div class="value">{{ item.value }}</div>
        </div>
      </div>
      <div v-if="treeFlow" class="flow-panel">
        <h3>收益树</h3>
        <p class="hint">总收益拆成交易 / 杠杆 / 持仓；持仓再拆主动与基准，主动再拆配置与选择。</p>
        <div class="flow-scroll">
          <ReturnsFlow :node="treeFlow" />
        </div>
      </div>
      <ChartPanel
        v-if="pnlTraces.length"
        title="交易 vs 持仓（日贡献累计）"
        :traces="pnlTraces"
        :height="300"
        :layout="{ yaxis: { title: { text: '累计收益' }, tickformat: '.2%' } }"
      />
      <div v-if="styleTraces.length || factorGroups.length" class="split split-factor">
        <ChartPanel
          v-if="styleTraces.length"
          title="十风格贡献（横向）"
          :traces="styleTraces"
          :height="420"
          :layout="{
            showlegend: false,
            margin: { t: 40, r: 28, b: 40, l: 96 },
            xaxis: { title: { text: '贡献' }, tickformat: '.2%', zeroline: true, gridcolor: '#2c3a4f' },
            yaxis: { automargin: true, autorange: 'reversed', gridcolor: '#2c3a4f' },
          }"
        />
        <div v-if="factorGroups.length" class="tree-panel sources-panel">
          <h3>收益来源</h3>
          <p class="hint">风格 / 行业 / 市场联动 / 特异。</p>
          <div class="source-list">
            <div v-for="item in factorGroups" :key="item.group" class="source">
              <div class="source-head">
                <span>{{ item.group }}</span>
                <span class="tree-val" :class="signClass(item.contribution)">{{ pct(item.contribution) }}</span>
              </div>
              <div class="source-track">
                <div
                  class="source-fill"
                  :class="signClass(item.contribution)"
                  :style="{ width: groupBarWidth(item) }"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
      <table v-if="factorTableSections.length" class="ind">
        <thead>
          <tr>
            <th>因子</th>
            <th>组合暴露</th>
            <th>基准暴露</th>
            <th>主动暴露</th>
            <th>因子收益</th>
            <th>贡献</th>
          </tr>
        </thead>
        <tbody>
          <template v-for="section in factorTableSections" :key="section.group">
            <tr class="group-head">
              <td colspan="6">{{ section.group }}</td>
            </tr>
            <tr v-for="row in section.rows" :key="row.key">
              <td>{{ row.name }}</td>
              <td>{{ num(row.portfolio_exposure, 3) }}</td>
              <td>{{ num(row.benchmark_exposure, 3) }}</td>
              <td>{{ num(row.active_exposure, 3) }}</td>
              <td :class="signClass(row.factor_return)">{{ pct(row.factor_return) }}</td>
              <td :class="signClass(row.contribution)">{{ pct(row.contribution) }}</td>
            </tr>
          </template>
        </tbody>
      </table>
      <ChartPanel
        v-if="industryTraces.length"
        title="行业配置 / 选择（日度加总）"
        :traces="industryTraces"
        :height="320"
        :layout="{ barmode: 'group', yaxis: { title: { text: '贡献' }, tickformat: '.2%' } }"
      />
      <table v-if="attribution?.industry?.rows?.length" class="ind">
        <thead>
          <tr>
            <th>行业</th>
            <th>组合权重</th>
            <th>基准权重</th>
            <th>配置</th>
            <th>选择</th>
            <th>合计</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in attribution.industry.rows" :key="row.industry">
            <td>{{ row.industry }}</td>
            <td>{{ pct(row.portfolio_weight) }}</td>
            <td>{{ pct(row.benchmark_weight) }}</td>
            <td>{{ pct(row.allocation) }}</td>
            <td>{{ pct(row.selection) }}</td>
            <td>{{ pct(row.total_effect) }}</td>
          </tr>
        </tbody>
      </table>
      <ul v-if="attribution?.warnings?.length" class="warns">
        <li v-for="(item, idx) in attribution.warnings.slice(0, 8)" :key="idx">
          {{ item.message }}
        </li>
      </ul>
    </section>

    <section v-if="runs.length" class="runs">
      <h2>最近回测</h2>
      <button
        v-for="item in runs"
        :key="item.id"
        class="run"
        :class="{ current: result?.id === item.id }"
        @click="openRun(item.id)"
      >
        {{ runLabel(item) }}{{ item.has_attribution ? "  ·已归因" : "" }}
      </button>
    </section>
  </div>
</template>

<style scoped>
.hero { display: flex; justify-content: space-between; gap: 16px; align-items: start; flex-wrap: wrap; }
.actions { display: flex; gap: 8px; flex-shrink: 0; }
h1 { margin: 0 0 6px; font-size: clamp(22px, 2.4vw, 28px); }
h2 { margin: 20px 0 10px; font-size: 14px; color: var(--muted); font-weight: 600; }
.toolbar { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin: 16px 0 8px; }
.check {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--muted);
  font-size: 13px;
  min-height: 40px;
}
.more { margin: 8px 0 12px; color: var(--muted); }
.more summary { cursor: pointer; }
.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 16px 20px;
}
.label { color: var(--muted); font-size: 12px; }
.value { font-size: clamp(20px, 2.2vw, 28px); font-weight: 600; }
.runs { display: flex; flex-direction: column; gap: 6px; margin-top: 20px; }
.run {
  text-align: left;
  background: var(--panel);
  border: 1px solid var(--line);
}
.run.current { border-color: var(--accent); }
.attr { margin-top: 28px; }
.split {
  display: grid;
  gap: 16px;
  align-items: stretch;
  margin: 12px 0 16px;
}
.split > * { min-width: 0; }
.split-factor {
  grid-template-columns: minmax(0, 1.4fr) minmax(260px, 0.68fr);
}
.split:has(> :only-child) { grid-template-columns: 1fr; }
.split :deep(.panel) { height: 100%; }
@media (max-width: 1100px) {
  .split-factor { grid-template-columns: 1fr; }
}
@media (max-width: 720px) {
  .actions { width: 100%; }
  .actions button { flex: 1; }
}
.flow-panel,
.tree-panel {
  display: flex;
  flex-direction: column;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 12px 16px 12px;
}
.flow-panel { margin: 12px 0 16px; }
.flow-panel h3,
.tree-panel h3 { margin: 0 0 4px; font-size: 13px; }
.flow-scroll {
  overflow-x: auto;
  padding: 8px 4px 12px;
}
.hint { color: var(--muted); font-size: 12px; margin: 0 0 8px; line-height: 1.5; }
.tree-val { font-variant-numeric: tabular-nums; white-space: nowrap; }
.tree-val.pos,
.pos { color: #5ad8a6; }
.tree-val.neg,
.neg { color: #f07178; }
.sources-panel { min-height: 420px; }
.source-list {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: space-evenly;
  gap: 10px;
}
.source {
  margin: 0;
  padding: 12px 14px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #121a26;
}
.source-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  margin-bottom: 8px;
}
.source-track {
  height: 6px;
  border-radius: 99px;
  background: var(--line);
  overflow: hidden;
}
.source-fill {
  height: 100%;
  border-radius: 99px;
  background: #0b8f79;
}
.source-fill.neg { background: #d75c66; }
.group-head td {
  text-align: left !important;
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 0.04em;
  padding-top: 12px;
}
.ind {
  width: 100%;
  border-collapse: collapse;
  margin-top: 12px;
  font-size: 13px;
}
.ind th, .ind td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: right; }
.ind th:first-child, .ind td:first-child { text-align: left; }
.warns { color: var(--muted); font-size: 12px; padding-left: 18px; }
</style>
