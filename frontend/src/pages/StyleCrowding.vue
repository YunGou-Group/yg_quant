<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import {
  fetchStyleCrowdingBootstrap,
  fetchStyleCrowdingSeries,
  pollStyleCrowdingJob,
  postStyleCrowdingRun,
} from "../api.js";

const loading = ref(true);
const error = ref("");
const tab = ref("matrix");
const data = ref(null);

const style = ref("style_momentum");
const indicator = ref("01_valuation_bp_z");
const groupCount = ref(10);
const orientation = ref("positive");
const transform = ref("raw");
const series = ref(null);
const running = ref(false);
const runStatus = ref("");

const styles = computed(() => data.value?.styles || []);
const styleLabels = computed(() => data.value?.style_labels || {});
const indicators = computed(() => data.value?.indicators || []);
const groupCounts = computed(() => {
  const fromBoot = data.value?.group_counts;
  const fromMan = data.value?.manifest?.group_counts;
  return (fromBoot && fromBoot.length ? fromBoot : fromMan) || [10];
});
const orientations = computed(() => {
  const fromBoot = data.value?.orientations;
  const fromMan = data.value?.manifest?.orientations;
  return (fromBoot && fromBoot.length ? fromBoot : fromMan) || ["positive"];
});
const sliceNote = computed(() => {
  const g = groupCounts.value[0];
  const fromMan = data.value?.manifest?.group_counts;
  const shown = g ?? (fromMan && fromMan[0]) ?? "—";
  const o = orientations.value[0] || "positive";
  const oLabel = o === "reverse" ? "反向" : "正向";
  return `当前切片：${shown} 分位 · ${oLabel}（不是全部方向）`;
});

const seriesTail = computed(() => {
  const s = series.value;
  if (!s) return { value: null, asOf: null };
  if (s.last_finite != null && Number.isFinite(Number(s.last_finite))) {
    return { value: s.last_finite, asOf: s.as_of || null };
  }
  const vals = s.values || [];
  const dates = s.dates || [];
  for (let i = vals.length - 1; i >= 0; i--) {
    const v = vals[i];
    if (v != null && Number.isFinite(Number(v))) {
      return { value: v, asOf: dates[i] || null };
    }
  }
  return { value: null, asOf: null };
});

const matrix = computed(() => data.value?.matrix || []);
const market = computed(() => data.value?.market_exposure || {});
const events = computed(() => data.value?.event_analysis || {});

function label(name) {
  return styleLabels.value[name] || name;
}

function tempClass(t) {
  if (t == null) return "unknown";
  if (t >= 0.9) return "critical";
  if (t >= 0.75) return "elevated";
  if (t >= 0.5) return "watch";
  return "calm";
}

async function loadBootstrap() {
  loading.value = true;
  error.value = "";
  try {
    data.value = await fetchStyleCrowdingBootstrap();
    if (styles.value.length && !styles.value.includes(style.value)) {
      style.value = styles.value[0];
    }
    if (indicators.value.length && !indicators.value.find((i) => i.key === indicator.value)) {
      indicator.value = indicators.value[0].key;
    }
    if (groupCounts.value.length && !groupCounts.value.includes(groupCount.value)) {
      groupCount.value = groupCounts.value[0];
    }
    if (orientations.value.length && !orientations.value.includes(orientation.value)) {
      orientation.value = orientations.value[0];
    }
  } catch (err) {
    error.value = String(err.message || err);
  } finally {
    loading.value = false;
  }
}

async function loadSeries() {
  try {
    series.value = await fetchStyleCrowdingSeries({
      style: style.value,
      indicator: indicator.value,
      group_count: groupCount.value,
      orientation: orientation.value,
      transform: transform.value,
    });
  } catch (err) {
    series.value = { error: String(err.message || err) };
  }
}

async function runCrowding() {
  running.value = true;
  runStatus.value = "已提交…";
  try {
    const started = await postStyleCrowdingRun({ incremental: true });
    runStatus.value = `任务 ${started.job_id} 排队中`;
    // 提交后要一直轮询到结束再刷新，否则页面还停在旧的 published/ 上
    await pollStyleCrowdingJob(started.job_id, {
      shouldAbort: () => leaving,
      onTick(seconds, status, progress) {
        const phase = progress?.phase ? ` ${progress.phase}` : "";
        runStatus.value = `任务 ${started.job_id} ${status}${phase} ${seconds}s`;
      },
    });
    runStatus.value = "计算完成，正在刷新…";
    await loadBootstrap();
    await loadSeries();
    runStatus.value = "已更新";
  } catch (err) {
    if (err?.name !== "AbortError") runStatus.value = String(err.message || err);
  } finally {
    running.value = false;
  }
}

let leaving = false;

onMounted(async () => {
  await loadBootstrap();
  await loadSeries();
});

onUnmounted(() => {
  leaving = true;
});
</script>

<template>
  <div class="page">
    <header class="head">
      <div>
        <h1>风格拥挤</h1>
        <p class="muted">Barra Style 拥挤监测（独立风控模块，只读 published/）</p>
      </div>
      <button class="btn" :disabled="running" @click="runCrowding">增量发布</button>
    </header>

    <p v-if="runStatus" class="muted">{{ runStatus }}</p>
    <p v-if="error" class="err">{{ error }}</p>
    <p v-if="loading">加载中…</p>

    <nav v-if="!loading && !error" class="tabs">
      <button :class="{ active: tab === 'matrix' }" @click="tab = 'matrix'">雷达矩阵</button>
      <button :class="{ active: tab === 'workbench' }" @click="tab = 'workbench'">工作台</button>
      <button :class="{ active: tab === 'market' }" @click="tab = 'market'">全市场暴露</button>
      <button :class="{ active: tab === 'events' }" @click="tab = 'events'">事件分析</button>
    </nav>

    <section v-if="tab === 'matrix' && matrix.length" class="panel">
      <p class="muted">{{ sliceNote }}</p>
      <table class="matrix">
        <thead>
          <tr>
            <th>Style</th>
            <th v-for="key in data.core_indicators" :key="key">{{ key.split('_').slice(-2).join('_') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in matrix" :key="row.style">
            <td>{{ label(row.style) }}</td>
            <td v-for="key in data.core_indicators" :key="key">
              <span
                v-if="row.indicators[key]"
                class="cell"
                :class="tempClass(row.indicators[key].temperature)"
                :title="row.indicators[key].title"
              >
                {{ row.indicators[key].temperature?.toFixed(2) ?? "—" }}
              </span>
              <span v-else>—</span>
            </td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-if="tab === 'workbench'" class="panel">
      <div class="controls">
        <label>Style <select v-model="style" @change="loadSeries"><option v-for="s in styles" :key="s" :value="s">{{ label(s) }}</option></select></label>
        <label>指标 <select v-model="indicator" @change="loadSeries"><option v-for="i in indicators" :key="i.key" :value="i.key">{{ i.title }}</option></select></label>
        <label>分组 <select v-model.number="groupCount" @change="loadSeries"><option v-for="g in groupCounts" :key="g" :value="g">{{ g }} 分位</option></select></label>
        <label>方向 <select v-model="orientation" @change="loadSeries"><option v-for="o in orientations" :key="o" :value="o">{{ o === "reverse" ? "反向" : "正向" }}</option></select></label>
        <label>变换 <select v-model="transform" @change="loadSeries"><option value="raw">raw</option><option value="prior_z">prior-Z</option><option value="percentile">分位</option></select></label>
      </div>
      <div v-if="series?.values" class="spark">
        <p class="muted">末值 {{ seriesTail.value ?? "—" }}（as of {{ seriesTail.asOf ?? "—" }}，{{ series.dates?.length }} 日）</p>
        <div class="bars">
          <span
            v-for="(v, idx) in series.values.slice(-120)"
            :key="idx"
            class="bar"
            :style="{ height: v != null ? `${Math.min(100, Math.abs(v) * 20)}%` : '2px' }"
          />
        </div>
      </div>
      <p v-else-if="series?.error" class="err">{{ series.error }}</p>
    </section>

    <section v-if="tab === 'market'" class="panel">
      <p class="muted">{{ sliceNote }}</p>
      <p v-if="market.composite_temperature != null">
        合成拥挤温度：<strong>{{ market.composite_temperature?.toFixed(3) }}</strong>
        （核心告警 {{ market.core_alarm_count ?? 0 }}）
      </p>
      <table v-if="market.core_indicators?.length" class="matrix">
        <thead><tr><th>Style</th><th>指标</th><th>分位</th><th>温度</th></tr></thead>
        <tbody>
          <tr v-for="(row, i) in market.core_indicators" :key="i">
            <td>{{ row.style_label }}</td>
            <td>{{ row.title }}</td>
            <td>{{ row.percentile?.toFixed(3) ?? "—" }}</td>
            <td>
              <span class="cell" :class="tempClass(row.temperature)">
                {{ row.temperature?.toFixed(3) ?? "—" }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-if="tab === 'events'" class="panel">
      <p class="muted">{{ sliceNote }}</p>
      <h3>活跃投票（{{ events.active_votes_by_style ? Object.keys(events.active_votes_by_style).length : 0 }} 个 style）</h3>
      <ul v-if="events.indicator_votes">
        <li v-for="(v, i) in events.indicator_votes.filter((x) => x.active).slice(0, 40)" :key="i">
          {{ v.style_label }} · {{ v.title }}
        </li>
      </ul>
      <h3>参考事件</h3>
      <ul v-if="events.reference_events">
        <li v-for="ev in events.reference_events" :key="ev.name">{{ ev.name }}（{{ ev.start }} ~ {{ ev.end }}）</li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.page { padding: 1rem 1.25rem; max-width: 1200px; }
.head { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.muted { color: var(--muted); font-size: 0.9rem; }
.err { color: var(--bad); }
.tabs { display: flex; gap: 0.5rem; margin: 1rem 0; flex-wrap: wrap; }
.tabs button { padding: 0.35rem 0.75rem; border: 1px solid var(--line); background: var(--panel); color: var(--text); cursor: pointer; border-radius: 4px; }
.tabs button.active { background: var(--accent); color: #081018; border-color: var(--accent); }
.panel { margin-top: 0.5rem; }
.matrix { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.matrix th, .matrix td { border: 1px solid var(--line); padding: 0.35rem 0.5rem; text-align: left; }
.cell {
  display: inline-block;
  min-width: 2.5rem;
  text-align: center;
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: #14181f;
}
.critical { background: #f87171; color: #14181f; }
.elevated { background: #fb923c; color: #14181f; }
.watch { background: #eab308; color: #14181f; }
.calm { background: #4ade80; color: #14181f; }
.unknown { background: #64748b; color: #f8fafc; }
.controls { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem; }
.controls label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.85rem; }
.btn { padding: 0.4rem 0.9rem; background: #1a56db; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
.bars { display: flex; align-items: flex-end; gap: 1px; height: 80px; overflow: hidden; }
.bar { flex: 1; background: #3b82f6; min-height: 2px; }
</style>
