<script setup>
import { computed } from "vue";

const props = defineProps({
  metrics: { type: Array, default: () => [] },
});

const selected = defineModel("selected", { type: Array, default: () => [] });

const DIMENSION_ORDER = [
  "预测力",
  "有效期",
  "失效风险",
  "暴露度",
  "风格暴露",
  "风格归因",
];

const groups = computed(() => {
  const buckets = new Map();
  for (const metric of props.metrics) {
    const key = metric.dimension || "其他";
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(metric);
  }
  const ordered = DIMENSION_ORDER.filter((key) => buckets.has(key)).map((key) => ({
    dimension: key,
    metrics: buckets.get(key),
  }));
  for (const [key, items] of buckets) {
    if (!DIMENSION_ORDER.includes(key)) {
      ordered.push({ dimension: key, metrics: items });
    }
  }
  return ordered;
});

function tooltip(metric) {
  return metric.description || "";
}

function selectedCount(group) {
  const names = new Set(selected.value);
  return group.metrics.filter((item) => names.has(item.name)).length;
}

function toggleGroup(group) {
  const names = group.metrics.map((item) => item.name);
  const current = new Set(selected.value);
  const allOn = names.every((name) => current.has(name));
  if (allOn) {
    selected.value = selected.value.filter((name) => !names.includes(name));
    return;
  }
  const next = [...selected.value];
  for (const name of names) {
    if (!current.has(name)) next.push(name);
  }
  selected.value = next;
}
</script>

<template>
  <div class="wrap">
    <div class="groups">
      <section v-for="group in groups" :key="group.dimension" class="group">
        <button type="button" class="group-head" @click="toggleGroup(group)">
          <span>{{ group.dimension }}</span>
          <span class="count">{{ selectedCount(group) }}/{{ group.metrics.length }}</span>
        </button>
        <div class="metrics">
          <label
            v-for="metric in group.metrics"
            :key="metric.name"
            :title="tooltip(metric)"
          >
            <input type="checkbox" :value="metric.name" v-model="selected" />
            {{ metric.name }}
          </label>
        </div>
      </section>
    </div>
    <details class="docs-card">
      <summary>
        <span>指标说明</span>
        <span class="hint">字段、维度与含义</span>
      </summary>
      <div class="docs-body">
        <table class="docs">
          <thead>
            <tr>
              <th>指标</th>
              <th>字段</th>
              <th>维度</th>
              <th>含义</th>
            </tr>
          </thead>
          <tbody>
            <template v-for="metric in metrics" :key="metric.name">
              <tr
                v-for="(field, index) in metric.fields || []"
                :key="metric.name + '.' + field.name"
              >
                <td v-if="index === 0" :rowspan="(metric.fields || []).length">
                  {{ metric.name }}
                </td>
                <td>{{ field.label }}</td>
                <td>{{ field.dimension }}</td>
                <td>{{ field.meaning }}</td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </details>
  </div>
</template>

<style scoped>
.wrap {
  margin-bottom: 12px;
}

.groups {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 10px;
  margin-bottom: 10px;
}

.group {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px 10px 10px;
}

.group-head {
  display: flex;
  width: 100%;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  padding: 2px 0;
  border: none;
  background: transparent;
  color: var(--text);
  font-size: 12px;
  font-weight: 600;
}

.group-head:hover {
  color: var(--accent);
}

.count {
  color: var(--muted);
  font-weight: 400;
}

.metrics {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.metrics label {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--muted);
  cursor: help;
}

.docs-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}

.docs-card summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  cursor: pointer;
  list-style: none;
  user-select: none;
}

.docs-card summary::-webkit-details-marker {
  display: none;
}

.docs-card summary::after {
  content: "▾";
  color: var(--muted);
  font-size: 12px;
}

.docs-card[open] summary::after {
  content: "▴";
}

.docs-card summary .hint {
  margin-right: auto;
  margin-left: 10px;
  color: var(--muted);
  font-size: 12px;
}

.docs-body {
  padding: 0 12px 10px;
  border-top: 1px solid var(--line);
}

.docs {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

th,
td {
  text-align: left;
  padding: 6px 8px;
  border-bottom: 1px solid var(--line);
  color: var(--muted);
  vertical-align: top;
}

th {
  font-weight: 500;
  color: var(--text);
}
</style>
