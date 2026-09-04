<script setup>
import { computed } from "vue";
import { fmt, nestedScalar } from "../format.js";

const props = defineProps({
  items: { type: Array, default: () => [] },
  sortKey: { type: String, default: "icir" },
  sortDir: { type: Number, default: -1 },
});

const emit = defineEmits(["update:sortKey", "update:sortDir", "select"]);

const columns = [
  { key: "factor", label: "因子" },
  { key: "horizon", label: "N" },
  { key: "universe", label: "股票池" },
  { key: "icir", label: "ICIR", path: "icir.icir" },
  { key: "ic_mean", label: "RankIC均值", path: "rank_ic.mean" },
  { key: "pure_ic", label: "纯化IC", path: "pure_ic.mean" },
  { key: "pure_icir", label: "纯化ICIR", path: "pure_ic.icir" },
  { key: "exp_size", label: "|maβ|Size", path: "exposure.ma_abs_style_size" },
  { key: "exp_ind", label: "|maβ|行业", path: "exposure.ma_abs_industry" },
  { key: "attr_ind", label: "累计行业", path: "attribution.cum_attr_industry" },
  { key: "attr_resid", label: "累计残差", path: "attribution.cum_residual" },
  { key: "resid_share", label: "残差占比", path: "attribution.residual_share" },
  { key: "positive", label: "正值比例", path: "rank_ic.positive_ratio" },
  { key: "ma_slow", label: "一致性", path: "long_short_term_consistency.last" },
  { key: "trend", label: "一致性均值", path: "long_short_term_consistency.mean" },
  { key: "spread", label: "分层价差", path: "quantile.spread" },
  { key: "coverage", label: "覆盖率", path: "coverage_rate.mean" },
  { key: "start", label: "起" },
  { key: "end", label: "止" },
  { key: "computed_at", label: "计算时间" },
];

function resultValue(row, key) {
  const col = columns.find((item) => item.key === key);
  if (col?.path) return nestedScalar(row, col.path);
  if (key === "horizon") return row.params?.horizon;
  if (key === "universe") return row.params?.universe;
  return row[key];
}

const sorted = computed(() => {
  const rows = [...props.items];
  rows.sort((left, right) => {
    const a = resultValue(left, props.sortKey);
    const b = resultValue(right, props.sortKey);
    if (a == null && b == null) return 0;
    if (a == null) return 1;
    if (b == null) return -1;
    if (a < b) return -props.sortDir;
    if (a > b) return props.sortDir;
    return 0;
  });
  return rows;
});

function toggleSort(key) {
  if (props.sortKey === key) emit("update:sortDir", props.sortDir * -1);
  else {
    emit("update:sortKey", key);
    emit("update:sortDir", key === "factor" ? 1 : -1);
  }
}

function display(row, key) {
  const numeric = ["icir", "ic_mean", "pure_ic", "pure_icir", "exp_size", "exp_ind", "attr_ind", "attr_resid", "resid_share", "positive", "ma_slow", "trend", "spread", "coverage"];
  if (numeric.includes(key)) return fmt(resultValue(row, key));
  return resultValue(row, key) ?? "";
}
</script>

<template>
  <table>
    <thead>
      <tr>
        <th v-for="col in columns" :key="col.key" @click="toggleSort(col.key)">
          {{ col.label }}
        </th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="row in sorted" :key="row.factor + (row.computed_at || '')" @click="emit('select', row)">
        <td v-for="col in columns" :key="col.key">{{ display(row, col.key) }}</td>
      </tr>
    </tbody>
  </table>
</template>

<style scoped>
table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  text-align: left;
  padding: 8px;
  border-bottom: 1px solid var(--line);
  font-variant-numeric: tabular-nums;
}

th {
  color: var(--muted);
  font-weight: 500;
  cursor: pointer;
}

tr:hover td {
  background: #1e2838;
}
</style>
