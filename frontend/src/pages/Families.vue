<script setup>
import { onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { fetchFamilies } from "../api.js";
import { fmt } from "../format.js";
import { useRunWorkspace } from "../runWorkspace.js";

const router = useRouter();
const { runId, runLink } = useRunWorkspace();
const error = ref("");
const items = ref([]);
const open = ref(null);

async function load() {
  error.value = "";
  open.value = null;
  if (!runId.value) {
    items.value = [];
    return;
  }
  try {
    const data = await fetchFamilies(runId.value);
    items.value = data.items || [];
  } catch (err) {
    error.value = String(err.message || err);
  }
}

function openMember(name) {
  if (!name) return;
  router.push(runLink(`/detail/${encodeURIComponent(name)}`));
}

onMounted(load);
watch(runId, load);
</script>

<template>
  <div class="page">
    <h1>因子家族</h1>
    <p class="meta">alphaNNN 归入 alpha101，style_* 归入 barra，其余按前缀。按 RankIC IR 排序。</p>
    <p v-if="error" class="status error">{{ error }}</p>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>家族</th>
            <th>成员</th>
            <th>RankIC IR</th>
            <th>RankIC</th>
            <th>覆盖率</th>
            <th>最佳</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="item in items"
            :key="item.family"
            :class="{ open: open?.family === item.family }"
            @click="open = item"
          >
            <td>{{ item.family }}</td>
            <td>{{ item.n }}</td>
            <td>{{ fmt(item.rank_ic_ir_mean) }}</td>
            <td>{{ fmt(item.rank_ic_mean) }}</td>
            <td>{{ fmt(item.coverage_rate_mean) }}</td>
            <td>{{ item.best_member || "—" }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <div v-if="open" class="drawer">
      <h2>{{ open.family }}</h2>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>因子</th>
              <th>RankIC IR</th>
              <th>RankIC</th>
              <th>覆盖率</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in open.members" :key="row.factor_name" @click="openMember(row.factor_name)">
              <td>{{ row.factor_name }}</td>
              <td>{{ fmt(row.rank_ic_ir) }}</td>
              <td>{{ fmt(row.rank_ic_mean) }}</td>
              <td>{{ fmt(row.coverage_rate_mean) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<style scoped>
h1 { font-size: clamp(22px, 2.4vw, 28px); }
.table-scroll { border: none; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; }
tbody tr { cursor: pointer; }
tbody tr:hover, tbody tr.open { background: #243044; }
.drawer { margin-top: 20px; }
h2 { margin: 0 0 8px; }
</style>
