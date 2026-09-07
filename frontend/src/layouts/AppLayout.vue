<script setup>
import { computed, ref, watch } from "vue";
import { RouterLink, RouterView, useRoute } from "vue-router";
import { provideRunWorkspace } from "../runWorkspace.js";

const route = useRoute();
const { datasets, runId, runLabel, runLink, setRun } = provideRunWorkspace();
const navOpen = ref(false);

watch(
  () => route.fullPath,
  () => {
    navOpen.value = false;
  }
);

const isHome = computed(() => route.path === "/");
const isBacktest = computed(() => route.path.startsWith("/backtest"));
const isFactors = computed(() => !isHome.value && !isBacktest.value);

const modules = [
  { to: "/factors", label: "因子分析", key: "factors" },
  { to: "/backtest", label: "策略回测", key: "backtest" },
];

const factorNav = [
  { to: "/factors", label: "因子总览", key: "overview" },
  { to: "/detail", label: "深度报告", key: "detail" },
  { to: "/families", label: "因子家族", key: "families" },
  { to: "/families-eval", label: "家族评估", key: "families-eval" },
  { to: "/filter", label: "因子筛选", key: "filter" },
  { to: "/batch", label: "全库评估", key: "batch" },
  { to: "/style-crowding", label: "风格拥挤", key: "style-crowding" },
  { to: "/eval", label: "交互评估", key: "eval" },
  { to: "/docs", label: "文档", key: "docs" },
];

function moduleActive(item) {
  if (item.key === "factors") return isFactors.value;
  return isBacktest.value;
}

function factorActive(item) {
  const path = route.path;
  if (item.to === "/factors") return path === "/factors";
  return path === item.to || path.startsWith(item.to + "/");
}
</script>

<template>
  <div class="shell">
    <header class="top">
      <RouterLink to="/" class="brand">yg_quant</RouterLink>
      <button
        class="menu"
        type="button"
        :aria-expanded="navOpen"
        aria-label="打开导航"
        @click="navOpen = !navOpen"
      >
        <span /><span /><span />
      </button>
      <nav :class="{ open: navOpen }">
        <RouterLink
          v-for="item in modules"
          :key="item.key"
          :to="item.key === 'factors' ? runLink(item.to) : item.to"
          class="nav-item"
          :class="{ active: moduleActive(item) }"
        >
          {{ item.label }}
        </RouterLink>
        <template v-if="isFactors">
          <span class="nav-split" />
          <RouterLink
            v-for="item in factorNav"
            :key="item.key"
            :to="runLink(item.to)"
            class="nav-item sub"
            :class="{ active: factorActive(item) }"
          >
            {{ item.label }}
          </RouterLink>
        </template>
      </nav>
      <label v-if="isFactors" class="run-pick" title="当前全库结果">
        数据集
        <select :value="runId" :disabled="!datasets.length" @change="setRun($event.target.value)">
          <option v-if="!datasets.length" value="">还没有全库结果</option>
          <option v-for="item in datasets" :key="item.run_id" :value="item.run_id">
            {{ runLabel(item) }}
          </option>
        </select>
      </label>
    </header>
    <div class="body">
      <RouterView :key="(isBacktest ? 'backtest' : runId) || 'none'" />
    </div>
  </div>
</template>

<style scoped>
.shell {
  height: 100%;
  display: flex;
  flex-direction: column;
}
.top {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 0 var(--page-pad-x);
  min-height: var(--header-h);
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  flex-shrink: 0;
  flex-wrap: wrap;
}
.menu {
  display: none;
  flex-direction: column;
  justify-content: center;
  gap: 4px;
  width: 40px;
  height: 40px;
  padding: 8px;
  margin-left: auto;
  background: transparent;
}
.menu span {
  display: block;
  height: 2px;
  background: var(--text);
  border-radius: 1px;
}
.brand {
  font-weight: 700;
  letter-spacing: 0.04em;
  color: var(--text);
  text-decoration: none;
}
nav {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
  flex: 1;
  align-items: center;
}
.nav-split {
  width: 1px;
  height: 18px;
  background: var(--line);
  margin: 0 6px;
}
.nav-item {
  color: var(--muted);
  text-decoration: none;
  padding: 6px 10px;
  border-radius: 6px;
}
.nav-item.sub {
  font-size: 13px;
}
.nav-item.active {
  color: #081018;
  background: var(--accent);
}
.run-pick {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 12px;
}
.run-pick select {
  max-width: 360px;
}
.body {
  flex: 1;
  min-height: 0;
  overflow: auto;
  height: 100%;
}
@media (max-width: 900px) {
  .menu {
    display: inline-flex;
  }
  nav {
    display: none;
    flex-basis: 100%;
    order: 4;
    padding: 0 0 10px;
  }
  nav.open {
    display: flex;
  }
  .nav-split {
    display: none;
  }
  .run-pick {
    flex-basis: 100%;
    order: 5;
    padding-bottom: 10px;
  }
  .run-pick select {
    max-width: none;
    flex: 1;
  }
}
</style>
