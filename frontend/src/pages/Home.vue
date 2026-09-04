<script setup>
import { RouterLink } from "vue-router";
import { useRunWorkspace } from "../runWorkspace.js";

const { runLink } = useRunWorkspace();
</script>

<template>
  <div class="page">
    <div class="hero">
      <h1>yg_quant</h1>
      <p class="meta">从这里进入因子分析或策略回测。两者并列，互不改写对方的结果。</p>
    </div>
    <div class="cards">
      <RouterLink class="card" :to="runLink('/factors')">
        <h2>因子分析</h2>
        <p>读全库结果、深度报告、家族评估、筛选和交互计算。信号契约仍是收盘 T、开盘 T+1。</p>
        <span class="go">进入 →</span>
      </RouterLink>
      <RouterLink class="card" to="/backtest">
        <h2>策略回测</h2>
        <p>选择策略与仓位分配，T 收盘给目标权重，T+1 开盘撮合。跑完可对这次结果做业绩归因（交易/持仓洋葱树、行业 Brinson、本地十风格）。</p>
        <span class="go">进入 →</span>
      </RouterLink>
    </div>
  </div>
</template>

<style scoped>
.page { padding: 40px 24px 48px; max-width: 920px; }
.hero { margin-bottom: 24px; }
h1 { margin: 0 0 8px; font-size: 28px; }
.cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 20px 22px;
  color: inherit;
  text-decoration: none;
  min-height: 180px;
}
.card:hover { border-color: var(--accent); }
h2 { margin: 0; font-size: 20px; }
.card p { margin: 0; color: var(--muted); line-height: 1.55; flex: 1; }
.go { color: var(--accent); font-size: 13px; }
@media (max-width: 720px) {
  .cards { grid-template-columns: 1fr; }
}
</style>
