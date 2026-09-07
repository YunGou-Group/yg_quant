<script setup>
import { RouterLink } from "vue-router";
import { useRunWorkspace } from "../runWorkspace.js";

const { runLink } = useRunWorkspace();
</script>

<template>
  <div class="page home">
    <div class="hero">
      <h1>yg_quant</h1>
      <p class="lead">从这里进入因子分析或策略回测。两者并列，互不改写对方的结果。</p>
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
.home {
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 100%;
  width: min(100%, 1200px);
  padding-top: clamp(40px, 8vh, 96px);
  padding-bottom: clamp(40px, 8vh, 96px);
}
.hero { margin-bottom: clamp(24px, 4vh, 40px); }
h1 {
  margin: 0 0 12px;
  font-size: clamp(32px, 4.5vw, 48px);
  letter-spacing: 0.02em;
}
.lead {
  margin: 0;
  max-width: 42em;
  color: var(--muted);
  font-size: clamp(15px, 1.6vw, 18px);
  line-height: 1.6;
}
.cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: clamp(16px, 2.5vw, 28px);
}
.card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: clamp(22px, 3vw, 36px);
  color: inherit;
  text-decoration: none;
  min-height: clamp(220px, 28vh, 320px);
}
.card:hover { border-color: var(--accent); }
h2 { margin: 0; font-size: clamp(20px, 2.2vw, 26px); }
.card p { margin: 0; color: var(--muted); line-height: 1.6; flex: 1; font-size: 15px; }
.go { color: var(--accent); font-size: 15px; }
@media (max-width: 720px) {
  .cards { grid-template-columns: 1fr; }
  .card { min-height: 180px; }
}
</style>
