<script setup>
import ReturnsFlow from "./ReturnsFlow.vue";

defineOptions({ name: "ReturnsFlow" });

defineProps({
  node: { type: Object, required: true },
  nested: { type: Boolean, default: false },
});

function pct(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function signClass(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return "";
  return n < 0 ? "neg" : "pos";
}
</script>

<template>
  <ul v-if="!nested" class="org-list">
    <ReturnsFlow :node="node" nested />
  </ul>
  <li v-else :class="[`lv${node.level || 0}`, node.tone]">
    <div class="node">
      <span class="name">{{ node.label }}</span>
      <span class="val" :class="signClass(node.value)">{{ pct(node.value) }}</span>
    </div>
    <ul v-if="node.children?.length">
      <ReturnsFlow
        v-for="(child, idx) in node.children"
        :key="`${child.label}-${idx}`"
        :node="child"
        nested
      />
    </ul>
  </li>
</template>

<style scoped>
.org-list,
.org-list ul {
  display: flex;
  justify-content: center;
  margin: 0;
  padding: 20px 0 0;
  position: relative;
}
.org-list {
  padding-top: 0;
  min-width: max-content;
  width: 100%;
}
li {
  list-style: none;
  position: relative;
  padding: 20px 10px 0;
  text-align: center;
}
li::before,
li::after {
  content: "";
  position: absolute;
  top: 0;
  right: 50%;
  width: 50%;
  height: 20px;
  border-top: 1px solid #4a5d78;
}
li::after {
  right: auto;
  left: 50%;
  border-left: 1px solid #4a5d78;
}
li:only-child {
  padding-top: 0;
}
li:only-child::before,
li:only-child::after {
  display: none;
}
li:first-child::before,
li:last-child::after {
  border: 0 none;
}
li:last-child::before {
  border-right: 1px solid #4a5d78;
  border-radius: 0 6px 0 0;
}
ul ul::before {
  content: "";
  position: absolute;
  top: 0;
  left: 50%;
  height: 20px;
  border-left: 1px solid #4a5d78;
}
.node {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  min-width: 118px;
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: #121a26;
  position: relative;
  z-index: 1;
}
.name {
  font-size: 11px;
  color: var(--muted);
  line-height: 1.3;
}
.val {
  font-size: 14px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.val.pos { color: #5ad8a6; }
.val.neg { color: #f07178; }
.lv0 .node {
  min-width: 136px;
  border-color: #3d9cf0;
  background: #1a2a40;
  padding: 10px 14px;
}
.lv0 .name {
  color: var(--text);
  font-weight: 600;
}
.lv0 .val { font-size: 16px; }
.positive .node { border-color: #0b8f79; }
.muted .node {
  border-style: dashed;
  background: transparent;
}
</style>
