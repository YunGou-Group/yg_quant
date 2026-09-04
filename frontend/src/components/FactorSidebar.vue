<script setup>
import { computed } from "vue";

const props = defineProps({
  factors: { type: Array, default: () => [] },
  modelValue: { type: String, default: null },
  search: { type: String, default: "" },
});

const emit = defineEmits(["update:modelValue", "update:search"]);

const filtered = computed(() => {
  const query = (props.search || "").toLowerCase();
  return (props.factors || []).filter(
    (name) => !query || name.toLowerCase().includes(query)
  );
});

const groups = computed(() => {
  const risk = [];
  const alpha = [];
  for (const name of filtered.value) {
    if (String(name).startsWith("style_")) risk.push(name);
    else alpha.push(name);
  }
  const out = [];
  if (risk.length) out.push({ title: "风险 / 风格", items: risk });
  if (alpha.length) out.push({ title: "Alpha", items: alpha });
  return out;
});

function select(name) {
  emit("update:modelValue", name);
}
</script>

<template>
  <aside>
    <input
      :value="search"
      placeholder="搜索因子"
      @input="emit('update:search', $event.target.value)"
    />
    <template v-for="group in groups" :key="group.title">
      <div class="group">{{ group.title }}</div>
      <div
        v-for="name in group.items"
        :key="name"
        class="factor"
        :class="{ active: name === modelValue }"
        @click="select(name)"
      >
        {{ name }}
      </div>
    </template>
  </aside>
</template>

<style scoped>
aside {
  border-right: 1px solid var(--line);
  padding: 12px;
  overflow: auto;
  background: #141c27;
}

aside input {
  width: 100%;
  margin-bottom: 8px;
}

.group {
  margin: 10px 0 4px;
  padding: 0 8px;
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 0.04em;
}

.factor {
  padding: 6px 8px;
  border-radius: 4px;
  cursor: pointer;
  color: var(--muted);
}

.factor:hover {
  background: #243044;
  color: var(--text);
}

.factor.active {
  background: #1e3a5f;
  color: var(--accent);
}
</style>
