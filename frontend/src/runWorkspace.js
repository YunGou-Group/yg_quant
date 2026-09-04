import { computed, inject, onMounted, provide, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { fetchDatasets } from "./api.js";

const KEY = Symbol("runWorkspace");
const STORAGE = "qmt-selected-run";

export function provideRunWorkspace() {
  const route = useRoute();
  const router = useRouter();
  const datasets = ref([]);
  const loading = ref(false);

  const runId = computed(() => String(route.query.run || ""));
  const current = computed(
    () => datasets.value.find((item) => item.run_id === runId.value) || null
  );

  function runLabel(item) {
    if (!item) return "";
    return item.display_label || item.label || item.run_id || "";
  }

  function runLink(path, extraQuery = {}) {
    const query = { ...extraQuery };
    if (runId.value) query.run = runId.value;
    return { path, query };
  }

  function setRun(id, { replace = false } = {}) {
    if (!id) return;
    localStorage.setItem(STORAGE, id);
    const query = { ...route.query, run: id };
    const loc = { query };
    if (replace) router.replace(loc);
    else router.push(loc);
  }

  function ensureRun() {
    const items = datasets.value;
    if (!items.length) return;
    const valid = new Set(items.map((item) => item.run_id));
    const q = String(route.query.run || "");
    const stored = localStorage.getItem(STORAGE) || "";
    let next = "";
    if (q && valid.has(q)) next = q;
    else if (stored && valid.has(stored)) next = stored;
    else next = items[0].run_id;
    if (next && next !== q) setRun(next, { replace: true });
    else if (next) localStorage.setItem(STORAGE, next);
  }

  async function loadDatasets() {
    loading.value = true;
    try {
      const data = await fetchDatasets();
      datasets.value = data.items || [];
      ensureRun();
    } catch {
      datasets.value = [];
    } finally {
      loading.value = false;
    }
  }

  watch(
    () => route.query.run,
    (id) => {
      if (id) localStorage.setItem(STORAGE, String(id));
    }
  );

  onMounted(loadDatasets);

  const state = {
    datasets,
    loading,
    runId,
    current,
    runLabel,
    runLink,
    setRun,
    loadDatasets,
  };
  provide(KEY, state);
  return state;
}

export function useRunWorkspace() {
  const ctx = inject(KEY);
  if (!ctx) throw new Error("run workspace 未提供");
  return ctx;
}
