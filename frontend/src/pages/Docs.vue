<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { fetchDoc, fetchDocs } from "../api.js";
import { useRunWorkspace } from "../runWorkspace.js";

const route = useRoute();
const router = useRouter();
const { runLink } = useRunWorkspace();
const items = ref([]);
const page = ref(null);
const error = ref("");
const slug = computed(() => route.params.slug || "contract");

async function loadList() {
  const data = await fetchDocs();
  items.value = data.items || [];
}

async function loadPage() {
  error.value = "";
  try {
    page.value = await fetchDoc(slug.value);
  } catch (err) {
    error.value = String(err.message || err);
  }
}

onMounted(async () => {
  await loadList();
  await loadPage();
});
watch(slug, loadPage);
</script>

<template>
  <div class="page">
    <aside>
      <button v-for="item in items" :key="item.slug" :class="{ active: item.slug === slug }" @click="router.push(runLink(`/docs/${item.slug}`))">
        {{ item.title }}
      </button>
    </aside>
    <article>
      <p v-if="error" class="status error">{{ error }}</p>
      <pre v-else>{{ page?.body }}</pre>
    </article>
  </div>
</template>

<style scoped>
.page { display: grid; grid-template-columns: 240px 1fr; height: 100%; }
aside { border-right: 1px solid var(--line); overflow: auto; padding: 12px; display: flex; flex-direction: column; gap: 4px; }
aside button { text-align: left; background: transparent; }
aside button.active { background: var(--accent); color: #081018; }
article { padding: 20px 24px; overflow: auto; }
pre { white-space: pre-wrap; font-family: inherit; }
</style>
