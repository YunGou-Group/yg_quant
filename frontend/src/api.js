export async function fetchMeta() {
  return fetchJson("/api/meta");
}

export async function fetchMetaRetrying(onWait, attempts = 40, delayMs = 500) {
  let lastError;
  for (let i = 0; i < attempts; i += 1) {
    try {
      return await fetchMeta();
    } catch (err) {
      lastError = err;
      if (onWait) onWait(i + 1, attempts);
      await sleep(delayMs);
    }
  }
  throw lastError;
}

// 所有后台任务共用一套轮询：连续失败有上限，瞬时网络错误才重试，
// 支持外部中止。后端重启或 job 被 TTL 清掉时不会无限转圈。
const MAX_MISSES = 90;
const MAX_DONE_WITHOUT_RESULT = 15;
const TRANSIENT_RE = /Failed to fetch|NetworkError|network|ECONNRESET|fetch/i;

export async function pollJob(jobId, options = {}) {
  if (!jobId) throw new Error("后端没有返回 job_id");
  const url = options.url || `/api/jobs/${encodeURIComponent(jobId)}`;
  const interval = options.intervalMs || 1000;
  const label = options.label || "任务";
  const requireResult = options.requireResult !== false;
  const t0 = Date.now();
  let misses = 0;
  while (true) {
    if (options.shouldAbort && options.shouldAbort()) {
      const err = new Error("aborted");
      err.name = "AbortError";
      throw err;
    }
    await sleep(interval);
    const seconds = Math.round((Date.now() - t0) / 1000);
    try {
      const poll = await fetch(url);
      const data = await poll.json().catch(() => ({}));
      if (!poll.ok) {
        misses += 1;
        if (options.onTick) options.onTick(seconds, "retry", data.detail || poll.statusText);
        if (misses > MAX_MISSES) throw new Error(data.detail || `${label}查询失败 ${poll.status}`);
        continue;
      }
      if (options.onTick) options.onTick(seconds, data.status, data.progress);
      if (data.status === "error") throw new Error(data.error || `${label}失败`);
      if (data.status === "done") {
        if (!requireResult) return data;
        if (data.result) return data.result;
        misses += 1;
        if (misses > MAX_DONE_WITHOUT_RESULT) throw new Error(`${label}完成但没有读到结果`);
        continue;
      }
      misses = 0;
    } catch (err) {
      if (err && err.name === "AbortError") throw err;
      const msg = String((err && err.message) || err || "");
      if (!TRANSIENT_RE.test(msg)) throw err;
      misses += 1;
      if (options.onTick) options.onTick(seconds, "retry", msg);
      if (misses > MAX_MISSES) throw err;
    }
  }
}

export async function postEval(body, options = {}) {
  const started = await postJob(body);
  if (started?.metrics && !started.job_id) return started;
  return pollJob(started.job_id, { ...options, label: "评估" });
}

async function postJob(body) {
  const res = await fetch("/api/eval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const started = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(started.detail || JSON.stringify(started) || res.statusText);
  return started;
}

export async function fetchResults() {
  return fetchJson("/api/results");
}

export async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function fetchDatasets() {
  return fetchJson("/api/datasets");
}

export async function fetchDatasetCoverage(runId, factor) {
  const params = new URLSearchParams();
  if (factor) params.set("factor", factor);
  const q = params.toString();
  return fetchJson(`/api/datasets/${encodeURIComponent(runId)}/coverage${q ? `?${q}` : ""}`);
}

export async function fetchDatasetCompare(left, right) {
  const params = new URLSearchParams({ left, right });
  return fetchJson(`/api/datasets/compare?${params}`);
}

export async function patchDataset(runId, body) {
  const res = await fetch(`/api/datasets/${encodeURIComponent(runId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

export async function deleteDataset(runId) {
  const res = await fetch(`/api/datasets/${encodeURIComponent(runId)}`, { method: "DELETE" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

export async function fetchFactorTable(runId) {
  const q = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return fetchJson(`/api/factors${q}`);
}

export async function fetchFactorDetail(name, runId) {
  const q = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return fetchJson(`/api/factors/${encodeURIComponent(name)}${q}`);
}

export async function fetchFactorSeries(name, metric, runId) {
  const params = new URLSearchParams({ metric });
  if (runId) params.set("run_id", runId);
  return fetchJson(`/api/factors/${encodeURIComponent(name)}/series?${params}`);
}

export async function fetchFactorSeriesBundle(name, runId, metrics) {
  const params = new URLSearchParams();
  if (runId) params.set("run_id", runId);
  if (metrics?.length) params.set("metrics", metrics.join(","));
  const q = params.toString();
  return fetchJson(`/api/factors/${encodeURIComponent(name)}/series-bundle${q ? `?${q}` : ""}`);
}

export async function fetchFamilies(runId) {
  const q = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return fetchJson(`/api/families${q}`);
}

export async function fetchFamilyEval(runId, extra = {}) {
  const params = new URLSearchParams();
  if (runId) params.set("run_id", runId);
  if (extra.start) params.set("start", extra.start);
  if (extra.end) params.set("end", extra.end);
  const q = params.toString();
  return fetchJson(`/api/families-eval${q ? `?${q}` : ""}`);
}

export async function previewSelect(body) {
  const res = await fetch("/api/select/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

export async function fetchDocs() {
  return fetchJson("/api/docs");
}

export async function fetchDoc(slug) {
  return fetchJson(`/api/docs/${encodeURIComponent(slug)}`);
}

export async function fetchBacktestMeta() {
  return fetchJson("/api/backtest/meta");
}

export async function fetchBacktestRuns() {
  return fetchJson("/api/backtest/runs");
}

export async function fetchBacktestRun(runId) {
  return fetchJson(`/api/backtest/runs/${encodeURIComponent(runId)}`);
}

export async function fetchAttribution(runId) {
  return fetchJson(`/api/backtest/runs/${encodeURIComponent(runId)}/attribution`);
}

export async function postAttribution(runId, options = {}) {
  const res = await fetch(`/api/backtest/runs/${encodeURIComponent(runId)}/attribution`, {
    method: "POST",
  });
  const started = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(started.detail || JSON.stringify(started) || res.statusText);
  return pollJob(started.job_id, { ...options, label: "业绩归因" });
}

export async function postBacktest(body, options = {}) {
  const res = await fetch("/api/backtest/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const started = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(started.detail || JSON.stringify(started) || res.statusText);
  return pollJob(started.job_id, { ...options, label: "回测" });
}

export async function postBatchEval(body, options = {}) {
  const res = await fetch("/api/batch/eval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const started = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(started.detail || res.statusText);
  return pollJob(started.job_id, {
    ...options,
    label: "全库评估",
    intervalMs: 1500,
  });
}

export async function fetchStyleCrowdingBootstrap() {
  return fetchJson("/api/style-crowding/bootstrap");
}

export async function fetchStyleCrowdingSeries(params) {
  const q = new URLSearchParams(params);
  return fetchJson(`/api/style-crowding/series?${q}`);
}

export async function postStyleCrowdingRun(body) {
  const res = await fetch("/api/style-crowding/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

export async function pollStyleCrowdingJob(jobId, options = {}) {
  return pollJob(jobId, {
    ...options,
    label: "风格拥挤计算",
    requireResult: false,
    intervalMs: options.intervalMs || 2000,
    url: `/api/style-crowding/jobs/${encodeURIComponent(jobId)}`,
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function errorText(res) {
  try {
    const data = await res.json();
    return data.detail || JSON.stringify(data);
  } catch {
    return res.statusText;
  }
}
