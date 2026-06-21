const state = {
  experiments: [],
  language: getInitialLanguage(),
};

const jsonHeaders = { Accept: "application/json" };
const translations = {
  en: {
    appTitle: "TS Platform Dashboard",
    artifactsTitle: "Artifacts / Leaderboard Preview",
    backendStatus: "Backend status",
    checkpointPath: "checkpoint path",
    compareTitle: "Compare Demo",
    complete: "Complete",
    dataMetadata: "data_metadata",
    datasetCount: "Dataset count",
    documentTitle: "TS Platform Demo Dashboard",
    error: "Error",
    experimentCount: "Experiment count",
    experimentName: "experiment name",
    experimentNameRequired: "experiment_name is required",
    experimentsTitle: "Experiments",
    eyebrow: "Local demo",
    idle: "Idle",
    jobIdRequired: "job_id is required",
    jobsTitle: "Jobs",
    languageToggle: "中文",
    loadArtifacts: "Load Artifacts",
    loadJob: "Load Job",
    loadJobLogs: "Load Job Logs",
    loadJobResult: "Load Job Result",
    loadLeaderboard: "Load Leaderboard",
    loadResults: "Load Results",
    loading: "Loading...",
    modelNames: "Model names",
    none: "None",
    noRows: "No rows.",
    overviewTitle: "Overview",
    primaryMetric: "primary_metric",
    ready: "Ready",
    refresh: "Refresh",
    refreshExperiments: "Refresh Experiments",
    refreshJobs: "Refresh Jobs",
    runPrefix: "Run",
    runningPrefix: "Running",
    successCount: "success_count",
    failedCount: "failed_count",
    testMetricsOriginal: "test_metrics.original",
    trainTitle: "Train Demo",
    version: "Version",
  },
  zh: {
    appTitle: "时间序列平台 Dashboard",
    artifactsTitle: "产物 / 排行榜预览",
    backendStatus: "后端状态",
    checkpointPath: "checkpoint 路径",
    compareTitle: "对比演示",
    complete: "完成",
    dataMetadata: "数据元信息",
    datasetCount: "数据集数量",
    documentTitle: "时间序列平台 Demo Dashboard",
    error: "错误",
    experimentCount: "实验数量",
    experimentName: "实验名",
    experimentNameRequired: "请填写 experiment_name",
    experimentsTitle: "实验",
    eyebrow: "本地演示",
    idle: "空闲",
    jobIdRequired: "请填写 job_id",
    jobsTitle: "任务",
    languageToggle: "English",
    loadArtifacts: "加载产物",
    loadJob: "加载任务",
    loadJobLogs: "加载任务日志",
    loadJobResult: "加载任务结果",
    loadLeaderboard: "加载排行榜",
    loadResults: "加载结果",
    loading: "加载中...",
    modelNames: "模型列表",
    none: "无",
    noRows: "暂无数据。",
    overviewTitle: "概览",
    primaryMetric: "主指标",
    ready: "就绪",
    refresh: "刷新",
    refreshExperiments: "刷新实验",
    refreshJobs: "刷新任务",
    runPrefix: "运行",
    runningPrefix: "正在运行",
    successCount: "成功数量",
    failedCount: "失败数量",
    testMetricsOriginal: "原始尺度测试指标",
    trainTitle: "训练演示",
    version: "版本",
  },
};

document.addEventListener("DOMContentLoaded", () => {
  applyLanguage();
  document.querySelector("#language-toggle").addEventListener("click", toggleLanguage);
  document.querySelector("#refresh-overview").addEventListener("click", loadOverview);
  document.querySelector("#refresh-experiments").addEventListener("click", loadExperiments);
  document.querySelector("#refresh-jobs").addEventListener("click", loadJobs);

  document.querySelectorAll("[data-train-demo]").forEach((button) => {
    button.addEventListener("click", () => runTrainDemo(button.dataset.trainDemo, button));
  });
  document.querySelectorAll("[data-compare-demo]").forEach((button) => {
    button.addEventListener("click", () => runCompareDemo(button.dataset.compareDemo, button));
  });

  document.querySelector("#load-job").addEventListener("click", () => loadJobDetail("job"));
  document
    .querySelector("#load-job-result")
    .addEventListener("click", () => loadJobDetail("result"));
  document.querySelector("#load-job-logs").addEventListener("click", () => loadJobDetail("logs"));

  document.querySelector("#load-results").addEventListener("click", () => loadRunData("results"));
  document
    .querySelector("#load-leaderboard")
    .addEventListener("click", () => loadRunData("leaderboard"));
  document
    .querySelector("#load-artifacts")
    .addEventListener("click", () => loadRunData("artifacts"));

  loadOverview();
  loadExperiments();
  loadJobs();
});

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    headers: jsonHeaders,
    ...options,
  });
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : payload || response.statusText;
    throw new Error(`${response.status} ${formatValue(detail)}`);
  }
  return payload;
}

async function loadOverview() {
  const status = document.querySelector("#overview-status");
  const grid = document.querySelector("#overview-grid");
  const error = document.querySelector("#overview-error");
  status.textContent = t("loading");
  error.textContent = "";

  try {
    const [health, datasets, models, experiments] = await Promise.all([
      apiFetch("/health"),
      apiFetch("/datasets"),
      apiFetch("/models"),
      apiFetch("/experiments"),
    ]);
    const datasetCount = Array.isArray(datasets.datasets)
      ? datasets.datasets.length
      : (datasets.names || []).length;
    const modelNames = models.models || [];
    const experimentRows = experiments.experiments || [];
    state.experiments = experimentRows;
    grid.innerHTML = [
      metric(t("backendStatus"), health.status || "unknown"),
      metric(t("version"), health.version || "unknown"),
      metric(t("datasetCount"), datasetCount),
      metric(t("modelNames"), modelNames.join(", ") || t("none")),
      metric(t("experimentCount"), experimentRows.length),
    ].join("");
    status.textContent = t("ready");
  } catch (errorValue) {
    status.textContent = t("error");
    grid.innerHTML = "";
    error.textContent = errorValue.message;
  }
}

async function runTrainDemo(demoName, button) {
  const status = document.querySelector("#train-status");
  const output = document.querySelector("#train-output");
  await withButton(button, status, `${t("runningPrefix")} ${demoName}...`, async () => {
    output.innerHTML = "";
    const payload = await apiFetch(`/demo/train/${demoName}`, { method: "POST" });
    output.innerHTML = [
      renderKeyValues({
        [t("experimentName")]: payload.experiment_name,
        run_id: payload.run_id,
        [t("checkpointPath")]: payload.checkpoint_path,
      }),
      `<h3>${escapeHtml(t("testMetricsOriginal"))}</h3>`,
      renderJson(payload.test_metrics ? payload.test_metrics.original : null),
      `<h3>${escapeHtml(t("dataMetadata"))}</h3>`,
      renderJson(payload.data_metadata || null),
    ].join("");
    fillRunInputs(payload.experiment_name, payload.run_id);
    status.textContent = t("complete");
    await loadExperiments();
  });
}

async function runCompareDemo(demoName, button) {
  const status = document.querySelector("#compare-status");
  const output = document.querySelector("#compare-output");
  await withButton(button, status, `${t("runningPrefix")} ${demoName}...`, async () => {
    output.innerHTML = "";
    const payload = await apiFetch(`/demo/compare/${demoName}`, { method: "POST" });
    const rows = payload.rows || [];
    output.innerHTML = [
      renderKeyValues({
        [t("successCount")]: payload.success_count,
        [t("failedCount")]: payload.failed_count,
        [t("primaryMetric")]: payload.primary_metric,
        feature_aware: firstDefined(rows, "feature_aware"),
        input_dim: firstDefined(rows, "input_dim"),
        target_dim: firstDefined(rows, "target_dim"),
        feature_dim: firstDefined(rows, "feature_dim"),
      }),
      renderTable(rows, [
        "rank",
        "model_name",
        "status",
        "primary_metric_value",
        "test_mae",
        "test_mse",
        "test_rmse",
        "feature_aware",
        "input_dim",
        "target_dim",
        "feature_dim",
      ]),
    ].join("");
    fillRunInputs(payload.experiment_name, payload.compare_run_id || payload.run_id);
    status.textContent = t("complete");
    await loadExperiments();
  });
}

async function loadExperiments() {
  const output = document.querySelector("#experiments-output");
  output.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    const payload = await apiFetch("/experiments");
    const rows = payload.experiments || [];
    state.experiments = rows;
    output.innerHTML = renderTable(rows, [
      "experiment_name",
      "run_id",
      "run_type",
      "status",
      "created_at",
      "success_count",
      "failed_count",
    ]);
  } catch (errorValue) {
    output.innerHTML = renderError(errorValue.message);
  }
}

async function loadJobs() {
  const output = document.querySelector("#jobs-output");
  output.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    const payload = await apiFetch("/jobs");
    output.innerHTML = renderTable(payload.jobs || [], [
      "job_id",
      "status",
      "kind",
      "created_at",
      "run_id",
      "error",
    ]);
  } catch (errorValue) {
    output.innerHTML = renderError(errorValue.message);
  }
}

async function loadJobDetail(kind) {
  const jobId = document.querySelector("#job-id").value.trim();
  const output = document.querySelector("#jobs-output");
  if (!jobId) {
    output.innerHTML = renderError(t("jobIdRequired"));
    return;
  }
  const suffix = kind === "job" ? "" : `/${kind}`;
  output.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    const payload = await apiFetch(`/jobs/${encodeURIComponent(jobId)}${suffix}`);
    output.innerHTML = renderJson(payload);
  } catch (errorValue) {
    output.innerHTML = renderError(errorValue.message);
  }
}

async function loadRunData(kind) {
  const experimentName = document.querySelector("#artifact-experiment").value.trim();
  const runId = document.querySelector("#artifact-run").value.trim() || "latest";
  const output = document.querySelector("#artifact-output");
  if (!experimentName) {
    output.innerHTML = renderError(t("experimentNameRequired"));
    return;
  }
  output.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    const base = `/experiments/${encodeURIComponent(experimentName)}/${encodeURIComponent(runId)}`;
    const payload = await apiFetch(`${base}/${kind}`);
    if (kind === "leaderboard") {
      output.innerHTML = renderTable(payload, [
        "rank",
        "model_name",
        "status",
        "primary_metric_value",
        "test_mae",
        "test_mse",
        "test_rmse",
        "feature_aware",
        "input_dim",
        "target_dim",
        "feature_dim",
      ]);
      return;
    }
    if (kind === "artifacts") {
      output.innerHTML = renderTable(payload.artifacts || [], [
        "name",
        "kind",
        "path",
        "description",
      ]);
      return;
    }
    output.innerHTML = renderJson(payload);
  } catch (errorValue) {
    output.innerHTML = renderError(errorValue.message);
  }
}

async function withButton(button, status, loadingText, callback) {
  const buttons = document.querySelectorAll("button");
  buttons.forEach((item) => {
    item.disabled = true;
  });
  status.textContent = loadingText;
  try {
    await callback();
  } catch (errorValue) {
    status.textContent = t("error");
    const target = button.dataset.trainDemo ? "#train-output" : "#compare-output";
    document.querySelector(target).innerHTML = renderError(errorValue.message);
  } finally {
    buttons.forEach((item) => {
      item.disabled = false;
    });
  }
}

function fillRunInputs(experimentName, runId) {
  if (experimentName) {
    document.querySelector("#artifact-experiment").value = experimentName;
  }
  if (runId) {
    document.querySelector("#artifact-run").value = runId;
  }
}

function getInitialLanguage() {
  try {
    const savedLanguage = window.localStorage.getItem("dashboardLanguage");
    if (savedLanguage === "en" || savedLanguage === "zh") {
      return savedLanguage;
    }
  } catch {
    return "zh";
  }
  return "zh";
}

function toggleLanguage() {
  state.language = state.language === "zh" ? "en" : "zh";
  try {
    window.localStorage.setItem("dashboardLanguage", state.language);
  } catch {
    // Ignore localStorage failures in restricted browser contexts.
  }
  applyLanguage();
  loadOverview();
  loadExperiments();
  loadJobs();
}

function applyLanguage() {
  document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
  document.title = t("documentTitle");
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelector("#language-toggle").textContent = t("languageToggle");
  document.querySelectorAll("[data-train-demo]").forEach((button) => {
    button.textContent = `${t("runPrefix")} ${button.dataset.trainDemo}`;
  });
  document.querySelectorAll("[data-compare-demo]").forEach((button) => {
    button.textContent = `${t("runPrefix")} ${button.dataset.compareDemo}`;
  });
}

function t(key) {
  return translations[state.language][key] || translations.en[key] || key;
}

function metric(label, value) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(
    formatValue(value),
  )}</strong></div>`;
}

function renderKeyValues(values) {
  return `<div class="summary">${Object.entries(values)
    .map(
      ([key, value]) =>
        `<div class="kv"><span class="key">${escapeHtml(key)}</span><span class="value">${escapeHtml(
          formatValue(value),
        )}</span></div>`,
    )
    .join("")}</div>`;
}

function renderTable(rows, columns) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return `<p class="muted">${escapeHtml(t("noRows"))}</p>`;
  }
  const header = columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
  const body = rows
    .map(
      (row) =>
        `<tr>${columns
          .map((column) => `<td>${escapeHtml(formatValue(row[column]))}</td>`)
          .join("")}</tr>`,
    )
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderJson(value) {
  return `<pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
}

function renderError(message) {
  return `<div class="error">${escapeHtml(message)}</div>`;
}

function firstDefined(rows, key) {
  const row = rows.find((item) => item[key] !== undefined && item[key] !== null);
  return row ? row[key] : null;
}

function formatValue(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
