const state = {
  experiments: [],
};

const jsonHeaders = { Accept: "application/json" };

document.addEventListener("DOMContentLoaded", () => {
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
  status.textContent = "Loading...";
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
      metric("Backend status", health.status || "unknown"),
      metric("Version", health.version || "unknown"),
      metric("Dataset count", datasetCount),
      metric("Model names", modelNames.join(", ") || "None"),
      metric("Experiment count", experimentRows.length),
    ].join("");
    status.textContent = "Ready";
  } catch (errorValue) {
    status.textContent = "Error";
    grid.innerHTML = "";
    error.textContent = errorValue.message;
  }
}

async function runTrainDemo(demoName, button) {
  const status = document.querySelector("#train-status");
  const output = document.querySelector("#train-output");
  await withButton(button, status, `Running ${demoName}...`, async () => {
    output.innerHTML = "";
    const payload = await apiFetch(`/demo/train/${demoName}`, { method: "POST" });
    output.innerHTML = [
      renderKeyValues({
        "experiment name": payload.experiment_name,
        run_id: payload.run_id,
        "checkpoint path": payload.checkpoint_path,
      }),
      "<h3>test_metrics.original</h3>",
      renderJson(payload.test_metrics ? payload.test_metrics.original : null),
      "<h3>data_metadata</h3>",
      renderJson(payload.data_metadata || null),
    ].join("");
    fillRunInputs(payload.experiment_name, payload.run_id);
    status.textContent = "Complete";
    await loadExperiments();
  });
}

async function runCompareDemo(demoName, button) {
  const status = document.querySelector("#compare-status");
  const output = document.querySelector("#compare-output");
  await withButton(button, status, `Running ${demoName}...`, async () => {
    output.innerHTML = "";
    const payload = await apiFetch(`/demo/compare/${demoName}`, { method: "POST" });
    const rows = payload.rows || [];
    output.innerHTML = [
      renderKeyValues({
        success_count: payload.success_count,
        failed_count: payload.failed_count,
        primary_metric: payload.primary_metric,
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
    status.textContent = "Complete";
    await loadExperiments();
  });
}

async function loadExperiments() {
  const output = document.querySelector("#experiments-output");
  output.innerHTML = '<p class="muted">Loading...</p>';
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
  output.innerHTML = '<p class="muted">Loading...</p>';
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
    output.innerHTML = renderError("job_id is required");
    return;
  }
  const suffix = kind === "job" ? "" : `/${kind}`;
  output.innerHTML = '<p class="muted">Loading...</p>';
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
    output.innerHTML = renderError("experiment_name is required");
    return;
  }
  output.innerHTML = '<p class="muted">Loading...</p>';
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
    status.textContent = "Error";
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
    return '<p class="muted">No rows.</p>';
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
