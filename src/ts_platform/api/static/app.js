const jsonHeaders = { Accept: "application/json" };
const postJsonHeaders = { ...jsonHeaders, "Content-Type": "application/json" };
const cudaUnavailableMessage = "CUDA is not available on this backend.";

const fallbackModels = [
  "naive",
  "moving_average",
  "seasonal_naive",
  "linear",
  "mlp",
  "nbeats",
  "rnn",
  "gru",
  "lstm",
  "tcn",
  "transformer",
];

const defaultModelParams = {
  gru: { hidden_size: 8 },
  linear: {},
  lstm: { hidden_size: 8 },
  mlp: { hidden_sizes: [8] },
  moving_average: { window_size: 2 },
  nbeats: { hidden_size: 16, num_blocks: 2, num_layers: 2 },
  naive: {},
  rnn: { hidden_size: 8 },
  seasonal_naive: { season_length: 2 },
  tcn: { hidden_channels: 8, num_layers: 2 },
  transformer: { d_model: 16, num_heads: 4, num_layers: 1, dim_feedforward: 32 },
};

const defaultCompareSelections = new Set(["naive", "moving_average", "linear"]);

const monitorPalette = [
  "#2563eb",
  "#059669",
  "#d97706",
  "#7c3aed",
  "#dc2626",
  "#0891b2",
  "#be185d",
  "#4d7c0f",
];

const dashboardPages = new Set(["overview", "datasets", "results", "custom", "jobs"]);

const fallbackDemoConfigs = {
  train: [
    "simple_forecast",
    "csv_forecast",
    "csv_feature_forecast",
    "appliances_energy_half_hour_demo",
  ],
  compare: ["compare_forecast", "compare_model_zoo", "compare_feature_forecast"],
};

const halfHourDemoName = "appliances_energy_half_hour_demo";

const state = {
  experiments: [],
  filteredExperiments: [],
  selectedKey: null,
  selectedRun: null,
  detailTab: "overview",
  page: getInitialPage(),
  language: getInitialLanguage(),
  overview: {
    health: null,
    datasets: null,
    models: null,
  },
  datasetCatalog: {
    allRows: [],
    rows: [],
    selectedKey: "",
    customRows: loadUserDatasets(),
    filters: {
      search: "",
      domain: "all",
    },
  },
  demoConfigs: fallbackDemoConfigs,
  favorites: new Set(loadFavorites()),
  custom: {
    mode: "train",
    availableModels: fallbackModels,
    lastTrainModel: "linear",
  },
  monitor: {
    smoothing: 0.35,
    metric: "all",
  },
  jobProgressTimer: null,
  filters: {
    search: "",
    type: "all",
    status: "all",
    sort: "created_desc",
    favoritesOnly: false,
  },
};

const translations = {
  en: {
    appTitle: "TS Platform Experiment Center",
    appSubtitle: "Manage run outputs, compare model performance, and inspect artifacts",
    eyebrow: "Local Experiment Workspace",
    artifactPreview: "Artifact Preview",
    artifactDownload: "Download",
    artifacts: "Artifacts",
    artifactsTitle: "Artifacts",
    backendStatus: "Backend",
    best: "Best",
    bestMetric: "Best metric",
    bestTrainMae: "Best train MAE",
    checkpointBlocked: "Checkpoint download is blocked by the local safety policy.",
    chooseCsvFile: "Choose CSV File",
    csvUploadComplete: "CSV file selected and copied into the local upload directory.",
    compareDemoComplete: "Compare complete",
    compareRuns: "Compare runs",
    compareTitle: "Compare Demo",
    completedRuns: "Completed",
    batchSize: "Batch size",
    checkpointEvery: "Checkpoint",
    addDatasetTitle: "Add Local Dataset",
    clearUserDatasets: "Clear custom",
    compareModelsRequired: "Select at least two models for compare mode.",
    compareParams: "Compare params JSON",
    configPreview: "Config Preview",
    csvData: "CSV File",
    csvPath: "CSV path",
    csvPathRequired: "CSV path is required.",
    customCaption: "Configure data, model, training, and evaluation parameters",
    customCompareComplete: "Custom compare complete",
    customConfigInvalid: "Config is not ready",
    customModeCompare: "Model Compare",
    customModeTrain: "Single Model Train",
    customSubmitCompare: "Run Custom Compare",
    customSubmitTrain: "Run Custom Train",
    customTitle: "Custom Experiment",
    customTrainComplete: "Custom training complete",
    dataMetadata: "Data Metadata",
    dataSource: "Data Source",
    delta: "Delta",
    datasetCatalogCaption: "Public dataset metadata and local custom CSV entries with source attribution",
    datasetCatalogTitle: "Dataset Catalog",
    datasetCount: "Datasets",
    datasetDescription: "Description",
    datasetDomain: "Domain",
    datasetName: "Dataset name",
    datasetSaved: "Dataset saved",
    datasetSource: "Source",
    datasetSources: "Dataset Sources",
    datasetDomainFilter: "Domain",
    datasetSearch: "Search datasets",
    datasetTemplate: "Dataset Template",
    datasetTemplateManual: "Manual / custom CSV",
    deleteDataset: "Delete",
    datasetDeleted: "Dataset deleted",
    datasetProfile: "Profile",
    datasetProfileTitle: "Dataset Profile",
    device: "Device",
    documentTitle: "TS Platform Experiment Center",
    emptyBody: "The local runs directory is listed on the left.",
    emptyTitle: "Select an experiment run",
    epochs: "Epochs",
    error: "Error",
    evaluationSettings: "Evaluation Settings",
    experimentName: "Experiment name",
    experimentNameInvalid: "Experiment name must use 1-80 letters, digits, underscore, dot, or hyphen.",
    exportReport: "Export Report",
    experimentsLoaded: "Experiments loaded",
    failedCount: "failed_count",
    favorite: "Favorite",
    favoritesOnly: "Favorites only",
    featureCols: "Feature columns",
    featureCount: "Feature count",
    forecastPreview: "Forecast Preview",
    frequency: "Frequency",
    filterAll: "All",
    filterCompare: "Compare",
    filterTrain: "Train",
    idle: "Idle",
    includeScaled: "Scaled metrics",
    inputLen: "Input length",
    jobDemoConfig: "Demo config",
    jobDemoKind: "Job type",
    jobId: "job_id",
    jobLaunchTitle: "Async Demo Jobs",
    halfHourDemo: "Half-Hour Monitor Demo",
    jobSubmitted: "Job submitted",
    jobType: "job_type",
    jobsTitle: "Jobs",
    languageToggle: "中文",
    learningRate: "Learning rate",
    leaderboard: "Leaderboard",
    libraryTitle: "Run Library",
    loadArtifacts: "Load Artifacts",
    loadJob: "Load Job",
    loadJobLogs: "Load Job Logs",
    loadJobResult: "Load Job Result",
    loading: "Loading...",
    loss: "Loss",
    latest: "Latest",
    metricsRequired: "Select at least one evaluation metric.",
    metricAll: "All metrics",
    metricFilter: "Metric",
    missingPolicy: "Missing values",
    model: "Model",
    modelCount: "Models",
    modelParams: "Model params JSON",
    noiseStd: "Noise",
    noArtifacts: "No artifacts.",
    noChartData: "No chart data available.",
    noDatasetPath: "External dataset: download it first, then enter the local CSV path.",
    noRows: "No rows.",
    operationsCaption: "Generate new results or inspect the local job queue",
    operationsTitle: "Runs and Jobs",
    optimizer: "Optimizer",
    outputLen: "Forecast length",
    overviewCaption: "Local experiment assets under the runs directory",
    overviewTitle: "Run Overview",
    pageCustom: "Custom",
    pageDatasets: "Datasets",
    pageJobs: "Jobs",
    pageOverview: "Overview",
    pageResults: "Results",
    primaryMetric: "primary_metric",
    primaryMetricRequired: "Primary metric must be one of the selected metrics.",
    refreshAll: "Refresh All",
    refreshJobs: "Refresh Jobs",
    refreshPreview: "Refresh Preview",
    reloadRun: "Reload",
    resultsCaption: "Filter, favorite, inspect, and visualize run outputs",
    resultsTitle: "Experiment Result Management",
    runPrefix: "Run",
    runningPrefix: "Running",
    saveDataset: "Save and Use",
    selectDataset: "Select dataset",
    scaler: "Scaler",
    searchLabel: "Search",
    sequenceAndSplit: "Window and Split",
    seriesLength: "Series length",
    sortLabel: "Sort",
    sortByTime: "Sort by time",
    sortMetric: "Lowest MAE",
    sortName: "Name",
    sortNewest: "Newest",
    sortOldest: "Oldest",
    splitRatioInvalid: "Train, validation, and test ratios must sum to 1.",
    statusComplete: "Complete",
    statusIncomplete: "Incomplete",
    statusLabel: "Status",
    successCount: "success_count",
    submitAllDemoJobs: "Submit all in type",
    submitDemoJob: "Submit Job",
    smoothing: "Smoothing",
    syntheticData: "Synthetic Data",
    tabArtifacts: "Artifacts",
    tabOverview: "Overview",
    tabRaw: "Raw JSON",
    targetCols: "Target columns",
    targetColsRequired: "Target columns are required.",
    testMetrics: "Test Metrics",
    testRatio: "Test ratio",
    timestampCol: "Timestamp column",
    trainDemoComplete: "Training complete",
    trainRatio: "Train ratio",
    trainRuns: "Train runs",
    trainTitle: "Train Demo",
    trainingSettings: "Training Settings",
    trainingCurve: "Training Curve",
    trainingMonitor: "Training Monitor",
    totalRuns: "Total runs",
    unfavorite: "Unfavorite",
    useDataset: "Use",
    usedDataset: "Dataset loaded into the custom experiment form.",
    userDatasetNameInvalid: "Dataset name must use 1-80 letters, digits, underscore, dot, or hyphen.",
    userDatasetPathRequired: "Local CSV path is required.",
    valRatio: "Validation ratio",
    version: "Version",
  },
  zh: {
    appTitle: "时间序列平台实验中心",
    appSubtitle: "管理运行结果、对比模型表现、检查训练产物",
    eyebrow: "本地实验工作台",
    artifactPreview: "产物预览",
    artifactDownload: "下载",
    artifacts: "产物",
    artifactsTitle: "产物",
    backendStatus: "后端",
    best: "最佳",
    bestMetric: "最佳指标",
    bestTrainMae: "最佳训练 MAE",
    checkpointBlocked: "checkpoint 下载已被本地安全策略禁用。",
    chooseCsvFile: "选择 CSV 文件",
    csvUploadComplete: "CSV 文件已选择并复制到本地上传目录。",
    compareDemoComplete: "对比完成",
    compareRuns: "对比运行",
    compareTitle: "对比演示",
    completedRuns: "已完成",
    batchSize: "Batch size",
    checkpointEvery: "Checkpoint",
    addDatasetTitle: "补充本地数据集",
    clearUserDatasets: "清空自定义",
    compareModelsRequired: "对比模式至少选择两个模型。",
    compareParams: "对比参数 JSON",
    configPreview: "配置预览",
    csvData: "CSV 文件",
    csvPath: "CSV 路径",
    csvPathRequired: "CSV 路径不能为空。",
    customCaption: "配置数据、模型、训练和评估参数",
    customCompareComplete: "自定义对比完成",
    customConfigInvalid: "配置尚未就绪",
    customModeCompare: "模型对比",
    customModeTrain: "单模型训练",
    customSubmitCompare: "运行自定义对比",
    customSubmitTrain: "运行自定义训练",
    customTitle: "自定义实验",
    customTrainComplete: "自定义训练完成",
    dataMetadata: "数据元信息",
    dataSource: "数据源",
    delta: "变化",
    datasetCatalogCaption: "公开数据集目录和本地自定义 CSV，均保留来源信息",
    datasetCatalogTitle: "数据集库",
    datasetCount: "数据集",
    datasetDescription: "描述",
    datasetDomain: "领域",
    datasetName: "数据集名",
    datasetSaved: "数据集已保存",
    datasetSource: "来源",
    datasetSources: "数据集来源",
    datasetDomainFilter: "领域筛选",
    datasetSearch: "搜索数据集",
    datasetTemplate: "数据集模板",
    datasetTemplateManual: "手动填写 / 自定义 CSV",
    deleteDataset: "删除",
    datasetDeleted: "数据集已删除",
    datasetProfile: "体检",
    datasetProfileTitle: "数据集体检",
    device: "设备",
    documentTitle: "时间序列平台实验中心",
    emptyBody: "左侧结果库会显示本地 runs 目录中的训练和对比运行。",
    emptyTitle: "选择一个实验 run",
    epochs: "Epochs",
    error: "错误",
    evaluationSettings: "评估设置",
    experimentName: "实验名",
    experimentNameInvalid: "实验名只能包含 1-80 位字母、数字、下划线、点或短横线。",
    exportReport: "导出报告",
    experimentsLoaded: "实验已加载",
    failedCount: "失败数量",
    favorite: "收藏",
    favoritesOnly: "只看收藏",
    featureCols: "特征列",
    featureCount: "特征数",
    forecastPreview: "预测预览",
    frequency: "频率",
    filterAll: "全部",
    filterCompare: "对比",
    filterTrain: "训练",
    idle: "空闲",
    includeScaled: "Scaled metrics",
    inputLen: "输入长度",
    jobDemoConfig: "演示配置",
    jobDemoKind: "任务类型",
    jobId: "job_id",
    jobLaunchTitle: "异步演示任务",
    halfHourDemo: "一键半小时监控演示",
    jobSubmitted: "任务已提交",
    jobType: "任务类型",
    jobsTitle: "任务",
    languageToggle: "English",
    learningRate: "学习率",
    leaderboard: "排行榜",
    libraryTitle: "结果库",
    loadArtifacts: "加载产物",
    loadJob: "加载任务",
    loadJobLogs: "加载任务日志",
    loadJobResult: "加载任务结果",
    loading: "加载中...",
    loss: "Loss",
    latest: "最新",
    metricsRequired: "至少选择一个评估指标。",
    metricAll: "全部指标",
    metricFilter: "指标",
    missingPolicy: "缺失值",
    model: "模型",
    modelCount: "模型",
    modelParams: "模型参数 JSON",
    noiseStd: "噪声",
    noArtifacts: "暂无产物。",
    noChartData: "暂无可视化数据。",
    noDatasetPath: "外部数据集需先下载到本地，再填写 CSV 路径。",
    noRows: "暂无数据。",
    operationsCaption: "快速生成新结果，或检查本地任务队列",
    operationsTitle: "运行与任务",
    optimizer: "优化器",
    outputLen: "预测长度",
    overviewCaption: "当前 runs 目录中的实验资产状态",
    overviewTitle: "运行概览",
    pageCustom: "自定义实验",
    pageDatasets: "数据集",
    pageJobs: "任务",
    pageOverview: "概览",
    pageResults: "实验结果",
    primaryMetric: "主指标",
    primaryMetricRequired: "主指标必须包含在已选评估指标中。",
    refreshAll: "刷新全部",
    refreshJobs: "刷新任务",
    refreshPreview: "刷新预览",
    reloadRun: "重新加载",
    resultsCaption: "筛选、收藏、查看 run 详情和可视化结果",
    resultsTitle: "实验结果管理",
    runPrefix: "运行",
    runningPrefix: "正在运行",
    saveDataset: "保存并使用",
    selectDataset: "选择数据集",
    scaler: "Scaler",
    searchLabel: "搜索",
    sequenceAndSplit: "窗口与切分",
    seriesLength: "序列长度",
    sortLabel: "排序",
    sortByTime: "按时间排序",
    sortMetric: "MAE 最低",
    sortName: "名称",
    sortNewest: "最新",
    sortOldest: "最早",
    splitRatioInvalid: "训练、验证、测试占比之和必须等于 1。",
    statusComplete: "完成",
    statusIncomplete: "未完成",
    statusLabel: "状态",
    successCount: "成功数量",
    submitAllDemoJobs: "提交当前类别全部",
    submitDemoJob: "提交任务",
    smoothing: "平滑",
    syntheticData: "合成数据",
    tabArtifacts: "产物",
    tabOverview: "概览",
    tabRaw: "原始 JSON",
    targetCols: "目标列",
    targetColsRequired: "目标列不能为空。",
    testMetrics: "测试指标",
    testRatio: "测试占比",
    timestampCol: "时间列",
    trainDemoComplete: "训练完成",
    trainRatio: "训练占比",
    trainRuns: "训练运行",
    trainTitle: "训练演示",
    trainingSettings: "训练设置",
    trainingCurve: "训练曲线",
    trainingMonitor: "训练监控",
    totalRuns: "总运行数",
    unfavorite: "取消收藏",
    useDataset: "使用",
    usedDataset: "已填入自定义实验表单。",
    userDatasetNameInvalid: "数据集名只能包含 1-80 位字母、数字、下划线、点或短横线。",
    userDatasetPathRequired: "本地 CSV 路径不能为空。",
    valRatio: "验证占比",
    version: "版本",
  },
};

document.addEventListener("DOMContentLoaded", () => {
  bindControls();
  initializeCustomBuilder();
  applyLanguage();
  initializeDashboardPage();
  loadDashboard();
});

function bindControls() {
  document.querySelector("#language-toggle").addEventListener("click", toggleLanguage);
  document.querySelector("#refresh-all").addEventListener("click", loadDashboard);
  document.querySelector("#refresh-jobs").addEventListener("click", loadJobs);
  document.querySelectorAll("[data-page-nav]").forEach((button) => {
    button.addEventListener("click", () => {
      setDashboardPage(button.dataset.pageNav, { updateHash: true });
    });
  });
  window.addEventListener("hashchange", () => {
    setDashboardPage(pageFromHash(), { updateHash: false });
  });

  document.querySelector("#experiment-search").addEventListener("input", (event) => {
    state.filters.search = event.target.value.trim().toLowerCase();
    renderExperimentList();
  });
  document.querySelector("#status-filter").addEventListener("change", (event) => {
    state.filters.status = event.target.value;
    renderExperimentList();
  });
  document.querySelector("#sort-filter").addEventListener("change", (event) => {
    state.filters.sort = event.target.value;
    renderExperimentList();
  });
  document.querySelector("#favorite-filter").addEventListener("change", (event) => {
    state.filters.favoritesOnly = event.target.checked;
    renderExperimentList();
  });
  document.querySelectorAll("[data-filter-type]").forEach((button) => {
    button.addEventListener("click", () => {
      state.filters.type = button.dataset.filterType;
      document.querySelectorAll("[data-filter-type]").forEach((item) => {
        item.classList.toggle("active", item === button);
      });
      renderExperimentList();
    });
  });

  document.querySelector("#favorite-run").addEventListener("click", toggleSelectedFavorite);
  document.querySelector("#export-report-run").addEventListener("click", exportSelectedReport);
  document.querySelector("#reload-run").addEventListener("click", reloadSelectedRun);
  document.querySelector("#lookup-results").addEventListener("click", (event) => {
    loadRunLookupPayload(event, "results");
  });
  document.querySelector("#lookup-leaderboard").addEventListener("click", (event) => {
    loadRunLookupPayload(event, "leaderboard");
  });
  document.querySelector("#lookup-artifacts").addEventListener("click", (event) => {
    loadRunLookupPayload(event, "artifacts");
  });
  document.querySelector("#download-lookup-artifact").addEventListener("click", downloadRunLookupArtifact);
  document.querySelectorAll("[data-detail-tab]").forEach((button) => {
    button.addEventListener("click", () => switchDetailTab(button.dataset.detailTab));
  });

  document.querySelectorAll("[data-train-demo]").forEach((button) => {
    button.addEventListener("click", () => runTrainDemo(button.dataset.trainDemo, button));
  });
  document.querySelectorAll("[data-compare-demo]").forEach((button) => {
    button.addEventListener("click", () => runCompareDemo(button.dataset.compareDemo, button));
  });
  document.querySelector("#job-demo-kind").addEventListener("change", renderJobDemoOptions);
  document.querySelector("#submit-demo-job").addEventListener("click", (event) => {
    submitSelectedDemoJob(event.currentTarget);
  });
  document.querySelector("#start-half-hour-demo").addEventListener("click", (event) => {
    startHalfHourDemo(event.currentTarget);
  });
  document.querySelector("#submit-all-demo-jobs").addEventListener("click", (event) => {
    submitAllDemoJobs(event.currentTarget);
  });

  document.querySelectorAll("[data-custom-mode]").forEach((button) => {
    button.addEventListener("click", () => setCustomMode(button.dataset.customMode));
  });
  document.querySelector("#custom-form").addEventListener("submit", runCustomExperiment);
  document.querySelector("#custom-preview-refresh").addEventListener("click", renderCustomPreview);
  document.querySelector("#custom-data-source").addEventListener("change", () => {
    syncCustomDataSource();
    renderCustomPreview();
  });
  document.querySelector("#custom-dataset-template").addEventListener("change", (event) => {
    applyDatasetTemplate(event.target.value);
  });
  document.querySelector("#user-dataset-form").addEventListener("submit", saveUserDataset);
  document.querySelector("#clear-user-datasets").addEventListener("click", clearUserDatasets);
  document.querySelector("#dataset-search").addEventListener("input", (event) => {
    state.datasetCatalog.filters.search = event.target.value.trim().toLowerCase();
    renderDatasetCatalog(state.overview.datasets || { datasets: [] });
  });
  document.querySelector("#dataset-domain-filter").addEventListener("change", (event) => {
    state.datasetCatalog.filters.domain = event.target.value;
    renderDatasetCatalog(state.overview.datasets || { datasets: [] });
  });
  document.querySelector("#custom-pick-csv").addEventListener("click", () => {
    document.querySelector("#custom-csv-file").click();
  });
  document.querySelector("#user-pick-csv").addEventListener("click", () => {
    document.querySelector("#user-csv-file").click();
  });
  document.querySelector("#custom-csv-file").addEventListener("change", (event) => {
    handleCsvFileSelection(event, "custom");
  });
  document.querySelector("#user-csv-file").addEventListener("change", (event) => {
    handleCsvFileSelection(event, "user");
  });
  document.querySelector("#custom-train-model").addEventListener("change", syncTrainModelParams);
  document.querySelector("#custom-primary-metric").addEventListener("change", renderCustomPreview);
  document.querySelectorAll("#custom-form input, #custom-form select, #custom-form textarea").forEach((field) => {
    field.addEventListener("input", renderCustomPreview);
    field.addEventListener("change", renderCustomPreview);
  });

  document.querySelector("#load-job").addEventListener("click", () => loadJobDetail("job"));
  document
    .querySelector("#load-job-result")
    .addEventListener("click", () => loadJobDetail("result"));
  document.querySelector("#load-job-logs").addEventListener("click", () => loadJobDetail("logs"));
  document.querySelector("#load-job-progress").addEventListener("click", () => loadJobProgress());
  document.querySelector("#auto-job-progress").addEventListener("change", toggleAutoJobProgress);
  document.querySelector("#load-job-events").addEventListener("click", () => loadJobDetail("events"));
  document.querySelector("#load-job-attempts").addEventListener("click", () => loadJobDetail("attempts"));
  document.querySelector("#cancel-job").addEventListener("click", runCancelJob);
  document.querySelector("#retry-job").addEventListener("click", runRetryJob);
  document.querySelector("#list-stale-jobs").addEventListener("click", listStaleJobs);
  document.querySelector("#mark-stale-timeout").addEventListener("click", markStaleTimeout);
  document.querySelector("#worker-once").addEventListener("click", runWorkerOnce);
  document.querySelector("#worker-loop").addEventListener("click", runWorkerLoop);
  document.querySelector("#run-config-file").addEventListener("click", runConfigFile);
  document.querySelector("#prediction-pick-values").addEventListener("click", () => {
    document.querySelector("#prediction-values-file").click();
  });
  document.querySelector("#prediction-values-file").addEventListener("change", handlePredictionValuesFile);
  document.querySelector("#predict-selected-run").addEventListener("click", predictSelectedRun);
  document.querySelector("#predict-model-export-path").addEventListener("click", predictModelExportPath);
  document.querySelector("#list-registered-datasets-tool").addEventListener("click", listRegisteredDatasetsTool);
  document.querySelector("#list-models-tool").addEventListener("click", listModelsTool);
  document.querySelector("#profile-csv-tool").addEventListener("click", profileCsvTool);
  document.querySelector("#list-catalog-tool").addEventListener("click", listCatalogTool);
  document.querySelector("#profile-catalog-tool").addEventListener("click", profileCatalogTool);
  document.querySelector("#generate-catalog-config-tool").addEventListener("click", generateCatalogConfigTool);
}

function initializeDashboardPage() {
  setDashboardPage(state.page || "overview", { updateHash: false });
}

function setDashboardPage(pageName, options = {}) {
  const nextPage = dashboardPages.has(pageName) ? pageName : "overview";
  state.page = nextPage;
  document.querySelectorAll("[data-page-section]").forEach((section) => {
    section.hidden = section.dataset.pageSection !== nextPage;
  });
  document.querySelectorAll("[data-page-nav]").forEach((button) => {
    const isActive = button.dataset.pageNav === nextPage;
    button.classList.toggle("active", isActive);
    if (isActive) {
      button.setAttribute("aria-current", "page");
    } else {
      button.removeAttribute("aria-current");
    }
  });
  if (options.updateHash) {
    const hash = `#${nextPage}`;
    if (window.location.hash !== hash) {
      window.history.pushState(null, "", hash);
    }
  }
  window.scrollTo({ top: 0, behavior: options.smooth ? "smooth" : "auto" });
}

function pageFromHash() {
  const hashPage = window.location.hash.replace(/^#/, "");
  return dashboardPages.has(hashPage) ? hashPage : "overview";
}

async function loadDashboard() {
  setText("#overview-status", t("loading"));
  setText("#results-status", t("loading"));
  try {
    const [health, datasets, models, experimentsPayload, demoConfigs] = await Promise.all([
      apiFetch("/health"),
      apiFetch("/datasets"),
      apiFetch("/models"),
      apiFetch("/experiments"),
      apiFetch("/demo/configs"),
    ]);
    state.overview = { health, datasets, models };
    state.demoConfigs = normalizeDemoConfigs(demoConfigs);
    state.experiments = experimentsPayload.experiments || [];
    syncDeviceOptions(health);
    renderCustomModelOptions(models);
    renderDatasetCatalog(datasets);
    renderJobDemoOptions();
    renderOverview(health, datasets, models, state.experiments);
    renderExperimentList();
    setText("#overview-status", t("experimentsLoaded"));
    setText("#results-status", t("experimentsLoaded"));
    await ensureSelectedRun();
  } catch (error) {
    setText("#overview-status", t("error"));
    setText("#results-status", t("error"));
    document.querySelector("#overview-error").textContent = error.message;
  }
  loadJobs();
}

async function ensureSelectedRun() {
  if (state.selectedKey && state.filteredExperiments.some((row) => runKey(row) === state.selectedKey)) {
    await selectRun(state.selectedKey);
    return;
  }
  const firstComplete = state.filteredExperiments.find((row) => row.status === "complete");
  if (firstComplete) {
    await selectRun(runKey(firstComplete));
  } else {
    state.selectedKey = null;
    state.selectedRun = null;
    renderEmptyDetail();
  }
}

function renderOverview(health, datasets, models, rows) {
  const completeRows = rows.filter((row) => row.status === "complete");
  const trainRows = rows.filter((row) => row.run_type === "train");
  const compareRows = rows.filter((row) => row.run_type === "compare");
  const bestTrain = trainRows
    .map((row) => metricFromSummary(row))
    .filter((metric) => metric !== null)
    .sort((a, b) => a - b)[0];
  const datasetCount = countMergedDatasets(datasets);
  const modelCount = (models.models || []).length;
  const deviceDetail = health.cuda_available
    ? `CUDA ${health.cuda_device_count || 1}`
    : "CPU only";
  const grid = document.querySelector("#overview-grid");
  grid.innerHTML = [
    metric(
      t("backendStatus"),
      health.status || "-",
      `${t("version")} ${health.version || "-"} | ${deviceDetail}`,
    ),
    metric(t("totalRuns"), rows.length, `${t("completedRuns")} ${completeRows.length}`),
    metric(t("trainRuns"), trainRows.length, t("trainTitle")),
    metric(t("compareRuns"), compareRows.length, t("compareTitle")),
    metric(t("datasetCount"), datasetCount, `${t("modelCount")} ${modelCount}`),
    metric(t("bestTrainMae"), bestTrain === undefined ? "-" : formatNumber(bestTrain), "MAE"),
  ].join("");
}

function syncDeviceOptions(health = {}) {
  const select = document.querySelector("#custom-device");
  const cudaOption = document.querySelector("#custom-device option[value='cuda']");
  if (!select || !cudaOption) {
    return;
  }
  const cudaAvailable = Boolean(health.cuda_available);
  cudaOption.disabled = !cudaAvailable;
  cudaOption.textContent = cudaAvailable ? "cuda" : "cuda (unavailable)";
  cudaOption.title = cudaAvailable
    ? `${health.cuda_device_count || 1} CUDA device(s) available`
    : cudaUnavailableMessage;
  select.title = cudaAvailable ? "" : cudaUnavailableMessage;
  if (!cudaAvailable && select.value === "cuda") {
    select.value = "cpu";
  }
}

function renderExperimentList() {
  const list = document.querySelector("#experiment-list");
  const rows = getFilteredExperiments();
  state.filteredExperiments = rows;
  document.querySelector("#library-count").textContent = String(rows.length);
  if (!rows.length) {
    list.innerHTML = `<div class="empty-state"><p>${escapeHtml(t("noRows"))}</p></div>`;
    return;
  }
  list.innerHTML = rows
    .map((row) => {
      const key = runKey(row);
      const isActive = key === state.selectedKey;
      const isFavorite = state.favorites.has(key);
      const typeClass = row.run_type === "train" ? "type-train" : row.run_type === "compare" ? "type-compare" : "type-unknown";
      const metricValue = metricFromSummary(row);
      return `<button class="run-item ${isActive ? "active" : ""}" type="button" data-run-key="${escapeHtml(key)}">
        <div class="run-item-head">
          <span class="run-name">${escapeHtml(row.experiment_name || "-")}</span>
          <span class="star" aria-label="favorite">${isFavorite ? "★" : "☆"}</span>
        </div>
        <div class="run-meta">
          <span class="type-pill ${typeClass}">${escapeHtml(formatRunType(row.run_type))}</span>
          <span>${escapeHtml(row.status || "-")}</span>
          <span>${escapeHtml(formatDate(row.created_at))}</span>
        </div>
        <div class="run-meta">
          <span>${escapeHtml(row.run_id || "-")}</span>
          <span>MAE ${escapeHtml(metricValue === null ? "-" : formatNumber(metricValue))}</span>
        </div>
      </button>`;
    })
    .join("");
  list.querySelectorAll("[data-run-key]").forEach((button) => {
    button.addEventListener("click", () => selectRun(button.dataset.runKey));
  });
}

function getFilteredExperiments() {
  const query = state.filters.search;
  const rows = state.experiments.filter((row) => {
    const key = runKey(row);
    const haystack = [row.experiment_name, row.run_id, row.run_type, row.status]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    if (query && !haystack.includes(query)) {
      return false;
    }
    if (state.filters.type !== "all" && row.run_type !== state.filters.type) {
      return false;
    }
    if (state.filters.status !== "all" && row.status !== state.filters.status) {
      return false;
    }
    if (state.filters.favoritesOnly && !state.favorites.has(key)) {
      return false;
    }
    return true;
  });
  return rows.sort((a, b) => {
    if (state.filters.sort === "created_asc") {
      return dateValue(a.created_at) - dateValue(b.created_at);
    }
    if (state.filters.sort === "name_asc") {
      return String(a.experiment_name || "").localeCompare(String(b.experiment_name || ""));
    }
    if (state.filters.sort === "metric_asc") {
      return (metricFromSummary(a) ?? Number.POSITIVE_INFINITY) - (metricFromSummary(b) ?? Number.POSITIVE_INFINITY);
    }
    return dateValue(b.created_at) - dateValue(a.created_at);
  });
}

async function selectRun(key) {
  const summary = state.experiments.find((row) => runKey(row) === key);
  if (!summary) {
    return;
  }
  state.selectedKey = key;
  renderExperimentList();
  if (summary.status !== "complete" || !summary.run_id) {
    state.selectedRun = { summary, results: null, artifacts: null, leaderboard: [] };
    renderDetail();
    return;
  }
  setText("#results-status", t("loading"));
  try {
    const base = `/experiments/${encodeURIComponent(summary.experiment_name)}/${encodeURIComponent(summary.run_id)}`;
    const [results, artifacts, leaderboard] = await Promise.all([
      apiFetch(`${base}/results`),
      apiFetch(`${base}/artifacts`).catch(() => null),
      summary.run_type === "compare" ? apiFetch(`${base}/leaderboard`) : Promise.resolve([]),
    ]);
    state.selectedRun = { summary, results, artifacts, leaderboard };
    renderDetail();
    setText("#results-status", t("idle"));
  } catch (error) {
    state.selectedRun = { summary, results: null, artifacts: null, leaderboard: [], error };
    renderDetail();
    setText("#results-status", t("error"));
  }
}

function renderEmptyDetail() {
  document.querySelector("#detail-empty").hidden = false;
  document.querySelector("#detail-content").hidden = true;
}

function renderDetail() {
  const selected = state.selectedRun;
  if (!selected) {
    renderEmptyDetail();
    return;
  }
  document.querySelector("#detail-empty").hidden = true;
  document.querySelector("#detail-content").hidden = false;
  const { summary, results, leaderboard, artifacts, error } = selected;
  const key = runKey(summary);
  setText("#detail-kind", `${formatRunType(summary.run_type)} · ${summary.status || "-"}`);
  setText("#detail-title", summary.experiment_name || "-");
  setText("#detail-subtitle", `${summary.run_id || "-"} · ${formatDate(summary.created_at)}`);
  syncRunLookupFields(summary);
  document.querySelector("#favorite-run").textContent = state.favorites.has(key)
    ? t("unfavorite")
    : t("favorite");
  document.querySelector("#export-report-run").disabled = !results || Boolean(error);

  if (error) {
    document.querySelector("#detail-metrics").innerHTML = "";
    document.querySelector("#detail-overview").innerHTML = renderError(error.message);
    document.querySelector("#detail-artifacts").innerHTML = "";
    document.querySelector("#detail-raw").innerHTML = "";
    return;
  }

  document.querySelector("#detail-metrics").innerHTML = renderDetailMetrics(summary, results, leaderboard);
  document.querySelector("#detail-overview").innerHTML =
    summary.run_type === "compare"
      ? renderCompareOverview(results, leaderboard)
      : renderTrainOverview(results);
  document.querySelector("#detail-artifacts").innerHTML = renderArtifactsPanel(summary, artifacts);
  document.querySelector("#detail-raw").innerHTML = renderRawPanel(results, leaderboard, artifacts);
  syncPredictionPath(results);
  bindTrainingMonitorControls();
  bindArtifactButtons();
  switchDetailTab(state.detailTab);
}

function syncPredictionPath(results) {
  const input = document.querySelector("#prediction-model-export-path");
  if (!input || input.value || !results?.model_export_path) {
    return;
  }
  input.value = results.model_export_path;
}

function renderDetailMetrics(summary, results, leaderboard) {
  if (!results) {
    return [
      metric(t("completedRuns"), summary.status || "-"),
      metric(t("totalRuns"), summary.run_type || "-"),
    ].join("");
  }
  if (summary.run_type === "compare") {
    const rows = leaderboard || results.rows || [];
    const metricName = results.primary_metric || "mae";
    const best = rows
      .map((row) => asNumber(row.primary_metric_value ?? row[`test_${metricName}`]))
      .filter((value) => value !== null)
      .sort((a, b) => a - b)[0];
    return [
      metric(t("successCount"), results.success_count ?? "-", t("compareRuns")),
      metric(t("failedCount"), results.failed_count ?? "-", t("statusLabel")),
      metric(t("primaryMetric"), metricName, t("leaderboard")),
      metric(t("bestMetric"), best === undefined ? "-" : formatNumber(best), metricName),
    ].join("");
  }
  const original = results.test_metrics?.original || {};
  const history = Array.isArray(results.history) ? results.history : [];
  return [
    metric("MAE", formatNumber(original.mae), t("testMetrics")),
    metric("RMSE", formatNumber(original.rmse), t("testMetrics")),
    metric("WAPE", formatNumber(original.wape), t("testMetrics")),
    metric("Epochs", history.length, t("trainingCurve")),
  ].join("");
}

function renderTrainOverview(results) {
  if (!results) {
    return `<p class="muted">${escapeHtml(t("noRows"))}</p>`;
  }
  const original = results.test_metrics?.original || {};
  return `<div class="visual-grid">
    ${renderTrainingMonitor(results)}
    <div class="visual-panel wide">
      <div class="visual-title">
        <h3>${escapeHtml(t("forecastPreview"))}</h3>
        <div class="legend"><span><i class="accent"></i>actual</span><span><i class="blue"></i>predicted</span></div>
      </div>
      ${renderForecastChart(results.forecast_samples)}
    </div>
    <div class="visual-panel">
      <div class="visual-title"><h3>${escapeHtml(t("testMetrics"))}</h3></div>
      ${renderMetricBars(original)}
    </div>
    <div class="visual-panel">
      <div class="visual-title"><h3>${escapeHtml(t("dataMetadata"))}</h3></div>
      ${renderKeyValues(results.data_metadata || {})}
    </div>
  </div>`;
}

function renderCompareOverview(results, leaderboard) {
  if (!results) {
    return `<p class="muted">${escapeHtml(t("noRows"))}</p>`;
  }
  const rows = leaderboard || results.rows || [];
  const metricName = results.primary_metric || "mae";
  return `<div class="visual-grid">
    <div class="visual-panel wide">
      <div class="visual-title">
        <h3>${escapeHtml(t("leaderboard"))}</h3>
        <span class="chip">${escapeHtml(metricName)}</span>
      </div>
      ${renderLeaderboardChart(rows, metricName)}
    </div>
    <div class="visual-panel wide">
      ${renderTable(rows, [
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
      ])}
    </div>
  </div>`;
}

function renderArtifactsPanel(summary, artifacts) {
  const rows = artifacts?.artifacts || [];
  if (!rows.length) {
    return `<p class="muted">${escapeHtml(t("noArtifacts"))}</p>`;
  }
  const body = rows
    .map((row) => {
      const artifactUrl = summary?.run_id
        ? `/experiments/${encodeURIComponent(summary.experiment_name)}/${encodeURIComponent(summary.run_id)}/artifacts/${encodeURIComponent(row.name)}`
        : "";
      const action =
        row.kind === "checkpoint"
          ? `<span class="artifact-disabled">${escapeHtml(t("checkpointBlocked"))}</span>`
          : row.kind === "model"
            ? `<div class="table-actions">
                <a class="table-action-link" href="${escapeHtml(artifactUrl)}" download>${escapeHtml(t("artifactDownload"))}</a>
              </div>`
          : `<div class="table-actions">
              <button class="table-action ghost" type="button" data-artifact-name="${escapeHtml(row.name)}">${escapeHtml(t("artifactPreview"))}</button>
              <a class="table-action-link" href="${escapeHtml(artifactUrl)}" download>${escapeHtml(t("artifactDownload"))}</a>
            </div>`;
      return `<tr>
        <td>${escapeHtml(row.name)}</td>
        <td>${escapeHtml(row.kind)}</td>
        <td>${escapeHtml(row.path)}</td>
        <td>${escapeHtml(row.description)}</td>
        <td>${action}</td>
      </tr>`;
    })
    .join("");
  return `<div class="table-wrap"><table><thead><tr><th>name</th><th>kind</th><th>path</th><th>description</th><th>${escapeHtml(t("artifacts"))}</th></tr></thead><tbody>${body}</tbody></table></div>
    <div id="artifact-preview" class="artifact-preview"></div>`;
}

function renderRawPanel(results, leaderboard, artifacts) {
  return [
    `<h3>results.json</h3>`,
    renderJson(results || null),
    `<h3>leaderboard</h3>`,
    renderJson(leaderboard || []),
    `<h3>artifacts.json</h3>`,
    renderJson(artifacts || null),
  ].join("");
}

function bindArtifactButtons() {
  document.querySelectorAll("[data-artifact-name]").forEach((button) => {
    button.addEventListener("click", () => previewArtifact(button.dataset.artifactName));
  });
}

function bindTrainingMonitorControls() {
  const metricSelect = document.querySelector("#monitor-metric");
  if (metricSelect) {
    metricSelect.addEventListener("change", (event) => {
      state.monitor.metric = event.target.value;
      refreshTrainingMonitor();
    });
  }
  const smoothingInput = document.querySelector("#monitor-smoothing");
  if (smoothingInput) {
    smoothingInput.addEventListener("input", (event) => {
      state.monitor.smoothing = Number(event.target.value) / 100;
      refreshTrainingMonitor();
    });
  }
}

function refreshTrainingMonitor() {
  const container = document.querySelector("#training-monitor");
  const results = state.selectedRun?.results;
  if (!container || !results) {
    return;
  }
  container.outerHTML = renderTrainingMonitor(results);
  bindTrainingMonitorControls();
}

async function previewArtifact(artifactName) {
  const selected = state.selectedRun;
  const preview = document.querySelector("#artifact-preview");
  if (!selected || !selected.summary.run_id) {
    return;
  }
  preview.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    const { experiment_name: experimentName, run_id: runId } = selected.summary;
    const response = await fetch(
      `/experiments/${encodeURIComponent(experimentName)}/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactName)}`,
      { headers: jsonHeaders },
    );
    const text = await response.text();
    if (!response.ok) {
      throw new Error(`${response.status} ${text}`);
    }
    let parsed = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
    preview.innerHTML = `<h3>${escapeHtml(artifactName)}</h3>${renderArtifactPayload(parsed)}`;
  } catch (error) {
    preview.innerHTML = renderError(error.message);
  }
}

function renderArtifactPayload(value) {
  if (Array.isArray(value) && value.every((item) => item && typeof item === "object" && !Array.isArray(item))) {
    const columns = Array.from(new Set(value.flatMap((item) => Object.keys(item)))).slice(0, 12);
    return renderTable(value, columns);
  }
  if (typeof value === "string") {
    return `<pre>${escapeHtml(value)}</pre>`;
  }
  return renderJson(value);
}

function renderDatasetProfile(profile) {
  const summary = {
    name: profile?.name,
    exists: profile?.exists,
    row_count: profile?.row_count,
    column_count: profile?.column_count,
    can_build_windows: profile?.can_build_windows,
    min_required_rows: profile?.min_required_rows,
    inferred_frequency: profile?.inferred_frequency,
    start_timestamp: profile?.start_timestamp,
    end_timestamp: profile?.end_timestamp,
  };
  const warnings = Array.isArray(profile?.warnings) && profile.warnings.length
    ? `<div class="error">${escapeHtml(profile.warnings.join("; "))}</div>`
    : "";
  const columns = Array.isArray(profile?.columns) ? profile.columns : [];
  const targets = Array.isArray(profile?.target_cols) ? profile.target_cols : [];
  return `<div class="profile-panel">
    <h3>${escapeHtml(t("datasetProfileTitle"))}</h3>
    ${renderKeyValues(summary)}
    ${warnings}
    <div class="summary">
      <div class="kv"><span class="key">columns</span><span class="value">${escapeHtml(columns.join(", ") || "-")}</span></div>
      <div class="kv"><span class="key">target_cols</span><span class="value">${escapeHtml(targets.join(", ") || "-")}</span></div>
      <div class="kv"><span class="key">missing</span><span class="value">${escapeHtml(formatValue(profile?.target_missing_counts || {}))}</span></div>
      <div class="kv"><span class="key">dtypes</span><span class="value">${escapeHtml(formatValue(profile?.target_dtypes || {}))}</span></div>
    </div>
  </div>`;
}

function switchDetailTab(tabName) {
  state.detailTab = tabName || "overview";
  document.querySelectorAll("[data-detail-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.detailTab === state.detailTab);
  });
  ["overview", "artifacts", "raw"].forEach((name) => {
    document.querySelector(`#detail-${name}`).hidden = name !== state.detailTab;
  });
}

function toggleSelectedFavorite() {
  if (!state.selectedRun) {
    return;
  }
  const key = runKey(state.selectedRun.summary);
  if (state.favorites.has(key)) {
    state.favorites.delete(key);
  } else {
    state.favorites.add(key);
  }
  saveFavorites();
  renderExperimentList();
  renderDetail();
}

function reloadSelectedRun() {
  if (state.selectedKey) {
    selectRun(state.selectedKey);
  }
}

function syncRunLookupFields(summary) {
  const experimentInput = document.querySelector("#run-lookup-experiment");
  const runInput = document.querySelector("#run-lookup-run");
  if (!experimentInput || !runInput || !summary) {
    return;
  }
  experimentInput.value = summary.experiment_name || "";
  runInput.value = summary.run_id || summary.compare_run_id || "latest";
}

function readRunLookupFields(options = {}) {
  const runsRoot = fieldValue("#run-lookup-runs-root") || "runs";
  const experimentName = fieldValue("#run-lookup-experiment");
  const runId = fieldValue("#run-lookup-run") || "latest";
  const artifactName = fieldValue("#run-lookup-artifact");
  if (!runsRoot) {
    throw new Error("runs root is required");
  }
  if (!experimentName) {
    throw new Error("experiment is required");
  }
  if (options.requireArtifact && !artifactName) {
    throw new Error("artifact is required");
  }
  return { runsRoot, experimentName, runId, artifactName };
}

async function loadRunLookupPayload(event, kind) {
  const button = event.currentTarget;
  const status = document.querySelector("#run-lookup-status");
  const output = document.querySelector("#run-lookup-output");
  let fields;
  try {
    fields = readRunLookupFields();
  } catch (error) {
    output.innerHTML = renderError(error.message);
    return;
  }
  await withBusyButton(button, status, t("loading"), async () => {
    const payload = await apiFetch(`/tools/experiments/${kind}`, {
      method: "POST",
      headers: postJsonHeaders,
      body: JSON.stringify({
        runs_root: fields.runsRoot,
        experiment: fields.experimentName,
        run: fields.runId,
      }),
    });
    output.innerHTML = renderJson(payload);
  });
}

async function downloadRunLookupArtifact(event) {
  const button = event.currentTarget;
  const status = document.querySelector("#run-lookup-status");
  const output = document.querySelector("#run-lookup-output");
  let fields;
  try {
    fields = readRunLookupFields({ requireArtifact: true });
  } catch (error) {
    output.innerHTML = renderError(error.message);
    return;
  }
  await withBusyButton(button, status, t("loading"), async () => {
    const response = await fetch("/tools/experiments/artifact", {
      method: "POST",
      headers: postJsonHeaders,
      body: JSON.stringify({
        runs_root: fields.runsRoot,
        experiment: fields.experimentName,
        run: fields.runId,
        artifact: fields.artifactName,
      }),
    });
    const blob = await response.blob();
    if (!response.ok) {
      const text = await blob.text();
      throw new Error(`${response.status} ${text || response.statusText}`);
    }
    const filename =
      filenameFromContentDisposition(response.headers.get("content-disposition")) ||
      safeFilename(fields.artifactName);
    downloadBlob(filename, blob);
    output.innerHTML = renderJson({
      downloaded: fields.artifactName,
      experiment: fields.experimentName,
      run: fields.runId,
      runs_root: fields.runsRoot,
      filename,
    });
  });
}

function exportSelectedReport() {
  const selected = state.selectedRun;
  if (!selected || !selected.results) {
    return;
  }
  const report = buildExperimentReport(selected);
  const filename = `${safeFilename(selected.summary.experiment_name || "experiment")}_${safeFilename(
    selected.summary.run_id || selected.summary.compare_run_id || "latest",
  )}_report.md`;
  downloadTextFile(filename, report);
}

function buildExperimentReport(selected) {
  const { summary, results, leaderboard, artifacts } = selected;
  const lines = [
    `# ${markdownText(summary.experiment_name || "Experiment")} Report`,
    "",
    "## Run Summary",
    "",
    markdownTable(
      [
        ["Experiment", summary.experiment_name],
        ["Run ID", summary.run_id || summary.compare_run_id],
        ["Run type", summary.run_type],
        ["Status", summary.status],
        ["Created at", summary.created_at],
      ],
      ["Field", "Value"],
    ),
    "",
  ];

  if (summary.run_type === "compare") {
    lines.push(...buildCompareReportSections(results, leaderboard));
  } else {
    lines.push(...buildTrainReportSections(results));
  }
  lines.push(...buildArtifactReportSection(artifacts));
  lines.push(
    "## Raw Result Pointers",
    "",
    "- `results.json`: canonical run payload.",
    "- `artifacts.json`: manifest-backed output inventory.",
    "",
    "_Generated from the local TS Platform dashboard._",
    "",
  );
  return lines.join("\n");
}

function buildTrainReportSections(results) {
  const originalMetrics = results?.test_metrics?.original || {};
  const historySeries = collectTrainingSeries(results?.history);
  const dataMetadata = results?.data_metadata || {};
  const forecastSamples = results?.forecast_samples || {};
  const sampleCount = Array.isArray(forecastSamples.samples) ? forecastSamples.samples.length : 0;
  return [
    "## Test Metrics",
    "",
    markdownTable(Object.entries(originalMetrics), ["Metric", "Value"]),
    "",
    "## Training Monitor Summary",
    "",
    markdownTable(
      historySeries.map((series) => {
        const stats = monitorStats(series.points);
        return [series.label, stats.latest, stats.best, stats.delta];
      }),
      ["Metric", "Latest", "Best", "Delta"],
    ),
    "",
    "## Forecast Preview",
    "",
    markdownTable(
      [
        ["Target columns", (forecastSamples.target_cols || []).join(", ")],
        ["Forecast horizon", (forecastSamples.horizon || []).join(", ")],
        ["Sample count", sampleCount],
      ],
      ["Field", "Value"],
    ),
    "",
    "## Data Metadata",
    "",
    markdownTable(Object.entries(dataMetadata), ["Field", "Value"]),
    "",
  ];
}

function buildCompareReportSections(results, leaderboard) {
  const rows = leaderboard || results?.rows || [];
  const primaryMetric = results?.primary_metric || "mae";
  const topRows = rows
    .filter((row) => row.status === "success")
    .slice(0, 10)
    .map((row) => [
      row.rank,
      row.model_alias || row.model_name,
      row.primary_metric_value ?? row[`test_${primaryMetric}`],
      row.test_mae,
      row.test_rmse,
      row.feature_aware,
    ]);
  return [
    "## Compare Summary",
    "",
    markdownTable(
      [
        ["Primary metric", primaryMetric],
        ["Success count", results?.success_count],
        ["Failed count", results?.failed_count],
      ],
      ["Field", "Value"],
    ),
    "",
    "## Leaderboard",
    "",
    markdownTable(topRows, ["Rank", "Model", primaryMetric, "MAE", "RMSE", "Feature aware"]),
    "",
  ];
}

function buildArtifactReportSection(artifacts) {
  const rows = (artifacts?.artifacts || []).map((item) => [
    item.name,
    item.kind,
    item.description,
  ]);
  return ["## Artifacts", "", markdownTable(rows, ["Name", "Kind", "Description"]), ""];
}

function markdownTable(rows, headers) {
  const normalizedRows = Array.isArray(rows) ? rows : [];
  const header = `| ${headers.map(markdownCell).join(" | ")} |`;
  const divider = `| ${headers.map(() => "---").join(" | ")} |`;
  if (!normalizedRows.length) {
    return [header, divider, `| ${headers.map(() => "-").join(" | ")} |`].join("\n");
  }
  const body = normalizedRows
    .map((row) => {
      const cells = Array.isArray(row) ? row : headers.map((headerName) => row?.[headerName]);
      return `| ${headers.map((_, index) => markdownCell(cells[index])).join(" | ")} |`;
    })
    .join("\n");
  return [header, divider, body].join("\n");
}

function markdownCell(value) {
  return markdownText(formatValue(value)).replaceAll("\n", " ");
}

function markdownText(value) {
  return String(value ?? "-").replaceAll("|", "\\|");
}

function safeFilename(value) {
  return (
    String(value || "report")
      .replace(/[^A-Za-z0-9_.-]+/g, "_")
      .replace(/^[_\-.]+|[_\-.]+$/g, "") || "report"
  );
}

function downloadTextFile(filename, content) {
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  downloadBlob(filename, blob);
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function filenameFromContentDisposition(value) {
  if (!value) {
    return "";
  }
  const utf8Match = value.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match) {
    try {
      return decodeURIComponent(utf8Match[1].replace(/"/g, ""));
    } catch {
      return utf8Match[1].replace(/"/g, "");
    }
  }
  const asciiMatch = value.match(/filename="?([^";]+)"?/i);
  return asciiMatch ? asciiMatch[1] : "";
}

async function runTrainDemo(demoName, button) {
  const status = document.querySelector("#train-status");
  const output = document.querySelector("#train-output");
  await withBusyButton(button, status, `${t("runningPrefix")} ${demoName}...`, async () => {
    output.innerHTML = "";
    const payload = await apiFetch(`/demo/train/${demoName}`, { method: "POST" });
    output.innerHTML = renderRunCompletion(payload, t("trainDemoComplete"));
    await loadDashboard();
    await selectRun(`${payload.experiment_name}::${payload.run_id}`);
    setDashboardPage("results", { updateHash: true, smooth: true });
  });
}

async function runCompareDemo(demoName, button) {
  const status = document.querySelector("#compare-status");
  const output = document.querySelector("#compare-output");
  await withBusyButton(button, status, `${t("runningPrefix")} ${demoName}...`, async () => {
    output.innerHTML = "";
    const payload = await apiFetch(`/demo/compare/${demoName}`, { method: "POST" });
    output.innerHTML = renderRunCompletion(payload, t("compareDemoComplete"));
    await loadDashboard();
    await selectRun(`${payload.experiment_name}::${payload.compare_run_id || payload.run_id}`);
    setDashboardPage("results", { updateHash: true, smooth: true });
  });
}

async function submitSelectedDemoJob(button) {
  const kind = currentJobDemoKind();
  const demoName = document.querySelector("#job-demo-name").value;
  const status = document.querySelector("#job-submit-status");
  const output = document.querySelector("#job-submit-output");
  if (!demoName) {
    output.innerHTML = renderError(t("noRows"));
    return;
  }
  await withBusyButton(button, status, `${t("runningPrefix")} ${demoName}...`, async () => {
    const payload = await apiFetch(`/demo/jobs/${kind}/${demoName}`, { method: "POST" });
    output.innerHTML = renderJobSubmission(payload);
    document.querySelector("#job-id").value = payload.job_id || "";
    await loadJobs();
    if (kind === "train") {
      startAutoJobProgress();
    }
  });
}

async function startHalfHourDemo(button) {
  const status = document.querySelector("#job-submit-status");
  const output = document.querySelector("#job-submit-output");
  document.querySelector("#job-demo-kind").value = "train";
  renderJobDemoOptions();
  document.querySelector("#job-demo-name").value = halfHourDemoName;
  await withBusyButton(button, status, `${t("runningPrefix")} ${halfHourDemoName}...`, async () => {
    const payload = await apiFetch(`/demo/jobs/train/${halfHourDemoName}`, { method: "POST" });
    output.innerHTML = renderJobSubmission(payload);
    document.querySelector("#job-id").value = payload.job_id || "";
    await loadJobs();
    startAutoJobProgress();
  });
}

async function submitAllDemoJobs(button) {
  const kind = currentJobDemoKind();
  const configs = state.demoConfigs[kind] || [];
  const status = document.querySelector("#job-submit-status");
  const output = document.querySelector("#job-submit-output");
  if (!configs.length) {
    output.innerHTML = renderError(t("noRows"));
    return;
  }
  await withBusyButton(button, status, `${t("runningPrefix")} ${kind} jobs...`, async () => {
    const payloads = [];
    for (const demoName of configs) {
      payloads.push(await apiFetch(`/demo/jobs/${kind}/${demoName}`, { method: "POST" }));
    }
    output.innerHTML = renderTable(payloads, [
      "job_id",
      "job_type",
      "status",
      "experiment_name",
      "created_at",
    ]);
    const lastJob = payloads[payloads.length - 1];
    document.querySelector("#job-id").value = lastJob?.job_id || "";
    await loadJobs();
    if (lastJob?.job_type === "train") {
      startAutoJobProgress();
    }
  });
}

function renderJobSubmission(payload) {
  const cards = [
    metric(t("jobSubmitted"), payload.job_id || "-", payload.experiment_name || "-"),
    metric(t("statusLabel"), payload.status || "-", t("jobId")),
    metric(t("jobType"), payload.job_type || "-", payload.run_id || payload.compare_run_id || "-"),
  ];
  return `<div class="summary">${cards.join("")}</div>`;
}

function currentJobDemoKind() {
  return document.querySelector("#job-demo-kind").value === "compare" ? "compare" : "train";
}

function renderJobDemoOptions() {
  const kindSelect = document.querySelector("#job-demo-kind");
  const configSelect = document.querySelector("#job-demo-name");
  if (!kindSelect || !configSelect) {
    return;
  }
  const kind = currentJobDemoKind();
  const previous = configSelect.value;
  const configs = state.demoConfigs[kind] || [];
  configSelect.innerHTML = configs
    .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
    .join("");
  configSelect.value = configs.includes(previous) ? previous : configs[0] || "";
}

function normalizeDemoConfigs(payload = {}) {
  return {
    train: Array.isArray(payload.train) && payload.train.length
      ? payload.train
      : fallbackDemoConfigs.train,
    compare: Array.isArray(payload.compare) && payload.compare.length
      ? payload.compare
      : fallbackDemoConfigs.compare,
  };
}

function initializeCustomBuilder() {
  renderCustomModelOptions({ models: fallbackModels });
  renderDatasetCatalog({ datasets: [] });
  renderJobDemoOptions();
  syncCustomDataSource();
  setCustomMode("train");
  renderCustomPreview();
}

function renderDatasetCatalog(datasetsPayload = {}) {
  const serverRows = Array.isArray(datasetsPayload.datasets) ? datasetsPayload.datasets : [];
  const allRows = mergeDatasetRows(serverRows, state.datasetCatalog.customRows);
  state.datasetCatalog.allRows = allRows;
  renderDatasetDomainFilterOptions(allRows);
  const rows = filterDatasetRows(allRows);
  state.datasetCatalog.rows = rows;
  state.datasetCatalog.selectedKey = resolveDatasetSelection(rows);
  setText("#dataset-catalog-status", t("experimentsLoaded"));
  setText(
    "#dataset-catalog-count",
    rows.length === allRows.length ? String(rows.length) : `${rows.length}/${allRows.length}`,
  );

  const list = document.querySelector("#dataset-catalog-list");
  if (!rows.length) {
    list.innerHTML = `<div class="empty-state compact-empty"><p>${escapeHtml(t("noRows"))}</p></div>`;
    renderCustomDatasetOptions();
    return;
  }

  const options = rows
    .map(
      (row) => `<option value="${escapeHtml(row._catalogKey)}">${escapeHtml(datasetOptionLabel(row))}</option>`,
    )
    .join("");
  list.innerHTML = `<label class="field dataset-picker">
    <span>${escapeHtml(t("selectDataset"))}</span>
    <select id="dataset-catalog-select">${options}</select>
  </label>
  <div id="dataset-catalog-detail"></div>`;
  const select = list.querySelector("#dataset-catalog-select");
  select.value = state.datasetCatalog.selectedKey;
  select.addEventListener("change", (event) => {
    state.datasetCatalog.selectedKey = event.target.value;
    renderDatasetCatalogDetail();
  });
  renderDatasetCatalogDetail();
  renderCustomDatasetOptions();
}

function renderDatasetCatalogDetail() {
  const detail = document.querySelector("#dataset-catalog-detail");
  if (!detail) {
    return;
  }
  const row = state.datasetCatalog.rows.find(
    (item) => item._catalogKey === state.datasetCatalog.selectedKey,
  );
  if (!row) {
    detail.innerHTML = `<div class="empty-state compact-empty"><p>${escapeHtml(t("noRows"))}</p></div>`;
    return;
  }
  const deleteAction = row.user_defined
    ? `<button class="table-action ghost danger-action" type="button" data-delete-dataset="${escapeHtml(row._catalogKey)}">${escapeHtml(t("deleteDataset"))}</button>`
    : "";
  const profileAction = row.dataset_type === "csv" && row.path
    ? `<button class="table-action ghost" type="button" data-profile-dataset="${escapeHtml(row._catalogKey)}">${escapeHtml(t("datasetProfile"))}</button>`
    : "";
  const path = row.path ? escapeHtml(row.path) : escapeHtml(t("noDatasetPath"));
  detail.innerHTML = `<article class="dataset-detail-card">
    <div class="panel-heading">
      <div>
        <h3>${escapeHtml(row.name || "-")}</h3>
        <p class="section-caption">${escapeHtml(row.description || "-")}</p>
      </div>
      <span class="count-pill">${escapeHtml(row.domain || "-")}</span>
    </div>
    <dl class="dataset-detail-grid">
      <div><dt>type</dt><dd>${escapeHtml(row.dataset_type || "-")}</dd></div>
      <div><dt>${escapeHtml(t("frequency"))}</dt><dd>${escapeHtml(row.frequency || "-")}</dd></div>
      <div><dt>${escapeHtml(t("targetCols"))}</dt><dd>${escapeHtml((row.target_cols || []).join(", ") || "-")}</dd></div>
      <div><dt>${escapeHtml(t("featureCols"))}</dt><dd>${escapeHtml((row.feature_cols || []).join(", ") || "-")}</dd></div>
      <div class="span-2"><dt>${escapeHtml(t("datasetSource"))}</dt><dd>${renderSourceLink(row.source)}</dd></div>
      <div class="span-2"><dt>path</dt><dd>${path}</dd></div>
    </dl>
    <div class="table-actions">
      <button class="table-action ghost" type="button" data-use-dataset="${escapeHtml(row._catalogKey)}">${escapeHtml(t("useDataset"))}</button>
      ${profileAction}
      ${deleteAction}
    </div>
  </article>`;
  detail.querySelectorAll("[data-use-dataset]").forEach((button) => {
    button.addEventListener("click", () => {
      applyDatasetTemplate(button.dataset.useDataset);
      document.querySelector("#custom-dataset-template").value = button.dataset.useDataset;
      setDashboardPage("custom", { updateHash: true, smooth: true });
    });
  });
  detail.querySelectorAll("[data-delete-dataset]").forEach((button) => {
    button.addEventListener("click", () => {
      deleteUserDataset(button.dataset.deleteDataset);
    });
  });
  detail.querySelectorAll("[data-profile-dataset]").forEach((button) => {
    button.addEventListener("click", () => {
      profileDataset(button.dataset.profileDataset);
    });
  });
}

function resolveDatasetSelection(rows) {
  const previous = state.datasetCatalog.selectedKey;
  if (rows.some((row) => row._catalogKey === previous)) {
    return previous;
  }
  return rows[0]?._catalogKey || "";
}

function datasetOptionLabel(row) {
  return [row.name || "-", row.domain || "-", row.dataset_type || "-"].join(" · ");
}

function mergeDatasetRows(serverRows, customRows) {
  const rows = new Map();
  serverRows.forEach((row) => {
    const normalized = normalizeName(row.name || "");
    if (normalized) {
      const prefix = row.user_defined ? "user" : "catalog";
      rows.set(`${prefix}:${normalized}`, { ...row, _catalogKey: `${prefix}:${normalized}` });
    }
  });
  customRows.forEach((row) => {
    const normalized = normalizeName(row.name || "");
    if (normalized) {
      rows.set(`user:${normalized}`, {
        ...row,
        user_defined: true,
        _catalogKey: `user:${normalized}`,
      });
    }
  });
  return [...rows.values()].sort((a, b) =>
    String(a.name || "").localeCompare(String(b.name || "")),
  );
}

function countMergedDatasets(datasetsPayload = {}) {
  if (Array.isArray(datasetsPayload.datasets)) {
    return mergeDatasetRows(datasetsPayload.datasets, state.datasetCatalog.customRows).length;
  }
  const serverCount = Array.isArray(datasetsPayload.names) ? datasetsPayload.names.length : 0;
  return serverCount + state.datasetCatalog.customRows.length;
}

function filterDatasetRows(rows) {
  const query = state.datasetCatalog.filters.search;
  const domain = state.datasetCatalog.filters.domain;
  return rows.filter((row) => {
    const haystack = [
      row.name,
      row.domain,
      row.dataset_type,
      row.frequency,
      row.description,
      row.source,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    if (query && !haystack.includes(query)) {
      return false;
    }
    return domain === "all" || String(row.domain || "") === domain;
  });
}

function renderDatasetDomainFilterOptions(rows) {
  const select = document.querySelector("#dataset-domain-filter");
  if (!select) {
    return;
  }
  const previous = state.datasetCatalog.filters.domain || select.value || "all";
  const domains = Array.from(
    new Set(rows.map((row) => row.domain).filter(Boolean).map((domain) => String(domain))),
  ).sort();
  select.innerHTML = [
    `<option value="all">${escapeHtml(t("filterAll"))}</option>`,
    ...domains.map((domain) => `<option value="${escapeHtml(domain)}">${escapeHtml(domain)}</option>`),
  ].join("");
  select.value = domains.includes(previous) ? previous : "all";
  state.datasetCatalog.filters.domain = select.value;
}

function datasetRowsForLookup() {
  return state.datasetCatalog.allRows.length ? state.datasetCatalog.allRows : state.datasetCatalog.rows;
}

function renderCustomDatasetOptions() {
  const select = document.querySelector("#custom-dataset-template");
  const previous = select.value;
  const rows = datasetRowsForLookup();
  select.innerHTML = [
    `<option value="">${escapeHtml(t("datasetTemplateManual"))}</option>`,
    ...rows.map(
      (row) => `<option value="${escapeHtml(row._catalogKey)}">${escapeHtml(row.name || "-")} · ${escapeHtml(row.domain || "-")}</option>`,
    ),
  ].join("");
  select.value = [...select.options].some((option) => option.value === previous) ? previous : "";
  renderDatasetSourceNote(select.value);
}

function applyDatasetTemplate(catalogKey) {
  const row = datasetRowsForLookup().find((item) => item._catalogKey === catalogKey);
  if (!row) {
    renderDatasetSourceNote("");
    renderCustomPreview();
    return;
  }
  const isSynthetic = row.dataset_type === "synthetic";
  document.querySelector("#custom-data-source").value = isSynthetic ? "synthetic" : "csv";
  if (!isSynthetic) {
    document.querySelector("#custom-csv-path").value = row.path || "";
    document.querySelector("#custom-timestamp-col").value = row.timestamp_col || "";
    document.querySelector("#custom-target-cols").value = (row.target_cols || []).join(", ");
    document.querySelector("#custom-feature-cols").value = (row.feature_cols || []).join(", ");
  }
  syncCustomDataSource();
  renderDatasetSourceNote(catalogKey);
  renderCustomPreview();
  document.querySelector("#custom-output").innerHTML = `<p class="muted">${escapeHtml(t("usedDataset"))}</p>`;
}

function renderDatasetSourceNote(catalogKey) {
  const note = document.querySelector("#custom-dataset-source");
  const row = datasetRowsForLookup().find((item) => item._catalogKey === catalogKey);
  if (!row) {
    note.innerHTML = "";
    return;
  }
  const source = renderSourceLink(row.source);
  const path = row.path ? `<span>${escapeHtml(row.path)}</span>` : `<span>${escapeHtml(t("noDatasetPath"))}</span>`;
  note.innerHTML = `<div>
    <strong>${escapeHtml(row.name || "-")}</strong>
    <span>${escapeHtml(row.domain || "-")} · ${escapeHtml(row.dataset_type || "-")} · ${escapeHtml(row.frequency || "-")}</span>
  </div>
  <p>${escapeHtml(row.description || "-")}</p>
  <p>${escapeHtml(t("datasetSource"))}: ${source}</p>
  <p>path: ${path}</p>`;
}

async function saveUserDataset(event) {
  event.preventDefault();
  const output = document.querySelector("#user-dataset-output");
  const name = fieldValue("#user-dataset-name");
  const path = fieldValue("#user-dataset-path");
  const targetCols = parseCommaList(fieldValue("#user-dataset-targets"));
  if (!/^[A-Za-z0-9_.-]{1,80}$/.test(name)) {
    output.innerHTML = renderError(t("userDatasetNameInvalid"));
    return;
  }
  if (!path) {
    output.innerHTML = renderError(t("userDatasetPathRequired"));
    return;
  }
  if (!targetCols.length) {
    output.innerHTML = renderError(t("targetColsRequired"));
    return;
  }
  const row = {
    name,
    dataset_type: "csv",
    domain: fieldValue("#user-dataset-domain") || "custom",
    description: fieldValue("#user-dataset-description") || "User supplied local CSV dataset.",
    source: fieldValue("#user-dataset-source") || path,
    path,
    timestamp_col: fieldValue("#user-dataset-timestamp") || null,
    target_cols: targetCols,
    feature_cols: parseCommaList(fieldValue("#user-dataset-features")),
    frequency: fieldValue("#user-dataset-frequency") || null,
    license: "user-supplied",
    citation: null,
  };
  output.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    const savedRow = await apiFetch("/datasets/user", {
      method: "POST",
      headers: postJsonHeaders,
      body: JSON.stringify(row),
    });
    const normalized = normalizeName(savedRow.name || name);
    state.datasetCatalog.customRows = [
      ...state.datasetCatalog.customRows.filter((item) => normalizeName(item.name) !== normalized),
      savedRow,
    ];
    const catalogKey = `user:${normalized}`;
    state.datasetCatalog.selectedKey = catalogKey;
    saveUserDatasets();
    state.overview.datasets = await apiFetch("/datasets");
    renderDatasetCatalog(state.overview.datasets || { datasets: [] });
    document.querySelector("#custom-dataset-template").value = catalogKey;
    applyDatasetTemplate(catalogKey);
    output.innerHTML = `<p class="muted">${escapeHtml(t("datasetSaved"))}</p>`;
  } catch (error) {
    output.innerHTML = renderError(error.message);
  }
}

async function clearUserDatasets() {
  const output = document.querySelector("#user-dataset-output");
  output.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    await apiFetch("/datasets/user", { method: "DELETE" });
    state.datasetCatalog.customRows = [];
    if (state.datasetCatalog.selectedKey.startsWith("user:")) {
      state.datasetCatalog.selectedKey = "";
    }
    saveUserDatasets();
    state.overview.datasets = await apiFetch("/datasets");
    renderDatasetCatalog(state.overview.datasets || { datasets: [] });
    output.innerHTML = "";
  } catch (error) {
    output.innerHTML = renderError(error.message);
  }
}

async function deleteUserDataset(catalogKey) {
  const row = datasetRowsForLookup().find((item) => item._catalogKey === catalogKey);
  const output = document.querySelector("#user-dataset-output");
  if (!row || !row.user_defined || !row.name) {
    return;
  }
  output.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    await apiFetch(`/datasets/user/${encodeURIComponent(row.name)}`, { method: "DELETE" });
    const normalized = normalizeName(row.name);
    state.datasetCatalog.customRows = state.datasetCatalog.customRows.filter(
      (item) => normalizeName(item.name) !== normalized,
    );
    if (state.datasetCatalog.selectedKey === catalogKey) {
      state.datasetCatalog.selectedKey = "";
    }
    saveUserDatasets();
    state.overview.datasets = await apiFetch("/datasets");
    renderDatasetCatalog(state.overview.datasets || { datasets: [] });
    output.innerHTML = `<p class="muted">${escapeHtml(t("datasetDeleted"))}</p>`;
  } catch (error) {
    output.innerHTML = renderError(error.message);
  }
}

async function profileDataset(catalogKey) {
  const row = datasetRowsForLookup().find((item) => item._catalogKey === catalogKey);
  const output = document.querySelector("#user-dataset-output");
  if (!row || !row.name) {
    return;
  }
  output.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    const params = new URLSearchParams();
    const inputLen = Number(fieldValue("#custom-input-len"));
    const outputLen = Number(fieldValue("#custom-output-len"));
    if (Number.isInteger(inputLen) && inputLen > 0) {
      params.set("input_len", String(inputLen));
    }
    if (Number.isInteger(outputLen) && outputLen > 0) {
      params.set("output_len", String(outputLen));
    }
    const query = params.toString() ? `?${params.toString()}` : "";
    const payload = await apiFetch(`/datasets/${encodeURIComponent(row.name)}/profile${query}`);
    output.innerHTML = renderDatasetProfile(payload);
  } catch (error) {
    output.innerHTML = renderError(error.message);
  }
}

async function handleCsvFileSelection(event, target) {
  const input = event.target;
  const file = input.files && input.files[0];
  if (!file) {
    return;
  }
  const output = document.querySelector(target === "user" ? "#user-dataset-output" : "#custom-output");
  output.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    const payload = await uploadCsvFile(file);
    if (target === "user") {
      applyUploadedCsvToUserDataset(payload, file.name);
    } else {
      applyUploadedCsvToCustomForm(payload);
    }
    output.innerHTML = `<p class="muted">${escapeHtml(t("csvUploadComplete"))}</p>`;
  } catch (error) {
    output.innerHTML = renderError(error.message);
  } finally {
    input.value = "";
  }
}

async function uploadCsvFile(file) {
  const content = await file.text();
  return apiFetch("/datasets/upload-csv", {
    method: "POST",
    headers: postJsonHeaders,
    body: JSON.stringify({ filename: file.name, content }),
  });
}

function applyUploadedCsvToUserDataset(payload, originalName) {
  const columns = payload.columns || [];
  const guesses = guessCsvColumns(columns);
  const currentName = fieldValue("#user-dataset-name");
  if (!currentName || currentName === "my_dataset") {
    document.querySelector("#user-dataset-name").value = datasetNameFromFilename(originalName);
  }
  document.querySelector("#user-dataset-path").value = payload.path || "";
  document.querySelector("#user-dataset-timestamp").value = guesses.timestamp || "";
  document.querySelector("#user-dataset-targets").value = guesses.targets.join(", ");
  document.querySelector("#user-dataset-features").value = guesses.features.join(", ");
}

function applyUploadedCsvToCustomForm(payload) {
  const columns = payload.columns || [];
  const guesses = guessCsvColumns(columns);
  document.querySelector("#custom-data-source").value = "csv";
  document.querySelector("#custom-dataset-template").value = "";
  document.querySelector("#custom-csv-path").value = payload.path || "";
  document.querySelector("#custom-timestamp-col").value = guesses.timestamp || "";
  document.querySelector("#custom-target-cols").value = guesses.targets.join(", ");
  document.querySelector("#custom-feature-cols").value = guesses.features.join(", ");
  syncCustomDataSource();
  renderDatasetSourceNote("");
  renderCustomPreview();
}

function guessCsvColumns(columns) {
  const cleanColumns = columns.map((column) => String(column || "").trim()).filter(Boolean);
  const timestamp =
    cleanColumns.find((column) => ["timestamp", "date", "datetime", "time"].includes(column.toLowerCase())) ||
    "";
  const preferredTarget =
    cleanColumns.find((column) => ["value", "target", "y", "count", "load", "demand"].includes(column.toLowerCase())) ||
    cleanColumns.find((column) => column !== timestamp) ||
    "";
  const targets = preferredTarget ? [preferredTarget] : [];
  const features = cleanColumns.filter((column) => column !== timestamp && !targets.includes(column));
  return { features, targets, timestamp };
}

function datasetNameFromFilename(filename) {
  const stem = String(filename || "dataset").replace(/\.[^.]+$/, "");
  return stem.replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^[_\-.]+|[_\-.]+$/g, "") || "dataset";
}

function renderSourceLink(source) {
  if (!source) {
    return "-";
  }
  if (isHttpUrl(source)) {
    return `<a href="${escapeHtml(source)}" target="_blank" rel="noreferrer">${escapeHtml(source)}</a>`;
  }
  return `<span>${escapeHtml(source)}</span>`;
}

function renderCustomModelOptions(modelsPayload = {}) {
  const names = Array.isArray(modelsPayload.models) && modelsPayload.models.length
    ? modelsPayload.models
    : fallbackModels;
  const uniqueNames = Array.from(new Set(names));
  const orderedNames = [
    ...fallbackModels.filter((name) => uniqueNames.includes(name)),
    ...uniqueNames.filter((name) => !fallbackModels.includes(name)),
  ];
  state.custom.availableModels = orderedNames.length ? orderedNames : fallbackModels;

  const trainSelect = document.querySelector("#custom-train-model");
  const previousTrainModel = trainSelect.value || state.custom.lastTrainModel || "linear";
  trainSelect.innerHTML = state.custom.availableModels
    .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
    .join("");
  trainSelect.value = state.custom.availableModels.includes(previousTrainModel)
    ? previousTrainModel
    : state.custom.availableModels.includes("linear")
      ? "linear"
      : state.custom.availableModels[0];
  state.custom.lastTrainModel = trainSelect.value;

  renderCompareModelChecklist(state.custom.availableModels);
  syncTrainModelParams({ force: !document.querySelector("#custom-model-params").value.trim() });
  renderCustomPreview();
}

function renderCompareModelChecklist(modelNames) {
  const selected = new Set(
    [...document.querySelectorAll("[data-custom-compare-model]:checked")].map(
      (input) => input.dataset.customCompareModel,
    ),
  );
  const effectiveSelection = selected.size ? selected : defaultCompareSelections;
  document.querySelector("#custom-compare-models").innerHTML = modelNames
    .map((name) => {
      const params = defaultModelParams[name] || {};
      return `<label class="model-choice">
        <input type="checkbox" data-custom-compare-model="${escapeHtml(name)}" ${effectiveSelection.has(name) ? "checked" : ""} />
        <span>
          <strong>${escapeHtml(name)}</strong>
          <small>${escapeHtml(JSON.stringify(params))}</small>
        </span>
      </label>`;
    })
    .join("");
  document.querySelectorAll("[data-custom-compare-model]").forEach((input) => {
    input.addEventListener("change", renderCustomPreview);
  });

  const compareParams = document.querySelector("#custom-compare-params");
  if (!compareParams.value.trim()) {
    compareParams.value = JSON.stringify(getDefaultCompareParams(modelNames), null, 2);
  }
}

function getDefaultCompareParams(modelNames) {
  return Object.fromEntries(
    modelNames.map((name) => [name, defaultModelParams[name] || {}]),
  );
}

function setCustomMode(modeName) {
  const mode = modeName === "compare" ? "compare" : "train";
  state.custom.mode = mode;
  document.querySelectorAll("[data-custom-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.customMode === mode);
  });
  document.querySelector("#custom-train-fields").hidden = mode !== "train";
  document.querySelector("#custom-compare-fields").hidden = mode !== "compare";
  const submit = document.querySelector("#custom-submit");
  const submitKey = mode === "compare" ? "customSubmitCompare" : "customSubmitTrain";
  submit.dataset.i18n = submitKey;
  submit.textContent = t(submitKey);
  renderCustomPreview();
}

function syncCustomDataSource() {
  const source = fieldValue("#custom-data-source");
  document.querySelector("#custom-synthetic-fields").hidden = source !== "synthetic";
  document.querySelector("#custom-csv-fields").hidden = source !== "csv";
}

function syncTrainModelParams(options = {}) {
  const trainSelect = document.querySelector("#custom-train-model");
  const paramsInput = document.querySelector("#custom-model-params");
  const nextModel = trainSelect.value || "linear";
  const previousParams = JSON.stringify(defaultModelParams[state.custom.lastTrainModel] || {}, null, 2);
  const shouldReplace =
    options.force || !paramsInput.value.trim() || paramsInput.value.trim() === previousParams;
  if (shouldReplace) {
    paramsInput.value = JSON.stringify(defaultModelParams[nextModel] || {}, null, 2);
  }
  state.custom.lastTrainModel = nextModel;
  renderCustomPreview();
}

async function runCustomExperiment(event) {
  event.preventDefault();
  const status = document.querySelector("#custom-status");
  const output = document.querySelector("#custom-output");
  let config;
  try {
    config = buildCustomConfig();
  } catch (error) {
    status.textContent = t("error");
    output.innerHTML = renderError(error.message);
    renderCustomPreview();
    return;
  }

  output.innerHTML = "";
  const buttons = document.querySelectorAll("button");
  buttons.forEach((button) => {
    button.disabled = true;
  });
  status.textContent = `${t("runningPrefix")} ${config.experiment.name}...`;
  try {
    const endpoint = state.custom.mode === "compare" ? "/experiments/compare" : "/experiments/train";
    const payload = await apiFetch(endpoint, {
      method: "POST",
      headers: postJsonHeaders,
      body: JSON.stringify(config),
    });
    output.innerHTML = renderRunCompletion(
      payload,
      state.custom.mode === "compare" ? t("customCompareComplete") : t("customTrainComplete"),
    );
    await loadDashboard();
    await selectRun(`${payload.experiment_name}::${payload.compare_run_id || payload.run_id}`);
    setDashboardPage("results", { updateHash: true, smooth: true });
    status.textContent = t("idle");
  } catch (error) {
    status.textContent = t("error");
    output.innerHTML = renderError(error.message);
  } finally {
    buttons.forEach((button) => {
      button.disabled = false;
    });
  }
}

function buildCustomConfig(options = {}) {
  const validate = options.validate !== false;
  const dataName = fieldValue("#custom-data-source");
  const metrics = [...document.querySelectorAll("[data-custom-metric]:checked")].map(
    (input) => input.dataset.customMetric,
  );
  const config = {
    experiment: {
      name: fieldValue("#custom-experiment-name"),
      output_dir: "runs",
      seed: 42,
      overwrite: true,
    },
    data: {
      name: dataName,
      input_len: readNumber("#custom-input-len", t("inputLen"), { integer: true, min: 1 }),
      output_len: readNumber("#custom-output-len", t("outputLen"), { integer: true, min: 1 }),
      batch_size: readNumber("#custom-batch-size", t("batchSize"), { integer: true, min: 1 }),
      train_ratio: readNumber("#custom-train-ratio", t("trainRatio"), { min: 0, max: 1 }),
      val_ratio: readNumber("#custom-val-ratio", t("valRatio"), { min: 0, max: 1 }),
      test_ratio: readNumber("#custom-test-ratio", t("testRatio"), { min: 0, max: 1 }),
      scaler: {
        name: fieldValue("#custom-scaler"),
      },
      params: buildCustomDataParams(dataName),
    },
    training: {
      epochs: readNumber("#custom-epochs", t("epochs"), { integer: true, min: 1 }),
      learning_rate: readNumber("#custom-learning-rate", t("learningRate"), { min: 0 }),
      device: fieldValue("#custom-device"),
      optimizer: fieldValue("#custom-optimizer"),
      loss: fieldValue("#custom-loss"),
      checkpoint_every: readNumber("#custom-checkpoint-every", t("checkpointEvery"), {
        integer: true,
        min: 1,
      }),
    },
    evaluation: {
      metrics,
      include_scaled_metrics: document.querySelector("#custom-include-scaled").checked,
    },
  };

  if (state.custom.mode === "compare") {
    config.models = buildCustomCompareModels();
    config.primary_metric = fieldValue("#custom-primary-metric");
    config.continue_on_error = true;
  } else {
    config.model = {
      name: fieldValue("#custom-train-model"),
      params: parseJsonObject("#custom-model-params", t("modelParams")),
    };
  }

  if (validate) {
    validateCustomConfig(config);
  }
  return config;
}

function buildCustomDataParams(dataName) {
  if (dataName === "csv") {
    const featureCols = parseCommaList(fieldValue("#custom-feature-cols"));
    const params = {
      path: fieldValue("#custom-csv-path"),
      timestamp_col: fieldValue("#custom-timestamp-col") || null,
      target_cols: parseCommaList(fieldValue("#custom-target-cols")),
      missing_policy: fieldValue("#custom-missing-policy"),
      sort_by_time: document.querySelector("#custom-sort-by-time").checked,
    };
    if (featureCols.length) {
      params.feature_cols = featureCols;
    }
    return params;
  }
  return {
    length: readNumber("#custom-synthetic-length", t("seriesLength"), { integer: true, min: 1 }),
    num_features: readNumber("#custom-synthetic-features", t("featureCount"), {
      integer: true,
      min: 1,
    }),
    noise_std: readNumber("#custom-synthetic-noise", t("noiseStd"), { min: 0 }),
  };
}

function buildCustomCompareModels() {
  const selectedModels = [...document.querySelectorAll("[data-custom-compare-model]:checked")].map(
    (input) => input.dataset.customCompareModel,
  );
  const paramMap = parseJsonObject("#custom-compare-params", t("compareParams"));
  return selectedModels.map((name) => ({
    name,
    params: normalizeParamObject(paramMap[name] || defaultModelParams[name] || {}, name),
  }));
}

function validateCustomConfig(config) {
  if (!/^[A-Za-z0-9_.-]{1,80}$/.test(config.experiment.name)) {
    throw new Error(t("experimentNameInvalid"));
  }
  const splitTotal = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio;
  if (Math.abs(splitTotal - 1) > 0.0001) {
    throw new Error(t("splitRatioInvalid"));
  }
  if (!config.evaluation.metrics.length) {
    throw new Error(t("metricsRequired"));
  }
  if (
    config.training.device === "cuda" &&
    state.overview.health &&
    !state.overview.health.cuda_available
  ) {
    throw new Error(cudaUnavailableMessage);
  }
  if (config.data.name === "csv") {
    if (!config.data.params.path) {
      throw new Error(t("csvPathRequired"));
    }
    if (!config.data.params.target_cols.length) {
      throw new Error(t("targetColsRequired"));
    }
  }
  if (state.custom.mode === "compare") {
    if (!config.models || config.models.length < 2) {
      throw new Error(t("compareModelsRequired"));
    }
    if (!config.evaluation.metrics.includes(config.primary_metric)) {
      throw new Error(t("primaryMetricRequired"));
    }
  }
}

function renderCustomPreview() {
  const preview = document.querySelector("#custom-config-preview");
  if (!preview) {
    return;
  }
  try {
    preview.textContent = JSON.stringify(buildCustomConfig({ validate: false }), null, 2);
  } catch (error) {
    preview.textContent = `${t("customConfigInvalid")}: ${error.message}`;
  }
}

function renderRunCompletion(payload, title) {
  const cards = [metric(title, payload.experiment_name || "-", payload.run_id || payload.compare_run_id || "-")];
  if (payload.success_count !== undefined) {
    cards.push(metric(t("successCount"), payload.success_count, t("compareRuns")));
  }
  if (payload.primary_metric) {
    cards.push(metric(t("primaryMetric"), payload.primary_metric, t("bestMetric")));
  }
  return `<div class="summary">${cards.join("")}</div>`;
}

function readJobCliSettings() {
  return {
    jobBackend: fieldValue("#job-backend") === "sqlite" ? "sqlite" : "json",
    jobsRoot: fieldValue("#jobs-root") || "runs/jobs",
    sqliteDb: fieldValue("#jobs-sqlite-db") || "runs/jobs.sqlite3",
    runsRoot: fieldValue("#jobs-runs-root") || "runs",
  };
}

function jobQuery(options = {}) {
  const settings = readJobCliSettings();
  const params = new URLSearchParams();
  if (options.includeBackend) {
    params.set("job_backend", settings.jobBackend);
  }
  if (options.includeJobsRoot !== false) {
    params.set("jobs_root", settings.jobsRoot);
  }
  if (options.includeSqliteDb !== false) {
    params.set("sqlite_db", settings.sqliteDb);
  }
  if (options.includeRunsRoot) {
    params.set("runs_root", settings.runsRoot);
  }
  if (options.extra) {
    Object.entries(options.extra).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") {
        params.set(key, String(value));
      }
    });
  }
  const query = params.toString();
  return query ? `?${query}` : "";
}

async function loadJobs() {
  const output = document.querySelector("#jobs-output");
  output.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    const payload = await apiFetch(`/jobs${jobQuery({ includeBackend: true })}`);
    output.innerHTML = renderTable(payload.jobs || [], [
      "job_id",
      "status",
      "job_type",
      "created_at",
      "run_id",
      "error",
    ]);
  } catch (error) {
    output.innerHTML = renderError(error.message);
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
  const query = jobQuery({
    includeBackend: ["job", "result", "logs"].includes(kind),
    includeRunsRoot: ["result", "logs"].includes(kind),
  });
  output.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    const payload = await apiFetch(`/jobs/${encodeURIComponent(jobId)}${suffix}${query}`);
    output.innerHTML = renderJson(payload);
  } catch (error) {
    output.innerHTML = renderError(error.message);
  }
}

async function loadJobProgress(options = {}) {
  const jobId = document.querySelector("#job-id").value.trim();
  const output = document.querySelector("#jobs-output");
  if (!jobId) {
    output.innerHTML = renderError("job_id is required");
    return;
  }
  const query = jobQuery({ includeBackend: true, includeRunsRoot: true });
  if (!options.silent) {
    output.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  }
  try {
    const payload = await apiFetch(`/jobs/${encodeURIComponent(jobId)}/progress${query}`);
    output.innerHTML = renderJobProgress(payload);
    if (payload?.job?.status && !["queued", "running", "cancel_requested"].includes(payload.job.status)) {
      stopAutoJobProgress();
    }
  } catch (error) {
    output.innerHTML = renderError(error.message);
    stopAutoJobProgress();
  }
}

function toggleAutoJobProgress(event) {
  if (event.target.checked) {
    startAutoJobProgress();
    return;
  }
  stopAutoJobProgress();
}

function startAutoJobProgress() {
  const checkbox = document.querySelector("#auto-job-progress");
  if (checkbox) {
    checkbox.checked = true;
  }
  if (state.jobProgressTimer !== null) {
    window.clearInterval(state.jobProgressTimer);
  }
  loadJobProgress();
  state.jobProgressTimer = window.setInterval(() => {
    loadJobProgress({ silent: true });
  }, 3000);
}

function stopAutoJobProgress() {
  if (state.jobProgressTimer !== null) {
    window.clearInterval(state.jobProgressTimer);
    state.jobProgressTimer = null;
  }
  const checkbox = document.querySelector("#auto-job-progress");
  if (checkbox) {
    checkbox.checked = false;
  }
}

function renderJobProgress(payload) {
  const progress = payload?.progress || {};
  const job = payload?.job || {};
  const completed = progress.completed_epochs ?? "-";
  const total = progress.total_epochs ?? "-";
  const percent = progress.progress_percent ?? 0;
  const latestLoss = progress.latest?.train_loss;
  const latestValMae = progress.latest?.validation_metrics?.original?.mae;
  const testMae = progress.test_metrics?.original?.mae;
  return `<div class="summary">
    ${metric("job", job.job_id || "-")}
    ${metric("status", job.status || "-")}
    ${metric("epoch", `${completed} / ${total}`)}
    ${metric("progress", `${formatNumber(percent)}%`)}
    ${metric("train_loss", latestLoss === undefined ? "-" : formatNumber(latestLoss))}
    ${metric("val_mae", latestValMae === undefined ? "-" : formatNumber(latestValMae))}
    ${metric("test_mae", testMae === undefined ? "-" : formatNumber(testMae))}
    ${metric("updated", progress.updated_at || "-")}
  </div>
  ${progress.history ? renderTrainingMonitor({ history: progress.history }) : ""}
  <h3>progress.json</h3>
  ${renderJson(progress || null)}
  <h3>log tail</h3>
  <pre>${escapeHtml(payload?.log_tail || "")}</pre>`;
}

async function runCancelJob() {
  const jobId = document.querySelector("#job-id").value.trim();
  const output = document.querySelector("#jobs-output");
  if (!jobId) {
    output.innerHTML = renderError("job_id is required");
    return;
  }
  await runOperation("#jobs-output", "#jobs-output", async () => {
    const payload = await apiFetch(
      `/jobs/${encodeURIComponent(jobId)}/cancel${jobQuery({ includeBackend: true })}`,
      { method: "POST" },
    );
    output.innerHTML = renderJson(payload);
    await loadJobs();
  });
}

async function runRetryJob() {
  const jobId = document.querySelector("#job-id").value.trim();
  const output = document.querySelector("#jobs-output");
  if (!jobId) {
    output.innerHTML = renderError("job_id is required");
    return;
  }
  let maxAttempts;
  try {
    maxAttempts = readNumber("#retry-max-attempts", "max attempts", { integer: true, min: 1 });
  } catch (error) {
    output.innerHTML = renderError(error.message);
    return;
  }
  await runOperation("#jobs-output", "#jobs-output", async () => {
    const payload = await apiFetch(
      `/jobs/${encodeURIComponent(jobId)}/retry${jobQuery({
        extra: { max_attempts: maxAttempts },
      })}`,
      { method: "POST" },
    );
    output.innerHTML = renderJson(payload);
    await loadJobs();
  });
}

async function listStaleJobs() {
  let seconds;
  try {
    seconds = readNumber("#stale-seconds", "stale seconds", { integer: true, min: 1 });
  } catch (error) {
    document.querySelector("#jobs-output").innerHTML = renderError(error.message);
    return;
  }
  await runOperation("#jobs-output", "#jobs-output", async () => {
    const payload = await apiFetch(
      `/jobs/stale${jobQuery({ extra: { older_than_seconds: seconds } })}`,
    );
    document.querySelector("#jobs-output").innerHTML = renderJson(payload);
  });
}

async function markStaleTimeout() {
  let seconds;
  try {
    seconds = readNumber("#stale-seconds", "stale seconds", { integer: true, min: 1 });
  } catch (error) {
    document.querySelector("#jobs-output").innerHTML = renderError(error.message);
    return;
  }
  const reason = fieldValue("#timeout-reason");
  await runOperation("#jobs-output", "#jobs-output", async () => {
    const payload = await apiFetch(
      `/jobs/stale/timeout${jobQuery({
        extra: { older_than_seconds: seconds, reason },
      })}`,
      {
        method: "POST",
      },
    );
    document.querySelector("#jobs-output").innerHTML = renderJson(payload);
    await loadJobs();
  });
}

async function runWorkerOnce() {
  await runOperation("#jobs-output", "#jobs-output", async () => {
    const workerId = fieldValue("#worker-id") || "api_worker";
    const payload = await apiFetch(
      `/jobs/worker/once${jobQuery({
        includeRunsRoot: true,
        extra: { worker_id: workerId },
      })}`,
      {
        method: "POST",
      },
    );
    document.querySelector("#jobs-output").innerHTML = renderJson(payload);
    await loadJobs();
  });
}

async function runWorkerLoop() {
  let maxJobs;
  let maxIdleCycles;
  let sleepSeconds;
  try {
    maxJobs = readNumber("#worker-max-jobs", "max jobs", { integer: true, min: 1 });
    maxIdleCycles = readNumber("#worker-idle-cycles", "idle cycles", { integer: true, min: 1 });
    sleepSeconds = readNumber("#worker-sleep-seconds", "sleep seconds", { min: 0 });
  } catch (error) {
    document.querySelector("#jobs-output").innerHTML = renderError(error.message);
    return;
  }
  await runOperation("#jobs-output", "#jobs-output", async () => {
    const settings = readJobCliSettings();
    const payload = await apiFetch("/jobs/worker/loop", {
      method: "POST",
      headers: postJsonHeaders,
      body: JSON.stringify({
        max_jobs: maxJobs,
        max_idle_cycles: maxIdleCycles,
        sleep_seconds: sleepSeconds,
        worker_id: fieldValue("#worker-id") || "api_worker",
        jobs_root: settings.jobsRoot,
        sqlite_db: settings.sqliteDb,
        runs_root: settings.runsRoot,
      }),
    });
    document.querySelector("#jobs-output").innerHTML = renderJson(payload);
    await loadJobs();
  });
}

async function runConfigFile(event) {
  const button = event.currentTarget;
  const status = document.querySelector("#config-file-status");
  const output = document.querySelector("#config-file-output");
  const kind = fieldValue("#config-file-kind") === "compare" ? "compare" : "train";
  const mode = fieldValue("#config-file-mode") === "job" ? "job" : "sync";
  const configPath = fieldValue("#config-file-path");
  if (!configPath) {
    output.innerHTML = renderError("config path is required");
    return;
  }
  await withBusyButton(button, status, `${t("runningPrefix")} ${configPath}...`, async () => {
    const endpoint = mode === "job" ? `/jobs/${kind}-config` : `/configs/${kind}/run`;
    const payload = await apiFetch(endpoint, {
      method: "POST",
      headers: postJsonHeaders,
      body: JSON.stringify({ config_path: configPath }),
    });
    if (mode === "job") {
      output.innerHTML = renderJobSubmission(payload);
      document.querySelector("#job-id").value = payload.job_id || "";
      await loadJobs();
      if (kind === "train") {
        startAutoJobProgress();
      }
      return;
    }
    output.innerHTML = renderRunCompletion(payload, kind === "compare" ? t("compareDemoComplete") : t("trainDemoComplete"));
    await loadDashboard();
    await selectRun(`${payload.experiment_name}::${payload.compare_run_id || payload.run_id}`);
    setDashboardPage("results", { updateHash: true, smooth: true });
  });
}

async function predictSelectedRun(event) {
  const button = event.currentTarget;
  const status = document.querySelector("#prediction-status");
  const output = document.querySelector("#prediction-output");
  const selected = state.selectedRun?.summary;
  if (!selected || !selected.experiment_name || !selected.run_id || selected.status !== "complete") {
    output.innerHTML = renderError("select a completed training run first");
    return;
  }
  let values;
  try {
    values = readPredictionValues();
  } catch (error) {
    output.innerHTML = renderError(error.message);
    return;
  }
  await withBusyButton(button, status, t("loading"), async () => {
    const payload = await apiFetch(
      `/experiments/${encodeURIComponent(selected.experiment_name)}/${encodeURIComponent(selected.run_id)}/predict`,
      {
        method: "POST",
        headers: postJsonHeaders,
        body: JSON.stringify({ values }),
      },
    );
    output.innerHTML = renderPredictionPayload(payload);
  });
}

async function predictModelExportPath(event) {
  const button = event.currentTarget;
  const status = document.querySelector("#prediction-status");
  const output = document.querySelector("#prediction-output");
  const modelExportPath = fieldValue("#prediction-model-export-path");
  if (!modelExportPath) {
    output.innerHTML = renderError("model export path is required");
    return;
  }
  let values;
  try {
    values = readPredictionValues();
  } catch (error) {
    output.innerHTML = renderError(error.message);
    return;
  }
  await withBusyButton(button, status, t("loading"), async () => {
    const payload = await apiFetch("/predict/model-export", {
      method: "POST",
      headers: postJsonHeaders,
      body: JSON.stringify({ model_export_path: modelExportPath, values }),
    });
    output.innerHTML = renderPredictionPayload(payload);
  });
}

function readPredictionValues() {
  const rawValue = fieldValue("#prediction-values");
  try {
    const parsed = JSON.parse(rawValue);
    if (!Array.isArray(parsed)) {
      throw new Error("values must be a JSON array");
    }
    return parsed;
  } catch (error) {
    throw new Error(`values JSON: ${error.message}`);
  }
}

async function handlePredictionValuesFile(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  const status = document.querySelector("#prediction-status");
  const output = document.querySelector("#prediction-output");
  status.textContent = t("loading");
  try {
    const rawText = await file.text();
    const parsed = JSON.parse(rawText);
    if (!Array.isArray(parsed)) {
      throw new Error("values must be a JSON array");
    }
    document.querySelector("#prediction-values").value = JSON.stringify(parsed, null, 2);
    output.innerHTML = renderJson({
      values_file: file.name,
      top_level_windows: parsed.length,
    });
    status.textContent = t("idle");
  } catch (error) {
    status.textContent = t("error");
    output.innerHTML = renderError(`values file: ${error.message}`);
  } finally {
    event.target.value = "";
  }
}

function renderPredictionPayload(payload) {
  return `<div class="summary">
    ${metric("model", payload?.model?.name || "-")}
    ${metric("output_len", payload?.model?.output_len || "-")}
    ${metric("target_cols", (payload?.data?.target_cols || []).join(", ") || "-")}
  </div>${renderJson(payload)}`;
}

async function listRegisteredDatasetsTool(event) {
  const button = event.currentTarget;
  const status = document.querySelector("#dataset-tools-status");
  const output = document.querySelector("#dataset-tools-output");
  await withBusyButton(button, status, t("loading"), async () => {
    const payload = await apiFetch("/datasets");
    output.innerHTML = renderJson(payload);
  });
}

async function listModelsTool(event) {
  const button = event.currentTarget;
  const status = document.querySelector("#dataset-tools-status");
  const output = document.querySelector("#dataset-tools-output");
  await withBusyButton(button, status, t("loading"), async () => {
    const payload = await apiFetch("/models");
    output.innerHTML = renderJson(payload);
  });
}

async function profileCsvTool(event) {
  const button = event.currentTarget;
  const status = document.querySelector("#dataset-tools-status");
  const output = document.querySelector("#dataset-tools-output");
  await withBusyButton(button, status, t("loading"), async () => {
    const payload = await apiFetch("/datasets/profile-csv", {
      method: "POST",
      headers: postJsonHeaders,
      body: JSON.stringify({
        path: fieldValue("#dataset-tool-csv-path"),
        target_cols: parseCommaList(fieldValue("#dataset-tool-target-cols")),
        timestamp_col: fieldValue("#dataset-tool-timestamp-col") || null,
        input_len: readNumber("#dataset-tool-input-len", "input_len", { integer: true, min: 1 }),
        output_len: readNumber("#dataset-tool-output-len", "output_len", { integer: true, min: 1 }),
        name: fieldValue("#dataset-tool-profile-name") || null,
      }),
    });
    output.innerHTML = renderDatasetProfile(payload);
  });
}

async function profileCatalogTool(event) {
  const button = event.currentTarget;
  const status = document.querySelector("#dataset-tools-status");
  const output = document.querySelector("#dataset-tools-output");
  await withBusyButton(button, status, t("loading"), async () => {
    const payload = await apiFetch("/datasets/catalog/profile", {
      method: "POST",
      headers: postJsonHeaders,
      body: JSON.stringify({
        catalog_path: fieldValue("#dataset-tool-catalog-path"),
        input_len: readNumber("#dataset-tool-input-len", "input_len", { integer: true, min: 1 }),
        output_len: readNumber("#dataset-tool-output-len", "output_len", { integer: true, min: 1 }),
      }),
    });
    output.innerHTML = renderJson(payload);
  });
}

async function listCatalogTool(event) {
  const button = event.currentTarget;
  const status = document.querySelector("#dataset-tools-status");
  const output = document.querySelector("#dataset-tools-output");
  await withBusyButton(button, status, t("loading"), async () => {
    const payload = await apiFetch("/datasets/catalog/list", {
      method: "POST",
      headers: postJsonHeaders,
      body: JSON.stringify({
        catalog_path: fieldValue("#dataset-tool-catalog-path"),
      }),
    });
    output.innerHTML = renderJson(payload);
  });
}

async function generateCatalogConfigTool(event) {
  const button = event.currentTarget;
  const status = document.querySelector("#dataset-tools-status");
  const output = document.querySelector("#dataset-tools-output");
  await withBusyButton(button, status, t("loading"), async () => {
    const requestPayload = {
      catalog_path: fieldValue("#dataset-tool-catalog-path"),
      dataset: fieldValue("#dataset-tool-dataset-name"),
      input_len: readNumber("#dataset-tool-input-len", "input_len", { integer: true, min: 1 }),
      output_len: readNumber("#dataset-tool-output-len", "output_len", { integer: true, min: 1 }),
      model: fieldValue("#dataset-tool-model-name"),
      epochs: readNumber("#dataset-tool-epochs", "epochs", { integer: true, min: 1 }),
      batch_size: readNumber("#dataset-tool-batch-size", "batch_size", { integer: true, min: 1 }),
    };
    const outputPath = fieldValue("#dataset-tool-output-path");
    if (outputPath) {
      requestPayload.output_path = outputPath;
    }
    const payload = await apiFetch("/datasets/catalog/config", {
      method: "POST",
      headers: postJsonHeaders,
      body: JSON.stringify(requestPayload),
    });
    output.innerHTML = renderJson(payload);
  });
}

async function runOperation(statusSelector, outputSelector, callback) {
  const status = document.querySelector(statusSelector);
  const output = document.querySelector(outputSelector);
  status.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
  try {
    await callback();
  } catch (error) {
    output.innerHTML = renderError(error.message);
  }
}

async function withBusyButton(button, status, loadingText, callback) {
  const buttons = document.querySelectorAll("button");
  buttons.forEach((item) => {
    item.disabled = true;
  });
  status.textContent = loadingText;
  try {
    await callback();
    status.textContent = t("idle");
  } catch (error) {
    status.textContent = t("error");
    const output = button.dataset.outputTarget
      ? document.querySelector(button.dataset.outputTarget)
      : button.dataset.trainDemo
        ? document.querySelector("#train-output")
        : document.querySelector("#compare-output");
    output.innerHTML = renderError(error.message);
  } finally {
    buttons.forEach((item) => {
      item.disabled = false;
    });
  }
}

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

function fieldValue(selector) {
  return document.querySelector(selector).value.trim();
}

function readNumber(selector, label, options = {}) {
  const rawValue = fieldValue(selector);
  const value = Number(rawValue);
  if (!Number.isFinite(value)) {
    throw new Error(`${label} is invalid`);
  }
  if (options.integer && !Number.isInteger(value)) {
    throw new Error(`${label} must be an integer`);
  }
  if (options.min !== undefined && value < options.min) {
    throw new Error(`${label} must be >= ${options.min}`);
  }
  if (options.max !== undefined && value > options.max) {
    throw new Error(`${label} must be <= ${options.max}`);
  }
  return value;
}

function parseCommaList(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function normalizeName(value) {
  return String(value || "").trim().toLowerCase();
}

function isHttpUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

function parseJsonObject(selector, label) {
  const rawValue = fieldValue(selector);
  if (!rawValue) {
    return {};
  }
  let parsed;
  try {
    parsed = JSON.parse(rawValue);
  } catch (error) {
    throw new Error(`${label}: ${error.message}`);
  }
  return normalizeParamObject(parsed, label);
}

function normalizeParamObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be a JSON object`);
  }
  return value;
}

function renderTrainingMonitor(results) {
  const series = collectTrainingSeries(results?.history);
  if (!series.length) {
    return `<div id="training-monitor" class="visual-panel wide">
      <div class="visual-title"><h3>${escapeHtml(t("trainingMonitor"))}</h3></div>
      <p class="muted">${escapeHtml(t("noChartData"))}</p>
    </div>`;
  }
  const availableIds = new Set(series.map((item) => item.id));
  if (state.monitor.metric !== "all" && !availableIds.has(state.monitor.metric)) {
    state.monitor.metric = "all";
  }
  const selectedId = state.monitor.metric;
  const visibleSeries = selectedId === "all"
    ? series
    : series.filter((item) => item.id === selectedId);
  const smoothingPercent = Math.round(state.monitor.smoothing * 100);
  const metricOptions = [
    `<option value="all">${escapeHtml(t("metricAll"))}</option>`,
    ...series.map(
      (item) =>
        `<option value="${escapeHtml(item.id)}"${item.id === selectedId ? " selected" : ""}>${escapeHtml(item.label)}</option>`,
    ),
  ].join("");
  return `<section id="training-monitor" class="training-monitor">
    <div class="monitor-header">
      <div class="visual-title">
        <h3>${escapeHtml(t("trainingMonitor"))}</h3>
        <div class="legend">
          <span><i class="monitor-legend-smooth"></i>${escapeHtml(t("smoothing"))}</span>
          <span><i class="monitor-legend-raw"></i>raw</span>
        </div>
      </div>
      <div class="monitor-toolbar">
        <label class="field compact-field">
          <span>${escapeHtml(t("metricFilter"))}</span>
          <select id="monitor-metric">${metricOptions}</select>
        </label>
        <label class="field compact-field smoothing-field">
          <span>${escapeHtml(t("smoothing"))} <strong id="monitor-smoothing-value">${escapeHtml(smoothingPercent)}%</strong></span>
          <input id="monitor-smoothing" type="range" min="0" max="90" step="5" value="${escapeHtml(smoothingPercent)}" />
        </label>
      </div>
    </div>
    <div class="monitor-panels">
      ${visibleSeries.map((item) => renderMonitorPanel(item)).join("")}
    </div>
  </section>`;
}

function collectTrainingSeries(history) {
  const series = new Map();
  const rows = Array.isArray(history) ? history : [];
  const addPoint = (id, label, epoch, value) => {
    const number = metricNumber(value);
    if (number === null) {
      return;
    }
    if (!series.has(id)) {
      series.set(id, { id, label, points: [] });
    }
    series.get(id).points.push({ epoch, value: number });
  };
  rows.forEach((row, index) => {
    const epoch = metricNumber(row?.epoch) ?? index + 1;
    addPoint("train_loss", "train_loss", epoch, row?.train_loss);
    const validation = row?.validation_metrics || {};
    Object.entries(validation).forEach(([scaleName, metrics]) => {
      if (!metrics || typeof metrics !== "object" || Array.isArray(metrics)) {
        return;
      }
      Object.entries(metrics).forEach(([metricName, value]) => {
        const label = scaleName === "original"
          ? `val_${metricName}`
          : `val_${scaleName}_${metricName}`;
        addPoint(`val_${scaleName}_${metricName}`, label, epoch, value);
      });
    });
  });
  return Array.from(series.values())
    .filter((item) => item.points.length)
    .map((item, index) => ({
      ...item,
      color: monitorPalette[index % monitorPalette.length],
      points: item.points.sort((a, b) => a.epoch - b.epoch),
    }));
}

function renderMonitorPanel(series) {
  const stats = monitorStats(series.points);
  return `<article class="monitor-panel" style="--series-color: ${escapeHtml(series.color)}">
    <div class="monitor-panel-head">
      <h4>${escapeHtml(series.label)}</h4>
      <span>${escapeHtml(series.points.length)} step${series.points.length === 1 ? "" : "s"}</span>
    </div>
    <div class="monitor-stats">
      <span><b>${escapeHtml(t("latest"))}</b><strong>${escapeHtml(formatNumber(stats.latest))}</strong></span>
      <span><b>${escapeHtml(t("best"))}</b><strong>${escapeHtml(formatNumber(stats.best))}</strong></span>
      <span><b>${escapeHtml(t("delta"))}</b><strong>${escapeHtml(formatSignedNumber(stats.delta))}</strong></span>
    </div>
    ${renderMonitorLineChart(series)}
  </article>`;
}

function renderMonitorLineChart(series) {
  const rawPoints = series.points;
  const smoothedPoints = smoothSeries(rawPoints, state.monitor.smoothing);
  const width = 420;
  const height = 220;
  const pad = { left: 54, right: 18, top: 18, bottom: 34 };
  const values = rawPoints.map((point) => point.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || Math.max(Math.abs(max), 1);
  const yMin = min - spread * 0.08;
  const yMax = max + spread * 0.08;
  const epochMin = Math.min(...rawPoints.map((point) => point.epoch));
  const epochMax = Math.max(...rawPoints.map((point) => point.epoch));
  const x = (point, index) => {
    if (epochMax === epochMin) {
      return pad.left + (width - pad.left - pad.right) / 2;
    }
    return pad.left + ((point.epoch - epochMin) / (epochMax - epochMin)) * (width - pad.left - pad.right);
  };
  const y = (value) =>
    height - pad.bottom - ((value - yMin) / (yMax - yMin || 1)) * (height - pad.top - pad.bottom);
  const line = (points) =>
    points
      .map((point, index) => `${x(point, index)},${y(point.value)}`)
      .join(" ");
  const grid = [0, 0.5, 1]
    .map((ratio) => {
      const gy = pad.top + ratio * (height - pad.top - pad.bottom);
      const value = yMax - ratio * (yMax - yMin);
      return `<line class="grid" x1="${pad.left}" x2="${width - pad.right}" y1="${gy}" y2="${gy}" /><text x="8" y="${gy + 4}">${escapeHtml(formatNumber(value))}</text>`;
    })
    .join("");
  const dots = rawPoints
    .map((point, index) => `<circle class="monitor-dot" cx="${x(point, index)}" cy="${y(point.value)}" r="3.6">
        <title>${escapeHtml(series.label)} | epoch ${formatValue(point.epoch)}: ${formatNumber(point.value)}</title>
      </circle>`)
    .join("");
  return `<svg class="chart monitor-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(series.label)}">
    ${grid}
    <line class="axis" x1="${pad.left}" x2="${width - pad.right}" y1="${height - pad.bottom}" y2="${height - pad.bottom}" />
    <line class="axis" x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height - pad.bottom}" />
    <polyline class="line-raw" points="${line(rawPoints)}" />
    <polyline class="line-smooth" points="${line(smoothedPoints)}" />
    ${dots}
    <text x="${pad.left}" y="${height - 10}">${escapeHtml(formatValue(epochMin))}</text>
    <text x="${width - pad.right - 22}" y="${height - 10}">${escapeHtml(formatValue(epochMax))}</text>
  </svg>`;
}

function smoothSeries(points, amount) {
  const smoothing = Math.max(0, Math.min(0.95, Number(amount) || 0));
  let previous = null;
  return points.map((point) => {
    const value = previous === null
      ? point.value
      : previous * smoothing + point.value * (1 - smoothing);
    previous = value;
    return { ...point, value };
  });
}

function monitorStats(points) {
  const first = points[0]?.value ?? null;
  const latest = points[points.length - 1]?.value ?? null;
  const values = points.map((point) => point.value);
  return {
    latest,
    best: values.length ? Math.min(...values) : null,
    delta: latest === null || first === null ? null : latest - first,
  };
}

function renderTrainingChart(history) {
  const rows = (history || [])
    .map((row) => ({
      epoch: asNumber(row.epoch),
      train: asNumber(row.train_loss),
      val: asNumber(row.validation_metrics?.original?.mae),
    }))
    .filter((row) => row.epoch !== null && (row.train !== null || row.val !== null));
  if (!rows.length) {
    return `<p class="muted">${escapeHtml(t("noChartData"))}</p>`;
  }
  const width = 720;
  const height = 260;
  const pad = { left: 48, right: 22, top: 20, bottom: 36 };
  const values = rows.flatMap((row) => [row.train, row.val]).filter((value) => value !== null);
  const min = Math.min(0, ...values);
  const max = Math.max(...values, 1);
  const x = (index) => {
    if (rows.length === 1) {
      return pad.left + (width - pad.left - pad.right) / 2;
    }
    return pad.left + (index / (rows.length - 1)) * (width - pad.left - pad.right);
  };
  const y = (value) => height - pad.bottom - ((value - min) / (max - min || 1)) * (height - pad.top - pad.bottom);
  const line = (key) =>
    rows
      .map((row, index) => (row[key] === null ? null : `${x(index)},${y(row[key])}`))
      .filter(Boolean)
      .join(" ");
  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((ratio) => {
      const gy = pad.top + ratio * (height - pad.top - pad.bottom);
      const value = max - ratio * (max - min);
      return `<line class="grid" x1="${pad.left}" x2="${width - pad.right}" y1="${gy}" y2="${gy}" /><text x="8" y="${gy + 4}">${escapeHtml(formatNumber(value))}</text>`;
    })
    .join("");
  const labels = rows
    .map((row, index) => `<text x="${x(index) - 8}" y="${height - 10}">${escapeHtml(row.epoch)}</text>`)
    .join("");
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(t("trainingCurve"))}">
    ${grid}
    <line class="axis" x1="${pad.left}" x2="${width - pad.right}" y1="${height - pad.bottom}" y2="${height - pad.bottom}" />
    <line class="axis" x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height - pad.bottom}" />
    <polyline class="line-main" points="${line("train")}" />
    <polyline class="line-second" points="${line("val")}" />
    ${labels}
  </svg>`;
}

function renderForecastChart(payload) {
  const sample = Array.isArray(payload?.samples) ? payload.samples[0] : null;
  if (!sample || !Array.isArray(sample.actual) || !Array.isArray(sample.predicted)) {
    return `<p class="muted">${escapeHtml(t("noChartData"))}</p>`;
  }
  const targetName = Array.isArray(payload.target_cols) && payload.target_cols.length
    ? payload.target_cols[0]
    : "target_1";
  const rows = sample.actual
    .map((actualStep, index) => ({
      horizon: Array.isArray(payload.horizon) ? payload.horizon[index] || index + 1 : index + 1,
      actual: asNumber(Array.isArray(actualStep) ? actualStep[0] : actualStep),
      predicted: asNumber(
        Array.isArray(sample.predicted[index]) ? sample.predicted[index][0] : sample.predicted[index],
      ),
    }))
    .filter((row) => row.actual !== null || row.predicted !== null);
  if (!rows.length) {
    return `<p class="muted">${escapeHtml(t("noChartData"))}</p>`;
  }
  const width = 760;
  const height = 280;
  const pad = { left: 56, right: 28, top: 22, bottom: 42 };
  const values = rows.flatMap((row) => [row.actual, row.predicted]).filter((value) => value !== null);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || 1;
  const yMin = min - spread * 0.08;
  const yMax = max + spread * 0.08;
  const x = (index) => {
    if (rows.length === 1) {
      return pad.left + (width - pad.left - pad.right) / 2;
    }
    return pad.left + (index / (rows.length - 1)) * (width - pad.left - pad.right);
  };
  const y = (value) => height - pad.bottom - ((value - yMin) / (yMax - yMin || 1)) * (height - pad.top - pad.bottom);
  const line = (key) =>
    rows
      .map((row, index) => (row[key] === null ? null : `${x(index)},${y(row[key])}`))
      .filter(Boolean)
      .join(" ");
  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((ratio) => {
      const gy = pad.top + ratio * (height - pad.top - pad.bottom);
      const value = yMax - ratio * (yMax - yMin);
      return `<line class="grid" x1="${pad.left}" x2="${width - pad.right}" y1="${gy}" y2="${gy}" /><text x="8" y="${gy + 4}">${escapeHtml(formatNumber(value))}</text>`;
    })
    .join("");
  const labels = rows
    .map((row, index) => `<text x="${x(index) - 7}" y="${height - 12}">${escapeHtml(row.horizon)}</text>`)
    .join("");
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(t("forecastPreview"))}">
    ${grid}
    <line class="axis" x1="${pad.left}" x2="${width - pad.right}" y1="${height - pad.bottom}" y2="${height - pad.bottom}" />
    <line class="axis" x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height - pad.bottom}" />
    <polyline class="line-main" points="${line("actual")}" />
    <polyline class="line-second" points="${line("predicted")}" />
    ${labels}
    <text x="${pad.left}" y="${height - 2}">${escapeHtml(targetName)}</text>
  </svg>`;
}

function renderMetricBars(metrics) {
  const rows = ["mae", "mse", "rmse", "wape"]
    .map((name) => ({ name, value: asNumber(metrics?.[name]) }))
    .filter((row) => row.value !== null);
  if (!rows.length) {
    return `<p class="muted">${escapeHtml(t("noChartData"))}</p>`;
  }
  return renderHorizontalBars(rows, { labelWidth: 70, width: 520, heightPerRow: 38 });
}

function renderLeaderboardChart(rows, metricName) {
  const chartRows = (rows || [])
    .filter((row) => row.status === "success")
    .map((row) => ({
      name: row.model_alias || row.model_name || "-",
      value: asNumber(row.primary_metric_value ?? row[`test_${metricName}`]),
    }))
    .filter((row) => row.value !== null)
    .slice(0, 8);
  if (!chartRows.length) {
    return `<p class="muted">${escapeHtml(t("noChartData"))}</p>`;
  }
  return renderHorizontalBars(chartRows, { labelWidth: 145, width: 760, heightPerRow: 34 });
}

function renderHorizontalBars(rows, options) {
  const width = options.width;
  const height = Math.max(120, rows.length * options.heightPerRow + 36);
  const labelWidth = options.labelWidth;
  const pad = { left: labelWidth, right: 70, top: 16, bottom: 16 };
  const max = Math.max(...rows.map((row) => row.value), 1);
  const barArea = width - pad.left - pad.right;
  const rowHeight = (height - pad.top - pad.bottom) / rows.length;
  const body = rows
    .map((row, index) => {
      const y = pad.top + index * rowHeight + 7;
      const barWidth = Math.max(2, (row.value / max) * barArea);
      const barClass = index === 0 ? "bar-second" : "bar";
      return `<text x="8" y="${y + 15}">${escapeHtml(truncate(row.name, 22))}</text>
        <rect class="${barClass}" x="${pad.left}" y="${y}" width="${barWidth}" height="${Math.max(12, rowHeight - 12)}" rx="4" />
        <text x="${pad.left + barWidth + 8}" y="${y + 15}">${escapeHtml(formatNumber(row.value))}</text>`;
    })
    .join("");
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img">${body}</svg>`;
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

function renderKeyValues(values) {
  const entries = Object.entries(values || {});
  if (!entries.length) {
    return `<p class="muted">${escapeHtml(t("noRows"))}</p>`;
  }
  return `<div class="summary">${entries
    .map(
      ([key, value]) =>
        `<div class="kv"><span class="key">${escapeHtml(key)}</span><span class="value">${escapeHtml(formatValue(value))}</span></div>`,
    )
    .join("")}</div>`;
}

function metric(label, value, hint = "") {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(
    formatValue(value),
  )}</strong>${hint ? `<small>${escapeHtml(hint)}</small>` : ""}</div>`;
}

function renderJson(value) {
  return `<pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
}

function renderError(message) {
  return `<div class="error">${escapeHtml(message)}</div>`;
}

function runKey(row) {
  return `${row.experiment_name || "-"}::${row.run_id || row.compare_run_id || "latest"}`;
}

function metricFromSummary(row) {
  return asNumber(row.test_metrics?.original?.mae);
}

function asNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function metricNumber(value) {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  return asNumber(value);
}

function dateValue(value) {
  const timestamp = Date.parse(value || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function formatDate(value) {
  const timestamp = Date.parse(value || "");
  if (!Number.isFinite(timestamp)) {
    return "-";
  }
  return new Intl.DateTimeFormat(state.language === "zh" ? "zh-CN" : "en", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(timestamp));
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  if (Number.isInteger(number)) {
    return String(number);
  }
  if (Math.abs(number) >= 100) {
    return number.toFixed(2);
  }
  if (Math.abs(number) >= 10) {
    return number.toFixed(3);
  }
  return number.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatSignedNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  const sign = number > 0 ? "+" : "";
  return `${sign}${formatNumber(number)}`;
}

function formatRunType(type) {
  if (type === "train") {
    return state.language === "zh" ? "训练" : "train";
  }
  if (type === "compare") {
    return state.language === "zh" ? "对比" : "compare";
  }
  return state.language === "zh" ? "未知" : "unknown";
}

function formatValue(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function truncate(value, maxLength) {
  const text = String(value);
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setText(selector, value) {
  const element = document.querySelector(selector);
  if (element) {
    element.textContent = value;
  }
}

function getInitialPage() {
  try {
    return pageFromHash();
  } catch {
    return "overview";
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
  renderOverviewFromState();
  renderExperimentList();
  renderDetail();
}

function renderOverviewFromState() {
  const grid = document.querySelector("#overview-grid");
  if (!grid.innerHTML) {
    return;
  }
  if (state.overview.health && state.overview.datasets && state.overview.models) {
    renderOverview(state.overview.health, state.overview.datasets, state.overview.models, state.experiments);
    return;
  }
  const rows = state.experiments;
  const completeRows = rows.filter((row) => row.status === "complete");
  const trainRows = rows.filter((row) => row.run_type === "train");
  const compareRows = rows.filter((row) => row.run_type === "compare");
  const bestTrain = trainRows
    .map((row) => metricFromSummary(row))
    .filter((metricValue) => metricValue !== null)
    .sort((a, b) => a - b)[0];
  grid.innerHTML = [
    metric(t("backendStatus"), "ok", t("version")),
    metric(t("totalRuns"), rows.length, `${t("completedRuns")} ${completeRows.length}`),
    metric(t("trainRuns"), trainRows.length, t("trainTitle")),
    metric(t("compareRuns"), compareRows.length, t("compareTitle")),
    metric(t("datasetCount"), "-", t("modelCount")),
    metric(t("bestTrainMae"), bestTrain === undefined ? "-" : formatNumber(bestTrain), "MAE"),
  ].join("");
}

function applyLanguage() {
  document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
  document.title = t("documentTitle");
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    const key = element.dataset.i18n;
    if (element.tagName === "OPTION") {
      element.textContent = t(key);
    } else {
      element.textContent = t(key);
    }
  });
  document.querySelector("#language-toggle").textContent = t("languageToggle");
  document.querySelectorAll("[data-train-demo]").forEach((button) => {
    button.textContent = `${t("runPrefix")} ${button.dataset.trainDemo}`;
  });
  document.querySelectorAll("[data-compare-demo]").forEach((button) => {
    button.textContent = `${t("runPrefix")} ${button.dataset.compareDemo}`;
  });
  document.querySelector("#custom-submit").textContent = t(
    document.querySelector("#custom-submit").dataset.i18n || "customSubmitTrain",
  );
  document.querySelector("#experiment-search").placeholder =
    state.language === "zh" ? "实验名 / run_id / 类型" : "experiment / run_id / type";
  renderDatasetCatalog(state.overview.datasets || { datasets: [] });
  renderJobDemoOptions();
  renderCustomPreview();
}

function t(key) {
  return translations[state.language][key] || translations.en[key] || key;
}

function loadFavorites() {
  try {
    return JSON.parse(window.localStorage.getItem("favoriteRuns") || "[]");
  } catch {
    return [];
  }
}

function saveFavorites() {
  try {
    window.localStorage.setItem("favoriteRuns", JSON.stringify([...state.favorites]));
  } catch {
    // Ignore localStorage failures in restricted browser contexts.
  }
}

function loadUserDatasets() {
  try {
    const rows = JSON.parse(window.localStorage.getItem("userDatasets") || "[]");
    return Array.isArray(rows) ? rows : [];
  } catch {
    return [];
  }
}

function saveUserDatasets() {
  try {
    window.localStorage.setItem("userDatasets", JSON.stringify(state.datasetCatalog.customRows));
  } catch {
    // Ignore localStorage failures in restricted browser contexts.
  }
}
