# 时间序列预测平台

本仓库是一个配置驱动的时间序列预测平台 MVP，用于展示从数据集、模型、
训练、评估、对比实验、产物管理到本地 Dashboard Demo 的完整工程闭环。

项目不复制 BasicTS 代码，但借鉴了类似的工程分层思想：数据集、缩放器、
模型、指标、Runner、配置和 API 都保持相对独立，并通过注册表和小接口扩展。

## 项目能力概览

当前版本支持：

- 合成时间序列数据集。
- 本地 CSV 时间序列数据集，支持按时间顺序切分。
- Naive、Moving Average、Seasonal Naive、Linear、MLP、RNN、GRU、LSTM、
  TCN 等预测模型。
- Standard 和 MinMax 缩放器。
- MAE、MSE、RMSE、MAPE、WAPE 等指标。
- 原始尺度评估指标，按需保存缩放空间指标。
- 训练配置快照、checkpoint、运行环境元数据和 `results.json`。
- 多模型对比实验，输出 `leaderboard.json` 和 `leaderboard.csv`。
- Feature-aware CSV 训练和对比实验，支持目标列与外生特征列分离。
- Artifact manifest，用于发现训练和对比产物。
- 安全的 manifest-based artifact 读取，默认禁止 checkpoint 下载。
- CLI、本地 FastAPI Demo API、本地异步 job prototype。
- 轻量本地 Dashboard UI，用于答辩或现场展示。

## 安装

需要 Python 3.10 或更新版本。

```bash
py -m pip install -e ".[dev]"
```

在 macOS 或 Linux 上通常使用：

```bash
python -m pip install -e ".[dev]"
```

## 快速开始

运行合成数据训练示例：

```bash
python -m ts_platform.cli.main train --config configs/examples/simple_forecast.yaml
```

运行本地 CSV 训练示例：

```bash
python -m ts_platform.cli.main train --config configs/examples/csv_forecast.yaml
```

运行 feature-aware CSV 训练示例：

```bash
python -m ts_platform.cli.main train --config configs/examples/csv_feature_forecast.yaml
```

运行多模型对比：

```bash
python -m ts_platform.cli.main compare --config configs/examples/compare_feature_forecast.yaml
```

查看对比结果和产物：

```bash
python -m ts_platform.cli.main show-results --experiment compare_feature_forecast --run latest
python -m ts_platform.cli.main show-leaderboard --experiment compare_feature_forecast --run latest
python -m ts_platform.cli.main show-artifacts --experiment compare_feature_forecast --run latest
python -m ts_platform.cli.main show-artifact --experiment compare_feature_forecast --run latest --artifact leaderboard_json
```

## Dashboard Demo

启动本地 FastAPI：

```bash
uvicorn ts_platform.api.app:create_app --factory
```

打开：

http://127.0.0.1:8000/ui

Dashboard 默认显示中文，右上角可以点击 `English` 切换到英文界面。

推荐演示流程：

1. 在 Overview 中点击 Refresh，展示后端状态、版本、数据集数量、模型列表和实验数量。
2. 运行 `csv_feature_forecast`，展示 feature-aware 单次训练。
3. 运行 `compare_feature_forecast`，展示多模型对比。
4. 查看 leaderboard 中的 `feature_aware`、`input_dim`、`target_dim`、
   `feature_dim`、`target_cols`、`feature_cols`。
5. 在 Artifacts / Leaderboard Preview 面板中加载 leaderboard 和 artifacts。
6. 可选展示 Jobs 面板，说明它是本地异步任务 prototype。

现场展示时，如果时间有限，可以提前运行：

```bash
python -m ts_platform.cli.main compare --config configs/examples/compare_feature_forecast.yaml
```

然后在 Dashboard 中直接 Load Leaderboard。

## 配置说明

训练和对比都通过 YAML 或 JSON 配置驱动。训练配置主要包含：

- `experiment`：实验名称、输出目录、随机种子、覆盖策略。
- `data`：数据集名称、窗口长度、预测长度、batch size、切分比例、缩放器和数据参数。
- `model`：模型名称和模型参数。
- `training`：epoch、学习率、设备、优化器、loss 和 checkpoint 策略。
- `evaluation`：评估指标和是否保存缩放空间指标。

示例配置：

- `configs/examples/simple_forecast.yaml`
- `configs/examples/csv_forecast.yaml`
- `configs/examples/csv_feature_forecast.yaml`
- `configs/examples/compare_feature_forecast.yaml`

## Feature-aware 训练

CSV 数据支持 `target_cols` 和 `feature_cols`：

- `target_cols` 是需要预测和评估的目标列。
- `feature_cols` 是只作为输入历史使用的外生特征列。
- 模型输入维度 `input_dim = target_dim + feature_dim`。
- 评估指标仍然只针对目标列，避免特征列污染预测指标。

`csv_feature_forecast.yaml` 和 `compare_feature_forecast.yaml` 都可用于展示该能力。

## 安全边界

本项目是本地研究和演示 MVP，不是多租户生产系统。当前安全边界包括：

- API 会把训练和对比输出目录约束到固定的 runs root。
- `experiment.name` 和 `run_id` 必须是安全路径组件。
- Demo endpoint 只允许运行白名单示例配置，不接受任意路径。
- Artifact 下载只能读取 manifest 中登记的安全文件。
- 默认禁止下载 checkpoint。
- Dashboard 是本地 demo UI，不是生产级 Web UI。

## 常用质量门禁

```bash
python -m pytest
ruff check .
ruff format --check .
mypy src
```

完整发布 smoke gate 见：

- `docs/release_checklist.md`
- `CONTRIBUTING.md`

## 文档入口

- 英文 README：`README.md`
- Dashboard Demo：`docs/dashboard_demo.md`
- Demo Guide：`docs/demo_guide.md`
- Release Checklist：`docs/release_checklist.md`
- Roadmap：`docs/roadmap.md`

## 当前状态

项目已进入 Final Freeze。建议后续只做：

- UI 小 bug 修复。
- Demo 文案和展示流程打磨。
- 答辩报告和最终材料准备。
- 发布前质量门禁复核。
