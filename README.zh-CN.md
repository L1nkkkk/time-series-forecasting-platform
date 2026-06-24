# 时间序列预测快速开发平台

本项目是一个面向课程验收和现场展示的时间序列深度学习模型快速开发平台。它以 Python、PyTorch、FastAPI、CLI 和本地 Dashboard 为基础，串起数据集准备、模型训练、模型对比、排行榜、预测样本、任务队列和实验产物管理。

项目不复制 BasicTS 代码，但采用类似的工程分层思想：数据集、模型、指标、训练器、对比 runner、配置、API 和 UI 都保持相对独立，并通过注册表和配置文件扩展。

## 课程题目需求对齐

本项目对应课程截图中的“题目 3：面向时间序列的深度学习模型快速开发平台”。截图要求开发一个综合性的时间序列预测平台，提供集中式数据集库、数据集访问接口、多种先进预测模型和快速训练框架，帮助研究人员和行业专家快速构建、训练和比较模型。

当前实现围绕这些要求展开：数据集库负责公开数据和本地 CSV 管理；FastAPI 与 CLI 提供数据准备、训练、对比和产物访问接口；`MODEL_REGISTRY` 集成 linear、DLinear、NLinear、PatchTST 以及 RNN/GRU/LSTM/TCN/Transformer 等模型；训练与 compare runner 输出 checkpoint、leaderboard、训练曲线、预测样本和报告材料。更详细的需求映射见 `docs/course_requirements.zh-CN.md`。

## 核心能力

- 数据集库：内置公开数据集目录、本地 CSV 数据集、用户自定义 CSV 元数据。
- 可准备数据资产：支持对 ETTh1、ETTm1、Exchange、Traffic 等公开数据集执行一键准备，写入 `data/external/` 和 `data/cache/`。
- 模型库：支持 naive、moving average、seasonal naive、linear、DLinear、NLinear、PatchTST、MLP、N-BEATS、RNN、GRU、LSTM、TCN、Transformer。
- 快速训练：配置驱动训练，支持 checkpoint、模型导出、原始尺度指标、预测样本和环境快照。
- 训练控制：支持 best validation checkpoint、early stopping、gradient clipping、step/cosine 学习率调度。
- 多模型对比：生成 `leaderboard.json` 和 `leaderboard.csv`，便于横向比较模型。
- 本地 Dashboard：提供概览、数据集库、实验结果、自定义实验、任务页面。
- 异步任务：支持本地 train/compare job、任务状态、日志、结果查询。
- 产物管理：通过 artifact manifest 安全发现和下载 JSON、YAML、CSV、log、model export 等产物。

## 快速开始

安装开发依赖：

```bash
py -m pip install -e ".[dev]"
```

运行基础训练：

```bash
py -m ts_platform.cli.main train --config configs/examples/simple_forecast.yaml
```

运行模型库对比：

```bash
py -m ts_platform.cli.main compare --config configs/examples/compare_model_zoo.yaml
```

准备公开数据集并运行“理想目标 Demo”：

```bash
py -m ts_platform.cli.main prepare-dataset --dataset etth1
py -m ts_platform.cli.main compare --config configs/examples/ideal_target_demo.yaml
```

启动本地 Dashboard：

```bash
uvicorn ts_platform.api.app:create_app --factory --port 8001
```

打开：

```text
http://127.0.0.1:8001/ui
```

## 推荐展示流程

1. 打开 Dashboard 的“概览”页，展示后端状态、数据集数量、模型数量和最近实验。
2. 进入“数据集库”，选择 `etth1`，点击“准备数据集”，展示公开数据从目录条目变成本地可训练 CSV。
3. 进入“任务”页，点击“理想目标 Demo”，提交 `ideal_target_demo` 异步对比任务。
4. 任务完成后进入“实验结果”，查看 leaderboard、训练曲线、预测样本和 artifacts。
5. 对比 linear、DLinear、NLinear、PatchTST，说明平台支持快速模型切换和统一指标评估。

## 常用命令

```bash
py -m ts_platform.cli.main list-datasets
py -m ts_platform.cli.main prepare-dataset --dataset etth1
py -m ts_platform.cli.main show-dataset-cache
py -m ts_platform.cli.main clear-dataset-cache --dataset etth1
py -m ts_platform.cli.main list-models
py -m ts_platform.cli.main show-results --experiment ideal_target_demo --run latest
py -m ts_platform.cli.main show-leaderboard --experiment ideal_target_demo --run latest
```

## 项目边界

当前版本定位为课程理想版和本地演示平台，不是多租户生产 SaaS。它暂不包含多用户账号、细粒度权限、Redis/Celery 生产队列和云端部署监控。这些内容保留为后续生产平台化方向。

## 文档入口

- Dashboard 中文演示指南：`docs/dashboard_demo.zh-CN.md`
- 课程题目需求对齐：`docs/course_requirements.zh-CN.md`
- API 设计：`docs/api_design.md`
- 数据格式：`docs/data_format.md`
- 数据集目录：`docs/dataset_catalog.md`
- 模型库说明：`docs/model_zoo.md`
- 交付报告提纲：`docs/final_report_outline.md`
