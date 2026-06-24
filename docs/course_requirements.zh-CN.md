# 课程题目需求对齐说明

本文档根据课程选题截图整理项目设计需求，用于答辩、验收和 GitHub 项目说明。

## 题目信息

- 题目：题目 3：面向时间序列的深度学习模型快速开发平台
- 种类：开发
- 项目开发环境：Python、PyTorch
- 要求：熟悉 Python、Git、PyTorch
- 项目等级：2 级

## 题目目标理解

截图中的项目描述强调：时间序列预测广泛应用于金融、气象、工业生产、医疗健康等行业；随着大数据和深度学习的发展，预测准确性和效率需要进一步提升。项目应帮助研究人员和行业专家快速构建、训练和比较时间序列预测模型，并提供集中式数据集库、数据集访问接口、先进预测模型和快速训练框架。

因此，本项目不是只写一个单模型训练脚本，而是设计为一个面向课程展示的一站式平台：

- 数据从集中式目录进入平台。
- 模型通过注册表和配置快速切换。
- 训练过程可复现、可监控、可导出。
- 多模型结果可以在同一指标体系下比较。
- Dashboard 和 CLI 都能覆盖核心流程，方便现场展示和验收。

## 设计需求映射

| 课程描述中的需求 | 平台设计响应 | 当前实现 |
| --- | --- | --- |
| 集中的数据集库，涵盖不同领域和行业的开源时间序列数据 | 建立 `DatasetCatalog`，用 YAML 管理公开数据集、本地 CSV 和用户自定义数据集元数据 | `configs/datasets/public_time_series.yaml`、数据集页面、`list-datasets` |
| 提供数据集访问接口 | 同时提供 CLI 与 FastAPI 数据接口，支持查询、体检、准备和缓存管理 | `/datasets`、`/datasets/{name}/prepare`、`prepare-dataset`、`show-dataset-cache` |
| 集成多种先进时间序列预测模型 | 使用 `MODEL_REGISTRY` 管理模型，统一输入输出接口，便于快速切换 | `linear`、`dlinear`、`nlinear`、`patchtst`、RNN/GRU/LSTM/TCN/Transformer 等 |
| 快速训练模型的框架 | 以配置驱动训练，支持 checkpoint、best metric、early stopping、学习率调度、梯度裁剪和导出产物 | `TrainingConfig`、`Trainer`、`results.json`、`artifacts.json` |
| 支持快速构建、训练和比较时间序列预测模型 | 提供 compare runner 和异步 job，将多个模型放在同一数据集、窗口和指标下比较 | `CompareRunner`、`leaderboard.json`、`leaderboard.csv`、Jobs/Monitor 页面 |
| 一站式解决方案，促进时间序列分析创新应用 | Dashboard 串起数据准备、模型训练、运行监控、排行榜、预测样本和报告导出 | `/ui` 的 Overview、Datasets、Results、Custom、Monitor、Jobs 页面 |

## 平台边界

本阶段定位为“课程理想版”和本地演示平台，重点满足截图中的开发类题目目标。它不追求多租户生产 SaaS，因此暂不包含多用户账号、细粒度权限、Redis/Celery 生产队列、云端部署和完整生产监控。这些内容作为后续生产平台版扩展方向。

## 验收演示路径

推荐按以下顺序讲解，能够直接对应截图中的设计要求：

1. 数据集库：打开 Dashboard 的“数据集库”，展示 `etth1`、`ettm1`、`exchange_rate_lai`、`traffic_lai` 等公开数据集条目。
2. 数据访问接口：点击“准备数据集”，或执行 `py -m ts_platform.cli.main prepare-dataset --dataset etth1`，展示公开数据被准备成本地可训练 CSV。
3. 快速训练：提交 `ideal_training_30min_demo`，进入“监控”页查看实时进度、ETA、loss/MAE 折线图和日志。
4. 多模型比较：提交 `ideal_target_demo`，比较 `linear`、`dlinear`、`nlinear`、`patchtst`。
5. 结果闭环：进入“实验结果”，展示 leaderboard、训练曲线、预测样本、artifacts 和 Markdown 报告导出。

## 技术选型说明

- Python：负责 CLI、配置解析、数据处理、训练流程和测试。
- PyTorch：负责深度学习模型定义、训练、checkpoint 和推理导出。
- FastAPI：负责本地 API 与 Dashboard 后端。
- 静态 Dashboard：负责课程展示，不引入复杂前端工程栈，降低部署和演示成本。
- Git/GitHub：用于版本管理、PR 追踪、测试记录和课程交付。
