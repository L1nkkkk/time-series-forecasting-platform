# Dashboard 中文演示指南

本指南用于课程答辩、现场演示和项目验收。Dashboard 是 FastAPI 提供的本地 Demo UI，目标是把数据集准备、模型训练、模型对比、任务队列、排行榜、预测样本和产物管理串成一个完整展示闭环。

## 课程题目视角

课程截图中的题目是“面向时间序列的深度学习模型快速开发平台”，开发环境为 Python、PyTorch，要求熟悉 Python、Git、PyTorch。演示时可以先说明：平台不是单个预测模型，而是围绕题目描述中的“集中数据集库、数据访问接口、先进预测模型、快速训练框架和模型比较”构建的一站式系统。

本 Dashboard 正好承接这条主线：

- “数据集库”页面对应集中式数据集库和数据访问接口。
- “自定义实验”和“任务”页面对应快速构建和训练模型。
- “监控”页面对应训练过程可视化、ETA、日志和实时 loss/MAE 曲线。
- “实验结果”页面对应模型比较、leaderboard、预测样本和产物导出。

完整需求映射见 `docs/course_requirements.zh-CN.md`。

## 启动方式

在项目根目录安装依赖并启动 API：

```bash
py -m pip install -e ".[dev]"
uvicorn ts_platform.api.app:create_app --factory --port 8001
```

打开：

```text
http://127.0.0.1:8001/ui
```

如果端口被占用，可以把 `8001` 换成其他端口。

## 页面结构

- 概览：查看后端状态、模型数量、数据集数量和最近实验。
- 数据集库：浏览公开数据集和用户 CSV，准备公开数据集，查看来源、列信息和本地路径。
- 实验结果：筛选训练/对比 run，查看指标、训练曲线、leaderboard、预测样本和 artifacts。
- 自定义实验：在 UI 中配置数据、模型、训练参数和评估指标，运行单模型训练或多模型对比。
- 任务：提交白名单 demo，查看异步 job 状态、日志和结果。

## 推荐展示流程

1. 进入“概览”，点击“刷新全部”，说明平台已经接通数据集、模型、实验和任务接口。
2. 进入“数据集库”，搜索 `etth1`，查看公开来源、领域、频率、目标列和特征列。
3. 点击“准备数据集”，平台会下载 ETTh1 并写入 `data/external/etth1/v1/ETTh1.csv`，同时生成 cache manifest、prepared catalog 和默认训练配置。
4. 进入“任务”，点击“理想目标 Demo”。后端会确保 ETTh1 已准备，然后提交 `ideal_target_demo` compare job。
5. 等待任务完成后进入“实验结果”，打开 `ideal_target_demo`，查看 linear、DLinear、NLinear、PatchTST 的 leaderboard。
6. 切换到训练 run 详情，展示训练曲线、best metric、best checkpoint、forecast samples 和 artifacts。
7. 点击“导出报告”，生成 Markdown 演示材料。

## CLI 等价命令

Dashboard 的核心演示也可以用 CLI 复现：

```bash
py -m ts_platform.cli.main prepare-dataset --dataset etth1
py -m ts_platform.cli.main compare --config configs/examples/ideal_target_demo.yaml
py -m ts_platform.cli.main show-leaderboard --experiment ideal_target_demo --run latest
py -m ts_platform.cli.main show-artifacts --experiment ideal_target_demo --run latest
```

## 讲解要点

- 数据层不是只有链接目录：首批公开数据集支持一键准备、本地缓存、manifest 和训练配置生成。
- 模型层不是单一 baseline：平台同时支持统计基线、经典深度模型和 DLinear/NLinear/PatchTST 等现代模型。
- 训练层不是黑盒脚本：训练过程会记录 history、validation metrics、best checkpoint、forecast samples 和环境信息。
- 对比层统一输出 leaderboard：不同模型在相同数据、窗口、指标下比较，便于做课程展示。
- Dashboard 是本地演示界面：当前不包含多用户账号和生产级权限，这属于后续生产平台化方向。

## 常见问题

- 如果点击“理想目标 Demo”时网络不可用，可以先用已准备好的本地 CSV，或改用 `compare_model_zoo` 合成数据对比演示。
- 如果 CUDA 不可用，请选择 CPU；后端会返回明确错误，避免出现不透明的 500。
- 如果任务运行时间较长，可以先运行 CLI 生成结果，再在 Dashboard 的“实验结果”页刷新查看。
