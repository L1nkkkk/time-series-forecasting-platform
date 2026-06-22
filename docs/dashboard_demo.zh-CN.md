# Dashboard 中文演示指南

本文档用于课程答辩、现场演示或项目验收。Dashboard 是本地 FastAPI 提供的轻量 Demo UI，不是生产级 Web 前端；它的目标是把数据集、模型训练、对比实验、任务队列、结果管理、训练曲线和报告导出串成一个可展示闭环。

## 启动方式

在项目根目录安装开发依赖后启动 API：

```bash
py -m pip install -e ".[dev]"
uvicorn ts_platform.api.app:create_app --factory
```

打开：

```text
http://127.0.0.1:8000/ui
```

如果本机端口被占用，可以指定其他端口，例如：

```bash
uvicorn ts_platform.api.app:create_app --factory --port 8001
```

此时打开：

```text
http://127.0.0.1:8001/ui
```

界面默认中文，右上角 `English` 可以切换英文。

## 页面结构

顶部导航把原来的长页面拆成 5 个页面：

- `概览`：展示后端状态、运行数量、模型数量、数据集数量和最佳训练指标。
- `数据集`：展示内置公开数据集和用户补充的本地 CSV 数据集。
- `实验结果`：筛选、收藏、查看训练/对比 run，展示训练曲线、预测图、leaderboard 和产物。
- `自定义实验`：在 UI 中配置数据、模型、训练参数和评估指标，直接运行单模型训练或多模型对比。
- `任务`：运行白名单 demo，或把 demo 配置提交为本地异步 job，并查看 job 状态、结果和日志。

## 推荐演示流程

1. 打开 Dashboard，先进入 `概览`，点击 `刷新全部`。
2. 进入 `数据集`，用搜索框或领域下拉缩小范围，再从 `选择数据集` 下拉中选一个数据集。
3. 在数据集详情卡中讲清楚：领域、类型、频率、目标列、特征列、来源和本地路径。
4. 点击 `使用`，跳到 `自定义实验`，确认数据路径和列配置已自动填入。
5. 在 `自定义实验` 中选择模型、epoch、学习率和指标，运行自定义训练或自定义对比。
6. 进入 `任务`，用 `异步演示任务` 下拉选择训练或对比 demo，点击 `提交任务`。
7. 刷新 job 列表，展示 job_id、状态、类型和日志/结果查询。
8. 进入 `实验结果`，选择一个训练 run，查看 `训练监控` 中的 `train_loss`、`val_mae`、`val_mse` 等曲线。
9. 对比 run 可查看 leaderboard，重点解释 `feature_aware`、`input_dim`、`target_dim`、`feature_dim`、`target_cols`、`feature_cols`。
10. 在结果详情中查看 Artifacts，最后点击 `导出报告` 下载 Markdown 汇总。

## 训练曲线演示

如果想让 `loss` 曲线更明显，可以跑几组多 epoch 训练。最简单的方式是在 `自定义实验` 页面：

- 数据源选择 `Synthetic Data`。
- 模型选择 `MLP`、`Linear` 或 `GRU`。
- `Epochs` 设置为 6 到 10。
- 评估指标选择 `mae`、`mse`、`rmse`、`wape`。
- 提交训练后自动跳到 `实验结果`。

结果详情中的 `训练监控` 会显示类似 W&B 的多面板曲线：

- `train_loss`：训练损失随 epoch 变化。
- `val_mae` / `val_mse` / `val_rmse` / `val_wape`：验证集指标变化。
- `latest`：最新值。
- `best`：当前最优值。
- `delta`：首尾变化。
- 平滑滑块可以调整曲线平滑程度。

也可以使用 API 快速生成 loss 演示 run：

```powershell
$script = @'
import json
import urllib.request

url = "http://127.0.0.1:8000/experiments/train"

def config(name, model_name, params):
    return {
        "experiment": {"name": name, "output_dir": "runs", "seed": 42, "overwrite": True},
        "data": {
            "name": "synthetic",
            "input_len": 18,
            "output_len": 4,
            "batch_size": 8,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "scaler": {"name": "standard"},
            "params": {"length": 180, "num_features": 2, "noise_std": 0.03},
        },
        "model": {"name": model_name, "params": params},
        "training": {
            "epochs": 8,
            "learning_rate": 0.01,
            "device": "cpu",
            "optimizer": "adam",
            "loss": "mse",
            "checkpoint_every": 1,
        },
        "evaluation": {"metrics": ["mae", "mse", "rmse", "wape"], "include_scaled_metrics": True},
    }

for payload in [
    config("loss_demo_mlp", "mlp", {"hidden_sizes": [32, 16], "dropout": 0.0}),
    config("loss_demo_linear", "linear", {}),
    config("loss_demo_gru", "gru", {"hidden_size": 12, "num_layers": 1, "dropout": 0.0}),
]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
    history = result["history"]
    print(result["experiment_name"], result["run_id"], history[0]["train_loss"], "->", history[-1]["train_loss"])
'@
$script | py -
```

运行后回到 Dashboard，点击 `刷新全部`，在 `实验结果` 中搜索或选择 `loss_demo_*`。

## 数据集管理

数据集页不会一次性展开所有条目，而是使用：

- 搜索框：按名称、领域、类型、描述、来源过滤。
- 领域筛选：按行业或场景缩小数据集范围。
- 数据集下拉：只选择一个当前关注的数据集。
- 详情卡：集中展示来源、路径、列信息和操作按钮。

用户可以补充本地 CSV：

1. 填写数据集名、领域、CSV 路径、时间列、目标列、特征列、频率和来源。
2. 或点击 `选择 CSV 文件`，由浏览器文件选择器填入本地文件。
3. 点击 `保存并使用` 后，数据集会加入下拉，并自动填入自定义实验表单。

## 自定义实验

自定义实验支持两种模式：

- `单模型训练`：选择一个模型，产生一个训练 run。
- `模型对比`：选择多个模型，产生 compare run 和 leaderboard。

常用参数：

- 数据源：合成数据或 CSV。
- 模型：Linear、MLP、RNN、GRU、LSTM、TCN、Transformer、N-BEATS 等。
- 训练：epoch、learning rate、batch size、optimizer、loss。
- 评估：MAE、MSE、RMSE、MAPE、WAPE，是否保存 scaled metrics。

提交前右侧会显示配置预览，便于解释“UI 实际生成的是配置驱动训练”。

## 任务页

任务页包含两类能力：

- 同步 demo：直接运行白名单训练/对比配置，完成后跳到结果页。
- 异步 demo job：从下拉选择训练或对比 demo，提交到 `/jobs` 本地任务接口。

异步 job 可查看：

- `job_id`
- `job_type`
- `status`
- `created_at`
- `run_id`
- `error`
- job result
- job logs

该 job 系统是本地 prototype，用于展示队列和任务状态管理，不是生产分布式调度系统。

## 安全边界

Dashboard 和 API 保持以下安全约束：

- demo endpoint 只允许运行白名单配置。
- API 会把输出目录限制在固定 `runs` root。
- `experiment.name`、`run_id`、`job_id` 必须是安全路径组件。
- artifact 下载只能读取 manifest 中登记的 JSON、YAML、CSV、log 等安全文件。
- checkpoint 默认禁止下载。
- 用户补充数据集只保存元数据和本地路径，不会绕过路径安全检查。

## 质量门禁

合并前建议运行：

```bash
py -m pytest
py -m mypy src
py -m ruff check .
py -m ruff format --check .
node --check src/ts_platform/api/static/app.js
```
