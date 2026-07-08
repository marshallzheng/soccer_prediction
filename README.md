# corner-predictor

实时计算足球比赛"全场角球数大于/小于阈值"概率的 MVP 系统。当前使用模拟比赛数据源（`MockMatchSimulator`）驱动，尚未接入真实付费数据API。

## 核心思路

- 已发生的角球数是确定值；剩余时间的角球产出率根据比赛状态（近期攻势、控球率、比分差、剩余时间）动态估计。
- 用时变泊松/负二项分布对"剩余角球数"建模，解析计算 P(总角球数 > 阈值) 及完整概率分布（PMF）。
- 每场比赛的逐拍数据会持久化到 SQLite（`data/corner_predictor.db`），为未来训练机器学习模型积累数据。

## 运行

```bash
uv sync
uv run uvicorn corner_predictor.main:app --reload
```

打开 `http://localhost:8000/`，点击"开始模拟比赛"即可看到实时概率曲线和角球数概率分布。

## 测试

```bash
uv run pytest
```

## Phase 2（未实现，路线图）

积累足够真实比赛的 `MatchTick` / `MatchEventRecord` 数据后，可离线训练梯度提升模型（如 LightGBM/XGBoost），并实现与 `CornerProbabilityModel`（`src/corner_predictor/model/probability.py`）相同的接口，通过 `get_model()` 工厂函数按配置切换 —— 数据源、特征引擎、API、前端均无需改动。

真实数据源接入方式：实现 `MatchDataSource` 协议（`src/corner_predictor/data_sources/base.py`），仿照 `MockMatchSimulator` 的形状对接具体供应商的实时API。
