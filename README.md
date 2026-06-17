# Quant Research Lab

一个面向 **A 股**、强调**研究纪律**的轻量级量化分析框架。它把 `docs/strategy-assessment.md` 中评审过的学习路线（"阶段 1：技能与工具搭建"）落地成**可运行、可测试**的代码。

> 设计哲学：**先学会不骗自己，再谈收益。** 框架里所有"赚钱"的功能都被"防过拟合 / 真实成本 / 样本外验证"包着。

## 它解决什么

学量化最常见的失败不是写不出策略，而是写出一个**只在历史上好看**的策略。本框架把容易被跳过的环节做成默认动作：

- **真实交易成本**：印花税、佣金、滑点、T+1、涨跌停撮合，全部进回测。
- **样本外纪律**：内置 train/test 切分与 walk-forward 滚动验证，让你一眼看出策略是不是过拟合。
- **开箱即用**：不依赖任何付费数据源，内置可复现的合成行情；同时提供 AKShare 适配器接真实数据。

## 安装

```bash
pip install -e .            # 核心：仅需 pandas / numpy
pip install -e ".[data]"    # 可选：AKShare 真实数据
pip install -e ".[viz]"     # 可选：matplotlib 画图
```

## 60 秒跑通（Gate 1 Demo）

```bash
python examples/toy_strategy.py            # 合成行情，离线可跑，演示完整方法论
python examples/real_data_backtest.py      # 真实行情：趋势跟踪 vs 买入持有，并讲清"为什么有用"
python examples/compare_strategies.py      # 趋势 vs 突破 vs 均值回归 vs 买入持有 横向对比
python examples/universe_backtest.py ma    # 多标的全池回测：跨标的稳健性（防过拟合真功夫）
python examples/check_symbol.py 600519     # 自检某代码能否拿到真实行情
python examples/build_dataset.py 2015-01-01 2024-12-31 1.0 yahoo   # 数据储备 + 质量报告
python examples/sync_offline.py 2015-01-01 2024-12-31 1.0 yahoo    # 增量同步到离线 parquet 数据仓
streamlit run dashboard/app.py             # Web 看板（需 pip install -e ".[dashboard]"）
```

### Web 看板（Streamlit + Plotly）

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```
左侧选标的/策略/参数，右侧顶部指标卡 + 净值/回撤/仓位/成交点多图，浏览器打开、布局紧凑。逻辑全在 `dashboard/core.py`（纯函数、可单测），UI 层 `dashboard/app.py` 很薄——要加指标监控只在 core 加图函数即可。

### 财务数据（point-in-time / 按公告日，防前视）

财报回测最大的坑是前视偏差：一季报报告期 3/31，但往往 4 月底才公告。本框架对每条财报都按**真实公告日**对齐，查询只返回"截至某日已公告"的数据。

```python
from quantlab.data import point_in_time
# 只返回截至该日已公告的、每只股票最新一期财报——绝不前视
snap = point_in_time("2024-06-15", symbols=["600519", "000858"])
print(snap[["symbol", "report_period", "announce_date", "announce_is_real", "net_profit", "roe"]])
```

数据源分工（均实测可用，与被墙的行情接口不同源）：
- **财务数字**：东财 `stock_yjbb_em(季度)`——一次调用覆盖全市场所有公司。
- **公告日**：巨潮 cninfo 定期报告**首发公告时间**（权威 PIT 基准）。
- **兜底**：cninfo 缺失时用**法定披露截止日**（年报/Q1≤4-30、半年报≤8-31、Q3≤10-31），保守、绝不前视。

```bash
python examples/download_fundamentals.py 2015 2024        # 财务数字（全公司，按季度）
python examples/download_announce_dates.py 20150101 20251231 0.2 all   # 全A股真实公告日
python examples/build_fundamentals_full.py 2015 2024      # 一键回补：数字 + 公告日 + PIT
```

> ⚠ 注意：`stock_yjbb_em` 自带的「最新公告日期」是该公司**最近一次披露日**、不是本期报告首发日，用于 PIT 会前视——本框架已弃用，改由 cninfo 取真实首发日。

### 离线数据仓

```python
from quantlab.data import load_prices, list_offline
# 先 sync_offline 把真实行情增量累积成 parquet（断网可用、长期累积）
prices = load_prices("600519", "2018-01-01", "2023-12-31", source="offline")
print(list_offline())   # 离线仓清单：每只的覆盖区间/天数
```
离线仓默认在 `~/.local/share/quantlab/offline`（可用环境变量 `QUANTLAB_OFFLINE` 改）。`sync_offline` 可反复跑、逐步拉长区间，数据按日期去重累积——慢慢攒成项目的"高质量数据总量"。

内置策略（`quantlab.strategies`）：`MACrossStrategy`（均线趋势）、`DonchianBreakoutStrategy`（通道突破）、`BollingerReversionStrategy`（均值回归）、`BuyHoldStrategy`（基准）。它们刻意覆盖**趋势**与**反转**两类逻辑——演示"没有万能策略，只有匹配市场状态的策略"。

它会完成一条完整链路：**数据 → 清洗 → 信号 → 带真实成本的回测 → 样本外验证**，并打印一份诚实的体检报告，包括"为什么这个结果可能是过拟合"。

### 真实数据流

`load_prices(..., source="auto")` 会按 **AKShare → Yahoo Finance → 合成行情** 的顺序自动回落，开箱即能接真实行情（Yahoo 适配器仅用标准库，自动复权）：

```python
from quantlab.data import load_prices
prices = load_prices("600519", "2018-01-01", "2023-12-31", source="yahoo")  # 真实茅台日线
print(prices.attrs["source"], prices.attrs["is_real"])  # 'yahoo' True —— 一眼分清真实/合成
```

### 能测哪些 A 股？

用 `python examples/check_symbol.py <代码...>` 可随时自检某只股票能否拿到**真实**行情。各板块实测覆盖：

| 板块 | 代码示例 | Yahoo 真实数据 |
|---|---|---|
| 沪市主板 | 600519 / 601398 | ✅ |
| 深市主板 / 中小 | 000001 / 002594 | ✅ |
| 创业板 | 300750 | ✅ |
| 科创板 | 688981 | ✅ |
| 沪/深 B 股 | 900xxx / 200xxx | ✅ |
| **北交所** | 830799 / 920xxx | ❌ Yahoo 不收录，需用 AKShare（其数据源可访问时） |

也就是说：**沪深两市全部（主板/中小/创业板/科创板，约 5000+ 只）开箱即可真实回测**；北交所（约 250 只）需走 AKShare。`load_prices(..., source="auto")` 真实源全部失败时会回落到合成数据，并通过 `df.attrs["is_real"]=False` 明确告诉你"这不是真实行情"，避免把合成数据当真实结论。

一次真实回测（茅台 2018–2023，含 A 股真实成本）得到的**诚实**结论：

| 策略 | 累计收益 | 年化波动 | 夏普 | 最大回撤 |
|---|---|---|---|---|
| MA(20,60) 趋势跟踪 | +110% | 21.4% | 0.71 | **-35.2%** |
| 买入持有 | +173% | 31.0% | 0.72 | -47.0% |

**为什么这个策略有用**：在茅台这种强趋势单票上，趋势跟踪并不会赢在收益——它赢在**回撤控制**。2021-02→2022-10 茅台从高点回撤 47% 的那段，策略平均仓位只有 47%（趋势转弱已离场），把最大回撤从 -47% 压到 -35%，波动从 31% 降到 21%，而夏普基本持平。**用差不多的风险调整后收益，换来小得多的痛苦**——这才是它的价值，而非"跑赢大盘"的幻觉。其经济学根基是动量/趋势异象（行为上的反应不足 + 羊群效应让趋势持续），且一次完整买卖成本仅约 0.20%，远小于它要捕捉的波幅。

```python
from quantlab.data import load_prices
from quantlab.strategies import MACrossStrategy
from quantlab.costs import AShareCostModel
from quantlab.backtest import Backtester

prices = load_prices("600519", start="2018-01-01", end="2023-12-31")  # 离线时自动用合成数据
strat   = MACrossStrategy(fast=10, slow=30)
costs   = AShareCostModel()                # A 股摩擦默认值
bt      = Backtester(cost_model=costs)

result = bt.run(prices, strat)
print(result.summary())                    # 夏普 / 最大回撤 / 成本拖累 / 换手率
```

## 模块地图

| 模块 | 对应 todo.txt 能力点 | 内容 |
|------|------|------|
| `quantlab.data`       | Python 数据栈 / 数据源 | 加载、清洗、缓存；AKShare / Yahoo 真实源 + 合成兜底 |
| `quantlab.stats`      | 统计基础 | 收益/波动/夏普/最大回撤/相关性/回归 |
| `quantlab.costs`      | 交易成本与 A 股摩擦 | 印花税/佣金/滑点/T+1/涨跌停 |
| `quantlab.backtest`   | 回测 | 事件驱动回测引擎，强制扣成本 |
| `quantlab.strategies` | 玩具策略 | 均线交叉等可继承的基类 |
| `quantlab.validation` | **量化研究方法论** | train/test、walk-forward、过拟合诊断 |

## 文档

- [docs/strategy-assessment.md](docs/strategy-assessment.md) — 对 todo.txt 学习策略的评审（**先读这个**）
- [docs/roadmap.md](docs/roadmap.md) — 阶段 1 通关标准与后续阶段规划

## 免责声明

本项目仅用于**研究与学习**，不构成任何投资建议。历史回测不代表未来收益。

## License

MIT
