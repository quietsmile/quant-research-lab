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
```

它会完成一条完整链路：**数据 → 清洗 → 信号 → 带真实成本的回测 → 样本外验证**，并打印一份诚实的体检报告，包括"为什么这个结果可能是过拟合"。

### 真实数据流

`load_prices(..., source="auto")` 会按 **AKShare → Yahoo Finance → 合成行情** 的顺序自动回落，开箱即能接真实行情（Yahoo 适配器仅用标准库，自动复权）：

```python
from quantlab.data import load_prices
prices = load_prices("600519", "2018-01-01", "2023-12-31", source="yahoo")  # 真实茅台日线
```

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
