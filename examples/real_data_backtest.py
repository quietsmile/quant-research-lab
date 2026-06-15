"""真实数据回测：趋势跟踪（双均线） vs 买入持有，并讲清"为什么有用"。

数据流：Yahoo Finance 真实日线（自动复权）→ 清洗 → 信号 → 带 A 股真实
成本回测 → 与买入持有对照 → walk-forward 样本外验证。

跑法：
    python examples/real_data_backtest.py [代码] [开始] [结束]
    例：python examples/real_data_backtest.py 600519 2018-01-01 2023-12-31
"""
from __future__ import annotations

import sys

import pandas as pd

from quantlab.data import load_prices
from quantlab.strategies import MACrossStrategy, BuyHoldStrategy
from quantlab.costs import AShareCostModel
from quantlab.backtest import Backtester
from quantlab.stats import metrics
from quantlab.validation import walk_forward


def _row(name: str, res) -> dict:
    s = res.stats()
    return {
        "策略": name,
        "累计收益": f"{s['cumulative_return']:+.1%}",
        "年化": f"{s['annualized_return']:+.1%}",
        "年化波动": f"{s['annualized_volatility']:.1%}",
        "夏普": f"{s['sharpe']:.2f}",
        "最大回撤": f"{s['max_drawdown']:.1%}",
        "Calmar": f"{s['calmar']:.2f}",
        "成本占比": f"{s['cost_drag_pct']:.1%}",
    }


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "600519"
    start = sys.argv[2] if len(sys.argv) > 2 else "2018-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2023-12-31"

    # 1) 真实数据（Yahoo 优先；不可用时框架自动回落，见 load_prices）
    prices = load_prices(symbol, start, end, source="auto")
    src = prices.attrs.get("source", "?")
    tag = "真实行情" if prices.attrs.get("is_real") else "⚠ 合成兜底(非真实)"
    print(f"[数据] {symbol}  {prices.index[0].date()} ~ {prices.index[-1].date()}  "
          f"({len(prices)} 交易日)  来源={src} [{tag}]")
    print(f"       期初收盘 {prices['close'].iloc[0]:.1f} → 期末 {prices['close'].iloc[-1]:.1f}\n")

    costs = AShareCostModel()
    bt = Backtester(initial_cash=1_000_000, cost_model=costs)

    # 2) 简单策略 vs 买入持有基准
    strat = MACrossStrategy(fast=20, slow=60)
    res_strat = bt.run(prices, strat)
    res_bh = bt.run(prices, BuyHoldStrategy())

    print("===== 对照回测（含 A 股真实成本）=====")
    table = pd.DataFrame([_row(strat.name, res_strat), _row("买入持有", res_bh)])
    print(table.to_string(index=False))
    print()

    # 3) "为什么有用"——量化趋势跟踪的真正价值：回撤控制
    bh_mdd = metrics.max_drawdown(res_bh.returns)
    st_mdd = metrics.max_drawdown(res_strat.returns)
    print("===== 为什么这个策略有用 =====")
    print(f"  · 趋势跟踪的核心不是博更高收益，而是【回撤控制】。")
    print(f"    买入持有最大回撤 {bh_mdd:.1%}，策略 {st_mdd:.1%}，"
          f"回撤改善 {bh_mdd - st_mdd:+.1%}。")
    # 找出买入持有最深回撤的时间窗，看策略当时在不在场内
    equity_bh = (1 + res_bh.returns).cumprod()
    peak = equity_bh.cummax()
    dd = equity_bh / peak - 1
    trough = dd.idxmin()
    peak_date = equity_bh.loc[:trough].idxmax()
    in_window = (res_strat.positions.index >= peak_date) & (res_strat.positions.index <= trough)
    avg_pos = res_strat.positions[in_window].mean()
    print(f"  · 买入持有最深回撤区间：{peak_date.date()} → {trough.date()}（跌 {dd.min():.1%}）。")
    print(f"    同期策略平均仓位仅 {avg_pos:.0%}——趋势转弱时它已离场，这就是回撤更小的来源。")
    print(f"  · 经济学依据：趋势/动量是跨市场、跨资产最稳健的异象之一（行为上的")
    print(f"    反应不足 + 羊群效应让趋势持续）。一次完整买卖成本仅约 "
          f"{costs.round_trip_cost_rate():.2%}，")
    print(f"    远小于它要捕捉的趋势波幅，因此摩擦吃不掉信号。\n")

    # 4) 样本外纪律
    wf = walk_forward(prices, strat, n_folds=5, backtester=bt)
    print("===== Walk-forward 样本外验证 =====")
    if not wf.folds.empty:
        print(wf.folds.to_string(index=False))
        oos = wf.oos_stats()
        print(f"\n  拼接 OOS 夏普 {oos['sharpe']:.2f}｜OOS 年化 {oos['annualized_return']:+.1%}"
              f"｜稳健性 {'是 ✅' if wf.is_robust() else '否 ⚠'}")
    print("\n注：单标的样本有限，结论仅作流程演示，不构成投资建议。")


if __name__ == "__main__":
    main()
