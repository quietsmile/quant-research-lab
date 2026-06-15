"""Gate 1 端到端 Demo：数据 → 清洗 → 信号 → 带成本回测 → 样本外验证。

跑法：
    python examples/toy_strategy.py

它完整演示 todo.txt 阶段 1 的通关标准：能独立把一个策略从数据跑到
带成本的回测，并能说清它为什么可能过拟合。
"""
from __future__ import annotations

from quantlab.data import load_prices
from quantlab.strategies import MACrossStrategy
from quantlab.costs import AShareCostModel
from quantlab.backtest import Backtester
from quantlab.validation import walk_forward, overfitting_report


def main() -> None:
    # 1) 数据：联网且装了 akshare 会拉真实数据，否则自动用可复现合成行情
    symbol = "600519"
    prices = load_prices(symbol, start="2016-01-01", end="2023-12-31")
    print(f"[数据] {symbol}: {len(prices)} 个交易日 "
          f"({prices.index[0].date()} ~ {prices.index[-1].date()})\n")

    # 2) 成本模型：A 股默认摩擦
    costs = AShareCostModel()
    print(f"[成本] 一次完整买卖的成本率约 {costs.round_trip_cost_rate():.3%}"
          f"（印花税仅卖出 {costs.stamp_tax_rate:.3%}）\n")

    # 3) 策略 + 4) 带成本回测
    strat = MACrossStrategy(fast=10, slow=30)
    bt = Backtester(initial_cash=1_000_000, cost_model=costs)
    result = bt.run(prices, strat)
    print(result.summary())
    print()

    # 5) 样本外纪律：train/test 过拟合诊断
    print(overfitting_report(prices, strat))
    print()

    # 6) walk-forward 滚动验证
    wf = walk_forward(prices, strat, n_folds=5, backtester=bt)
    print("===== Walk-forward（逐段样本外）=====")
    if not wf.folds.empty:
        print(wf.folds.to_string(index=False))
        oos = wf.oos_stats()
        print(f"\n  拼接后 OOS 夏普: {oos['sharpe']:.2f} | OOS 年化: {oos['annualized_return']:+.2%}")
        print(f"  稳健性（≥50% fold 夏普为正）: {'是 ✅' if wf.is_robust() else '否 ⚠'}")
    print("\n结论：能跑通 ≠ 能赚钱。请永远以 OOS 指标为准，并对参数搜索保持警惕。")


if __name__ == "__main__":
    main()
