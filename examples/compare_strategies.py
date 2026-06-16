"""多策略横向对比：趋势跟踪 vs 突破 vs 均值回归 vs 买入持有。

用真实行情，扣 A 股真实成本，输出对照表 + 样本外验证 + 诚实点评：
没有万能策略，只有"匹配市场状态"的策略。

跑法：
    python examples/compare_strategies.py [代码] [开始] [结束]
"""
from __future__ import annotations

import sys

import pandas as pd

from quantlab.data import load_prices
from quantlab.strategies import (
    MACrossStrategy,
    DonchianBreakoutStrategy,
    BollingerReversionStrategy,
    BuyHoldStrategy,
)
from quantlab.costs import AShareCostModel
from quantlab.backtest import Backtester
from quantlab.validation import walk_forward


def _row(res, wf) -> dict:
    s = res.stats()
    return {
        "策略": res.strategy_name,
        "累计收益": f"{s['cumulative_return']:+.1%}",
        "年化": f"{s['annualized_return']:+.1%}",
        "波动": f"{s['annualized_volatility']:.1%}",
        "夏普": f"{s['sharpe']:.2f}",
        "最大回撤": f"{s['max_drawdown']:.1%}",
        "换手/年": f"{s['turnover_annual']:.1f}x",
        "OOS夏普": f"{wf.oos_stats()['sharpe']:.2f}" if wf and len(wf.oos_returns) else "-",
    }


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "600519"
    start = sys.argv[2] if len(sys.argv) > 2 else "2018-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2023-12-31"

    prices = load_prices(symbol, start, end, source="auto")
    tag = "真实行情" if prices.attrs.get("is_real") else "⚠合成兜底"
    print(f"[数据] {symbol} {prices.index[0].date()}~{prices.index[-1].date()} "
          f"{len(prices)}天 来源={prices.attrs.get('source')} [{tag}]\n")

    bt = Backtester(initial_cash=1_000_000, cost_model=AShareCostModel())
    strategies = [
        MACrossStrategy(20, 60),
        DonchianBreakoutStrategy(20, 10),
        BollingerReversionStrategy(20, 2.0),
        BuyHoldStrategy(),
    ]

    rows = []
    for strat in strategies:
        res = bt.run(prices, strat)
        try:
            wf = walk_forward(prices, strat, n_folds=5, backtester=bt)
        except ValueError:
            wf = None
        rows.append(_row(res, wf))

    print("===== 多策略对照（含 A 股真实成本）=====")
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("===== 诚实点评 =====")
    print("  · 趋势/突破（MA、Donchian）在单边趋势里强在【回撤控制】，震荡市易被反复打脸。")
    print("  · 均值回归（Bollinger）在区间/震荡里有效，遇强趋势会过早抄底、吃亏。")
    print("  · 买入持有是基准：任何策略都要先回答'凭什么比一直拿着更好'。")
    print("  · 关注【OOS夏普】而非样本内：样本内好看几乎没意义。")
    print("\n结论：没有万能策略，只有匹配市场状态的策略。下一步是用多标的、多时段")
    print("      检验同一规则的稳健性，而不是在单票上调参数。")


if __name__ == "__main__":
    main()
