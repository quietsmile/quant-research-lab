"""多标的全池回测：把一个策略跑遍标的池，检验跨标的稳健性。

跑法：
    python examples/universe_backtest.py [策略] [开始] [结束]
    策略 ∈ {ma, donchian, bollinger}（默认 ma）

建议先 `python examples/build_dataset.py ... yahoo` 把数据落到本地缓存，
这样本脚本秒级完成。
"""
from __future__ import annotations

import sys

import pandas as pd

from quantlab.research import universe_backtest
from quantlab.data.universe import default_universe, LIQUID_LEADERS
from quantlab.strategies import (
    MACrossStrategy, DonchianBreakoutStrategy, BollingerReversionStrategy,
)

FACTORIES = {
    "ma": (lambda: MACrossStrategy(20, 60), "均线趋势 MA(20,60)"),
    "donchian": (lambda: DonchianBreakoutStrategy(20, 10), "通道突破 Donchian(20,10)"),
    "bollinger": (lambda: BollingerReversionStrategy(20, 2.0), "均值回归 Bollinger(20,2)"),
}


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "ma"
    start = sys.argv[2] if len(sys.argv) > 2 else "2018-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2024-12-31"
    factory, label = FACTORIES.get(key, FACTORIES["ma"])

    print(f"策略：{label}　|　标的池：{len(default_universe())} 只　|　{start}~{end}\n")
    out = universe_backtest(default_universe(), factory, start, end, source="auto")

    df = out["per_symbol"].copy()
    if "sharpe" in df:
        df["名称"] = df["symbol"].map(LIQUID_LEADERS)
        view = df[["symbol", "名称", "cum_return", "sharpe", "max_dd", "bh_sharpe", "beats_bh"]].copy()
        view["cum_return"] = view["cum_return"].map(lambda x: f"{x:+.0%}")
        view["max_dd"] = view["max_dd"].map(lambda x: f"{x:.0%}")
        view["sharpe"] = view["sharpe"].map(lambda x: f"{x:.2f}")
        view["bh_sharpe"] = view["bh_sharpe"].map(lambda x: f"{x:.2f}")
        view = view.sort_values("名称")
        print(view.to_string(index=False))

    s = out["summary"]
    print("\n===== 跨标的稳健性 =====")
    print(f"  有效标的           : {s['n_symbols']}")
    print(f"  夏普中位数         : {s['median_sharpe']:.2f}")
    print(f"  夏普为正占比       : {s['pct_positive_sharpe']:.0%}")
    print(f"  跑赢买入持有占比   : {s['pct_beat_buyhold']:.0%}")
    print(f"  等权组合夏普       : {s['portfolio_sharpe']:.2f}")
    print(f"  等权组合累计/回撤  : {s['portfolio_cum_return']:+.0%} / {s['portfolio_max_dd']:.0%}")
    print("\n解读：看的是【多数标的是否都站得住】+【等权组合】，而非单票最佳。")
    print("      跑赢持有占比若远低于 50%，说明该策略在这批标的上未必有择时价值。")


if __name__ == "__main__":
    main()
