"""演示完整评估面板：对月度多因子多空策略输出六大类指标 + 分年。

跑法：python examples/strategy_report.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

from examples.monthly_price_factors import composite_long_short
from quantlab.report import performance_report, format_report


def main():
    ls, ic, lo = composite_long_short()
    comp = ls["composite"]
    m = performance_report(comp, periods=12, n_trials=4, trials_sr_std=0.3)
    print(format_report(m, "月度多因子(反转+动量+低波动) 多空"))
    if "yearly" in m:
        print("\n分年收益(%):")
        print((m["yearly"] * 100).round(1).to_string())


if __name__ == "__main__":
    main()
