"""(C) 容量曲线:给 Top-20 策略加成交参与率/冲击模型,反推资金容量天花板。

补上之前缺的「参与率/市场冲击」维度(对照 ashare-lowfreq-research 的回测引擎)。
对每个 AUM:单票下单额=AUM/20;order/ADV 比例→平方根冲击模型 impact=σ√(order/ADV);
统计触参与率上限(>10%ADV)的比例 + 年化冲击拖累 + 净年化(基线−拖累)。

跑法：python examples/capacity_curve.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from examples.ml_trade import load_signal, simulate, metrics, syms
from examples.strategy_family import idx
from quantlab.data.tushare_adapter import load_daily_ohlcv

ANN = 242
SIGMA = 0.025            # 日波动(冲击模型)
ETA = 1.0               # 冲击系数
PR = 0.10               # 参与率上限(单日单票最多吃 10% 成交额)


def main():
    pred = load_signal()
    o = load_daily_ohlcv(); o = o[o.symbol.isin(syms)]
    amt = o.pivot_table(index="trade_date", columns="symbol", values="amount").reindex(index=idx, columns=syms) * 1000  # 千元→元
    port, tr = simulate(pred, hold=10, top_n=20, realistic=True, exclude_st=True, use_fund=True)
    base = metrics(port)
    tr["amt"] = [amt.loc[t["entry"], t["symbol"]] if t["symbol"] in amt.columns else np.nan
                 for _, t in tr.iterrows()]
    tr = tr.dropna(subset=["amt"])
    n_rb_per_yr = ANN / 10
    print(f"基线(Top20/hold10/真实撮合/扣成本): 年化 {base['cagr']*100:+.0f}% 夏普 {base['sharpe']:.2f}\n")
    print(f"{'AUM':>8s} {'单票下单':>9s} {'中位order/ADV':>13s} {'>10%ADV占比':>11s} {'年化冲击拖累':>11s} {'净年化':>7s}", flush=True)
    for aum in [1e6, 5e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9]:
        order = aum / 20
        ratio = order / tr["amt"]
        # 额外冲击只对 >0.5%ADV 的单子计(更小的单 base 成本已覆盖,避免双重计费)
        impact_ow = ETA * SIGMA * np.sqrt((ratio - 0.005).clip(lower=0, upper=5))   # 单边额外冲击(平方根)
        drag = n_rb_per_yr * impact_ow.mean() * 2                   # 年化:每年~24次换手×双边
        over = (ratio > PR).mean()
        aum_s = f"{aum/1e8:.2f}亿" if aum >= 1e8 else f"{aum/1e4:.0f}万"
        print(f"{aum_s:>8s} {order/1e4:>7.0f}万 {ratio.median():>12.3f} {over*100:>10.0f}% "
              f"{drag*100:>10.1f}% {(base['cagr']-drag)*100:>+6.0f}%", flush=True)
    print("\n判读: 净年化跌到≈0 的 AUM 就是容量天花板。Top-20极集中→容量很小;"
          "\n      叠加之前对冲后净alpha本就薄(+0.25夏普),冲击会进一步压低 → 这是个小资金策略。")


if __name__ == "__main__":
    main()
