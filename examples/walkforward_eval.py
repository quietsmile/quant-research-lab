"""严肃回测评估：Walk-Forward + 冻结 Test + Deflated Sharpe + Bootstrap。

把因子策略当"模拟研发—上线流程"评估（针对季度调仓）：
- 开发区间(≤2023) 做滚动 Train/Validation（Train 与 Val 间留 1 季 embargo）；
- Test(2024–2025) 完全冻结，只在规则定死后跑一次；
- 用 Deflated Sharpe 校正"试了多个因子"的选择偏差，用 block bootstrap 给置信区间。

跑法：python examples/walkforward_eval.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data import point_in_time
from quantlab.data.tushare_adapter import load_market_panel
from quantlab.factors import winsorize, zscore
from quantlab import eval as ev

NQ = 5
COST = 0.003


def factor_long_short_series():
    """一次遍历，算各因子 + 合成 的每季多空(Q5-Q1)收益序列。返回 dict[name]->Series。"""
    panel = load_market_panel().sort_values("trade_date")
    dates = sorted(panel["trade_date"].unique())
    aclose = panel.pivot_table(index="trade_date", columns="symbol", values="adj_close")
    names = ["ey", "bm", "roe", "growth", "composite"]
    out = {n: {} for n in names}
    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        cs = panel[panel["trade_date"] == t].set_index("symbol")
        keep = cs.index[cs["total_mv"] >= cs["total_mv"].quantile(0.2)]
        fwd = (aclose.loc[t1] / aclose.loc[t] - 1.0).reindex(keep)
        sn = point_in_time(pd.Timestamp(t).strftime("%Y-%m-%d")).set_index("symbol")
        df = pd.DataFrame(index=keep)
        df["ey"] = (1.0 / cs["pe_ttm"]).where(cs["pe_ttm"] > 0)
        df["bm"] = (1.0 / cs["pb"]).where(cs["pb"] > 0)
        df["roe"] = sn["roe"].reindex(keep)
        df["growth"] = sn["net_profit_q_yoy"].reindex(keep)
        df["fwd"] = fwd
        zs = {c: zscore(winsorize(df[c])) for c in ["ey", "bm", "roe", "growth"]}
        df["composite"] = pd.concat(zs.values(), axis=1).mean(axis=1, skipna=True)
        for name in names:
            d = pd.concat([df[name].rename("f"), df["fwd"].rename("r")], axis=1).dropna()
            if len(d) < 50:
                continue
            d["q"] = pd.qcut(d["f"].rank(method="first"), NQ, labels=range(1, NQ + 1)).astype(int)
            ls = d.loc[d["q"] == NQ, "r"].mean() - d.loc[d["q"] == 1, "r"].mean() - COST
            out[name][pd.Timestamp(t)] = ls
    return {n: pd.Series(s).sort_index() for n, s in out.items()}


def main() -> None:
    series = factor_long_short_series()
    comp = series["composite"]
    dev_dates, test_dates = ev.dev_test_split(comp.index, test_frac=0.2)
    dev, test = comp.loc[comp.index.isin(dev_dates)], comp.loc[comp.index.isin(test_dates)]
    print(f"开发区间 {dev.index[0].date()}~{dev.index[-1].date()} ({len(dev)}季) | "
          f"冻结Test {test.index[0].date()}~{test.index[-1].date()} ({len(test)}季)\n")

    # 1) Walk-Forward：开发区内滚动验证(Train 5年=20季, Val 1年=4季, 步长4季, embargo 1季)
    folds = ev.walk_forward_splits(dev.index, train_size=20, val_size=4, step=4, embargo=1)
    print("===== Walk-Forward 验证折(开发区内, embargo=1季) =====")
    val_sr = []
    for k, (tr, va) in enumerate(folds, 1):
        vr = dev.loc[dev.index.isin(va)]
        s = ev.sharpe(vr, 4); val_sr.append(s)
        print(f"  Fold{k}: Train {tr[0].date()}~{tr[-1].date()} | Val {va[0].date()}~{va[-1].date()} "
              f"多空年化 {(1+vr).prod()**(4/len(vr))-1:+.1%} 夏普 {s:.2f}")
    if val_sr:
        print(f"  → 验证折夏普: 均值 {np.mean(val_sr):.2f}, 为正 {sum(s>0 for s in val_sr)}/{len(val_sr)} 折(稳健性)")

    # 2) 多重检验：试了 5 个因子，用其(每期)夏普离散度做 Deflated Sharpe 的 trials_std
    trial_sr_per = [ev.sharpe(series[n].loc[series[n].index.isin(dev_dates)], 1)
                    for n in ["ey", "bm", "roe", "growth", "composite"]]
    trials_std = float(np.std(trial_sr_per, ddof=1))

    # 3) 冻结 Test 一次性评估
    print("\n===== 冻结 Test 评估(规则定死后只跑一次) =====")
    print(f"  Test 多空年化 {(1+test).prod()**(4/len(test))-1:+.1%} | 夏普 {ev.sharpe(test,4):.2f}")
    dsr = ev.deflated_sharpe_ratio(test, n_trials=5, trials_sr_std=trials_std, periods=4)
    print(f"  PSR(vs 0): {dsr['psr_vs_0']:.0%}  | Deflated SR(校正试了5个因子): {dsr['dsr']:.0%} "
          f"(基准年化夏普门槛 {dsr['sr_benchmark_annual']:.2f})")
    ci = ev.block_bootstrap_ci(test, block=4, periods=4)
    if ci:
        print(f"  Block-bootstrap 95%CI: 年化收益 [{ci['ann_return_ci'][0]:+.1%}, {ci['ann_return_ci'][1]:+.1%}] | "
              f"夏普 [{ci['sharpe_ci'][0]:.2f}, {ci['sharpe_ci'][1]:.2f}]")

    print("\n判读：DSR>~90% 才算在'试了多个因子'后仍稳健显著；CI 含 0 说明样本不足以下定论。"
          "\n本例 Test 仅 ~8 季、独立交易次数少，结论谨慎——这正是诚实评估该呈现的。")


if __name__ == "__main__":
    main()
