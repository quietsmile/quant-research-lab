"""月度价量多因子策略 + 严肃评估（一开始就上 Walk-Forward / 冻结 Test / DSR）。

因子（用日频价格算，月度调仓）：
- 反转 reversal：−过去 21 日收益（买近期跌的，A 股反转通常很强）
- 动量 momentum：过去 252→21 日收益（12-1，跳过最近 1 月）
- 低波动 lowvol：−过去 60 日日收益标准差

合成 = 三因子各自 winsorize+zscore 等权 → 行业+市值中性化 → 分 5 层。
评估：单因子 IC + 多空(Q5−Q1)在【开发区 Walk-Forward + 冻结 Test + Deflated Sharpe
+ bootstrap】下的表现（针对月频，embargo=1 月）。

跑法：python examples/monthly_price_factors.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data.tushare_adapter import load_daily_prices, load_market_panel, load_listing
from quantlab.factors import winsorize, zscore, neutralize_industry_size
from quantlab import eval as ev

P = 12  # 年化周期：月频


def build_factor_panels():
    px = load_daily_prices().pivot_table(index="trade_date", columns="symbol",
                                         values="adj_close").sort_index()
    ret = px.pct_change(fill_method=None)
    me = list(px.groupby(px.index.to_period("M")).tail(1).index)   # 月末调仓日
    pos = {d: i for i, d in enumerate(px.index)}
    # 季度末市值 → 月度 asof（PIT：用最近一个已知季度市值）
    mvq = load_market_panel().pivot_table(index="trade_date", columns="symbol", values="total_mv").sort_index()
    ind = load_listing().drop_duplicates("symbol").set_index("symbol")["industry"]

    factors = {"reversal": {}, "momentum": {}, "lowvol": {}}
    fwd = {}
    mv_at = {}
    for j, t in enumerate(me[:-1]):
        i = pos[t]
        if i < 252:
            continue
        nxt = me[j + 1]
        factors["reversal"][t] = -(px.loc[t] / px.iloc[i - 21] - 1.0)
        factors["momentum"][t] = px.iloc[i - 21] / px.iloc[i - 252] - 1.0
        factors["lowvol"][t] = -ret.iloc[i - 60:i].std()
        fwd[t] = px.loc[nxt] / px.loc[t] - 1.0
        mv_at[t] = mvq.loc[:t].iloc[-1] if len(mvq.loc[:t]) else pd.Series(dtype=float)
    return factors, fwd, mv_at, ind


def composite_long_short():
    factors, fwd, mv_at, ind = build_factor_panels()
    dates = sorted(fwd)
    ls_each = {k: {} for k in ["reversal", "momentum", "lowvol", "composite"]}
    ic_each = {k: [] for k in ["reversal", "momentum", "lowvol", "composite"]}
    lo_excess = {}   # 纯多头(top20%) - 等权 的月度差
    for t in dates:
        mv = mv_at[t]
        keep = mv.index[mv >= mv.quantile(0.2)] if len(mv) else fwd[t].dropna().index
        f = pd.DataFrame(index=keep)
        for name in ["reversal", "momentum", "lowvol"]:
            f[name] = factors[name][t].reindex(keep)
        f["fwd"] = fwd[t].reindex(keep)
        f["ind"] = ind.reindex(keep)
        f["lsz"] = np.log(mv.reindex(keep)) if len(mv) else 0.0
        f = f.dropna(subset=["fwd"])
        if len(f) < 100:
            continue
        comp = pd.concat([zscore(winsorize(f[c])) for c in ["reversal", "momentum", "lowvol"]],
                         axis=1).mean(axis=1, skipna=True)
        f["composite"] = neutralize_industry_size(comp, f["ind"], f["lsz"])
        for name in ["reversal", "momentum", "lowvol", "composite"]:
            d = pd.concat([f[name].rename("x"), f["fwd"].rename("r")], axis=1).dropna()
            if len(d) < 100:
                continue
            ic_each[name].append(d["x"].corr(d["r"], method="spearman"))
            d["q"] = pd.qcut(d["x"].rank(method="first"), 5, labels=range(1, 6)).astype(int)
            ls_each[name][t] = d.loc[d["q"] == 5, "r"].mean() - d.loc[d["q"] == 1, "r"].mean() - 0.002
        # 纯多头超额(用 composite)
        d = pd.concat([f["composite"].rename("x"), f["fwd"].rename("r")], axis=1).dropna()
        if len(d) >= 100:
            k = max(20, int(len(d) * 0.2))
            top = d.nlargest(k, "x")["r"].mean()
            lo_excess[t] = top - d["r"].mean()
    return ({k: pd.Series(v).sort_index() for k, v in ls_each.items()},
            ic_each, pd.Series(lo_excess).sort_index())


def main() -> None:
    ls, ic, lo_excess = composite_long_short()

    print("===== 单因子 IC（月频，全样本）=====")
    for name in ["reversal", "momentum", "lowvol", "composite"]:
        s = pd.Series(ic[name]).dropna()
        ir = s.mean() / s.std() if s.std() else 0
        print(f"  {name:<10} IC均值 {s.mean():+.3f} | IC_IR {ir:.2f} | IC>0 {(s>0).mean():.0%} | t {s.mean()/s.std()*np.sqrt(len(s)) if s.std() else 0:.1f}")

    comp = ls["composite"]
    dev_d, test_d = ev.dev_test_split(comp.index, test_frac=0.2)
    dev = comp.loc[comp.index.isin(dev_d)]; test = comp.loc[comp.index.isin(test_d)]
    print(f"\n开发 {dev.index[0].date()}~{dev.index[-1].date()}({len(dev)}月) | 冻结Test {test.index[0].date()}~{test.index[-1].date()}({len(test)}月)")

    print("\n===== Walk-Forward 验证折(Train60月/Val12月/步12/embargo1) =====")
    folds = ev.walk_forward_splits(dev.index, train_size=60, val_size=12, step=12, embargo=1)
    vsr = []
    for k, (tr, va) in enumerate(folds, 1):
        vr = dev.loc[dev.index.isin(va)]; s = ev.sharpe(vr, P); vsr.append(s)
        print(f"  Fold{k}: Val {va[0].date()}~{va[-1].date()} 多空年化 {(1+vr).prod()**(P/len(vr))-1:+.1%} 夏普 {s:.2f}")
    if vsr:
        print(f"  → 验证折夏普均值 {np.mean(vsr):.2f}, 为正 {sum(s>0 for s in vsr)}/{len(vsr)}")

    trial_std = float(np.std([ev.sharpe(ls[n].loc[ls[n].index.isin(dev_d)], 1)
                              for n in ["reversal", "momentum", "lowvol", "composite"]], ddof=1))
    print("\n===== 冻结 Test(只跑一次)=====")
    print(f"  多空年化 {(1+test).prod()**(P/len(test))-1:+.1%} | 夏普 {ev.sharpe(test,P):.2f}")
    dsr = ev.deflated_sharpe_ratio(test, n_trials=4, trials_sr_std=trial_std, periods=P)
    print(f"  PSR(vs0) {dsr['psr_vs_0']:.0%} | Deflated SR {dsr['dsr']:.0%}(门槛年化夏普 {dsr['sr_benchmark_annual']:.2f})")
    ci = ev.block_bootstrap_ci(test, block=6, periods=P)
    if ci:
        print(f"  bootstrap 95%CI 年化 [{ci['ann_return_ci'][0]:+.1%},{ci['ann_return_ci'][1]:+.1%}] 夏普 [{ci['sharpe_ci'][0]:.2f},{ci['sharpe_ci'][1]:.2f}]")
    le_test = lo_excess.loc[lo_excess.index.isin(test_d)]
    print(f"  [参考] 纯多头(top20%)相对等权 月超额年化(Test) {(1+le_test).prod()**(P/len(le_test))-1:+.1%}")


if __name__ == "__main__":
    main()
