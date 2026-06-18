"""因子工坊：在"反转+低波动"基础上再发散 10 个价量 idea，逐个严肃评估并迭代。

每个因子都走：全样本 IC（含 t 值）→ 开发区/冻结 Test 切分 → 冻结 Test 多空夏普 +
Deflated Sharpe。最后把 OOS 最稳的几个合成"迭代版"，再严肃评估一次。

针对月频、A 股；价格用日频后复权（含退市，无幸存者偏差）。
跑法：python examples/factor_zoo_research.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from quantlab.data.tushare_adapter import load_daily_prices, load_market_panel, load_listing
from quantlab.factors import winsorize, zscore, neutralize_industry_size
from quantlab import eval as ev

P = 12
COST = 0.002


def _prep():
    px = load_daily_prices().pivot_table(index="trade_date", columns="symbol",
                                         values="adj_close").sort_index()
    ret = px.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)                       # 等权市场日收益
    me = list(px.groupby(px.index.to_period("M")).tail(1).index)
    mvq = load_market_panel().pivot_table(index="trade_date", columns="symbol", values="total_mv").sort_index()
    return px, ret, mkt, me, mvq


# 10 个新 idea（外加 2 个基线 rev_21 / lowvol_60 用于对照）
def factor_cross_section(name, px, ret, mkt, i, t):
    W = ret.iloc[i - 60:i]                       # 60 日窗
    if name == "rev_5":      return -(px.loc[t] / px.iloc[i - 5] - 1)
    if name == "rev_10":     return -(px.loc[t] / px.iloc[i - 10] - 1)
    if name == "rev_21":     return -(px.loc[t] / px.iloc[i - 21] - 1)
    if name == "lowvol_60":  return -W.std()
    if name == "lowvol_120": return -ret.iloc[i - 120:i].std()
    if name == "downvol_60": return -W.where(W < 0).std()
    if name == "skew_60":    return -W.skew()
    if name == "max_21":     return -ret.iloc[i - 21:i].max()             # 彩票/MAX
    if name == "dist_52whigh": return px.loc[t] / px.iloc[i - 252:i + 1].max()
    if name == "volscaled_rev": return (-(px.loc[t] / px.iloc[i - 21] - 1)) / (W.std() + 1e-9)
    if name in ("ivol_60", "beta_60"):
        Wv = W.values; mw = mkt.iloc[i - 60:i].values
        mwc = mw - np.nanmean(mw); var = np.nanmean(mwc ** 2)
        Wc = Wv - np.nanmean(Wv, axis=0)
        beta = np.nanmean(Wc * mwc[:, None], axis=0) / (var + 1e-12)
        if name == "beta_60":
            return pd.Series(-beta, index=W.columns)                       # BAB 低beta
        resid = Wv - beta[None, :] * mw[:, None]
        return pd.Series(-np.nanstd(resid, axis=0), index=W.columns)       # 特异波动
    raise ValueError(name)


IDEAS = ["rev_5", "rev_10", "lowvol_120", "downvol_60", "skew_60", "max_21",
         "dist_52whigh", "volscaled_rev", "ivol_60", "beta_60"]
BASE = ["rev_21", "lowvol_60"]


def long_short_and_ic(names, px, ret, mkt, me, mvq, neutralize=False, ind=None):
    """对一组因子，返回每个的(月度多空序列, IC列表)。neutralize=合成时行业/市值中性。"""
    ls = {n: {} for n in names}; ic = {n: [] for n in names}
    for j, t in enumerate(me[:-1]):
        i = px.index.get_loc(t)
        if i < 252:
            continue
        nxt = me[j + 1]
        mv = mvq.loc[:t].iloc[-1] if len(mvq.loc[:t]) else None
        keep = mv.index[mv >= mv.quantile(0.2)] if mv is not None else px.columns
        fwd = (px.loc[nxt] / px.loc[t] - 1).reindex(keep)
        for n in names:
            f = factor_cross_section(n, px, ret, mkt, i, t).reindex(keep)
            if neutralize and ind is not None and mv is not None:
                f = neutralize_industry_size(f, ind.reindex(keep), np.log(mv.reindex(keep)))
            d = pd.concat([f.rename("x"), fwd.rename("r")], axis=1).dropna()
            if len(d) < 100:
                continue
            ic[n].append(d["x"].corr(d["r"], method="spearman"))
            d["q"] = pd.qcut(d["x"].rank(method="first"), 5, labels=range(1, 6)).astype(int)
            ls[n][t] = d.loc[d["q"] == 5, "r"].mean() - d.loc[d["q"] == 1, "r"].mean() - COST
    return {n: pd.Series(ls[n]).sort_index() for n in names}, ic


def evaluate(series, ic_list):
    s = series; icv = pd.Series(ic_list).dropna()
    dev_d, test_d = ev.dev_test_split(s.index, test_frac=0.2)
    dev, test = s.loc[s.index.isin(dev_d)], s.loc[s.index.isin(test_d)]
    t_ic = icv.mean() / icv.std() * np.sqrt(len(icv)) if icv.std() else 0
    return {
        "ic": icv.mean(), "ic_t": t_ic,
        "dev_sr": ev.sharpe(dev, P), "test_sr": ev.sharpe(test, P),
        "test_ann": (1 + test).prod() ** (P / len(test)) - 1 if len(test) else np.nan,
        "dev": dev, "test": test, "dev_d": dev_d, "test_d": test_d,
    }


def main() -> None:
    px, ret, mkt, me, mvq = _prep()
    ind = load_listing().drop_duplicates("symbol").set_index("symbol")["industry"]
    allnames = IDEAS + BASE
    print(f"评估 {len(IDEAS)} 个新 idea + {len(BASE)} 基线，月频 {me[12].date()}~{me[-1].date()}\n")
    ls, ic = long_short_and_ic(allnames, px, ret, mkt, me, mvq)

    rows = []
    for n in allnames:
        e = evaluate(ls[n], ic[n])
        rows.append({"factor": n, **{k: e[k] for k in ["ic", "ic_t", "dev_sr", "test_sr", "test_ann"]}})
    tab = pd.DataFrame(rows).sort_values("test_sr", ascending=False)
    print("===== 10 个 idea + 基线：严肃评估(按冻结Test夏普排序) =====")
    print(f"{'因子':<14}{'IC':>7}{'IC_t':>6}{'开发夏普':>9}{'Test夏普':>9}{'Test年化':>9}")
    for _, r in tab.iterrows():
        flag = " ✅" if r.test_sr > 0.5 and r.ic_t > 2 else ""
        print(f"{r.factor:<14}{r.ic:>+7.3f}{r.ic_t:>6.1f}{r.dev_sr:>9.2f}{r.test_sr:>9.2f}{r.test_ann:>+9.1%}{flag}")

    # ---- 迭代：取 OOS(test夏普)最稳的前 4 个，行业/市值中性化后合成 ----
    top = tab.head(4)["factor"].tolist()
    print(f"\n===== 迭代合成(取 OOS 前4: {top}，行业+市值中性) =====")
    comp_ls, comp_ic = long_short_and_ic_composite(top, px, ret, mkt, me, mvq, ind)
    e = evaluate(comp_ls, comp_ic)
    trial_std = float(np.std([evaluate(ls[n], ic[n])["dev_sr"] / np.sqrt(P) for n in allnames], ddof=1))
    dsr = ev.deflated_sharpe_ratio(e["test"], n_trials=len(allnames), trials_sr_std=trial_std, periods=P)
    icv = pd.Series(comp_ic).dropna()
    print(f"  合成 IC {icv.mean():+.3f}(t {icv.mean()/icv.std()*np.sqrt(len(icv)):.1f}) | 开发夏普 {e['dev_sr']:.2f} | "
          f"Test夏普 {e['test_sr']:.2f}(年化 {e['test_ann']:+.1%})")
    print(f"  Deflated SR {dsr['dsr']:.0%}(门槛年化夏普 {dsr['sr_benchmark_annual']:.2f}, 校正试了{len(allnames)}个因子) | PSR {dsr['psr_vs_0']:.0%}")
    ci = ev.block_bootstrap_ci(e["test"], block=6, periods=P)
    if ci:
        print(f"  bootstrap 95%CI 年化 [{ci['ann_return_ci'][0]:+.1%},{ci['ann_return_ci'][1]:+.1%}] 夏普 [{ci['sharpe_ci'][0]:.2f},{ci['sharpe_ci'][1]:.2f}]")
    # 存结果供写文档
    tab.to_csv("/tmp/factor_zoo_results.csv", index=False)
    print("\n结果表已存 /tmp/factor_zoo_results.csv")


def long_short_and_ic_composite(names, px, ret, mkt, me, mvq, ind):
    ls = {}; ic = []
    for j, t in enumerate(me[:-1]):
        i = px.index.get_loc(t)
        if i < 252:
            continue
        nxt = me[j + 1]
        mv = mvq.loc[:t].iloc[-1] if len(mvq.loc[:t]) else None
        keep = mv.index[mv >= mv.quantile(0.2)] if mv is not None else px.columns
        fwd = (px.loc[nxt] / px.loc[t] - 1).reindex(keep)
        zs = []
        for n in names:
            f = factor_cross_section(n, px, ret, mkt, i, t).reindex(keep)
            zs.append(zscore(winsorize(f)))
        comp = pd.concat(zs, axis=1).mean(axis=1, skipna=True)
        comp = neutralize_industry_size(comp, ind.reindex(keep), np.log(mv.reindex(keep)) if mv is not None else pd.Series(0, index=keep))
        d = pd.concat([comp.rename("x"), fwd.rename("r")], axis=1).dropna()
        if len(d) < 100:
            continue
        ic.append(d["x"].corr(d["r"], method="spearman"))
        d["q"] = pd.qcut(d["x"].rank(method="first"), 5, labels=range(1, 6)).astype(int)
        ls[t] = d.loc[d["q"] == 5, "r"].mean() - d.loc[d["q"] == 1, "r"].mean() - COST
    return pd.Series(ls).sort_index(), ic


if __name__ == "__main__":
    main()
