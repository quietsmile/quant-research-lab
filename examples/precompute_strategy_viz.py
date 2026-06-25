"""预计算'等权买入持有'策略(及对比变体)全周期日收益, 供看板可视化。
存 ~/.local/share/quantlab/fundamentals/strat_viz.parquet (date索引, 各变体日收益列)。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib
D = pathlib.Path.home() / ".local/share/quantlab/fundamentals"
oh = pd.read_parquet(D / "daily_ohlcv.parquet")[["trade_date", "symbol", "adj_close", "amount"]]
oh["symbol"] = oh["symbol"].astype(str).str.zfill(6)
p26 = pd.read_parquet(D / "daily_2026.parquet")[["trade_date", "symbol", "adj_close", "amount"]]
p26["symbol"] = p26["symbol"].astype(str).str.zfill(6)
liq = set(oh.groupby("symbol")["amount"].mean().nlargest(2000).index)
oh = oh[oh.symbol.isin(liq)]; p26 = p26[p26.symbol.isin(liq)]; syms = sorted(liq)
close = pd.concat([oh.pivot_table(index="trade_date", columns="symbol", values="adj_close"),
                   p26.pivot_table(index="trade_date", columns="symbol", values="adj_close")])
close = close[~close.index.duplicated()].sort_index().reindex(columns=syms)
amt = pd.concat([oh.pivot_table(index="trade_date", columns="symbol", values="amount"),
                 p26.pivot_table(index="trade_date", columns="symbol", values="amount")])
amt = amt[~amt.index.duplicated()].sort_index().reindex(columns=syms)
ret = close.pct_change(fill_method=None)
idx = [d for d in ret.index if d >= pd.Timestamp("2017-01-01")]
ret = ret.reindex(idx); close = close.reindex(idx); amt = amt.reindex(idx)

out = pd.DataFrame(index=idx)
out["每日再平衡等权"] = ret.mean(axis=1)
# 真买入持有等权(期初等权漂移, 固定全程在册股票)
valid = close.iloc[0].notna() & close.iloc[-1].notna()
nav = close.loc[:, valid].div(close.loc[:, valid].iloc[0])
out["买入持有等权"] = nav.mean(axis=1).pct_change()
# 月度再平衡等权 扣成本
def monthly_ew(k=20, cost=0.002):
    w = None; o = pd.Series(0.0, index=idx); cst = pd.Series(0.0, index=idx)
    for i, d in enumerate(idx):
        avail = ret.loc[d].dropna().index
        if i % k == 0:
            nw = pd.Series(1.0 / len(avail), index=avail)
            if w is not None:
                u = w.index.union(nw.index)
                cst.loc[d] = (nw.reindex(u).fillna(0) - w.reindex(u).fillna(0)).abs().sum() * cost / 2
            w = nw
        o.loc[d] = (ret.loc[d, w.index] * w).sum()
        w = w * (1 + ret.loc[d, w.index].fillna(0)); w = w / w.sum()
    return o - cst
out["月度再平衡等权(扣成本)"] = monthly_ew()
# 市值加权买入持有(成交额近似权重)
a0 = amt.iloc[:20].mean(); v2 = valid & a0.notna()
nav2 = close.loc[:, v2].div(close.loc[:, v2].iloc[0]); w0 = a0[v2] / a0[v2].sum()
out["市值加权买入持有"] = (nav2 * w0).sum(axis=1).pct_change()

out.index.name = "date"
out.to_parquet(D / "strat_viz.parquet")
print("saved strat_viz.parquet", out.shape, "范围", out.index.min().date(), "~", out.index.max().date())
print(out.describe().T[["mean", "std"]].to_string())
