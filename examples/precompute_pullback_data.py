"""为'强势板块回撤买入'可调参看板预计算数据底座。
保存 (liq1500, 2017-07起): 个股 close/ret、行业日收益(算板块涨幅/趋势)、质量(扣非ROE/毛利/市值)、行业映射、基准。
存 dashboard_data/pullback_*.parquet。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib
D = pathlib.Path.home() / ".local/share/quantlab/fundamentals"
OUT = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data"); OUT.mkdir(exist_ok=True)

ind = pd.read_parquet(D / "industry.parquet"); ind["symbol"] = ind["symbol"].astype(str).str.zfill(6)
imap = ind.set_index("symbol")["industry"]
oh = pd.read_parquet(D / "daily_ohlcv.parquet")[["trade_date", "symbol", "adj_close", "amount"]]
oh["symbol"] = oh["symbol"].astype(str).str.zfill(6)
p26 = pd.read_parquet(D / "daily_2026.parquet")[["trade_date", "symbol", "adj_close", "amount", "total_mv"]]
p26["symbol"] = p26["symbol"].astype(str).str.zfill(6)
liq = set(oh.groupby("symbol")["amount"].mean().nlargest(1500).index)
oh = oh[oh.symbol.isin(liq)]; p26 = p26[p26.symbol.isin(liq)]; syms = sorted(liq)
def Wd(df, c): return df.pivot_table(index="trade_date", columns="symbol", values=c)
close = pd.concat([Wd(oh, "adj_close"), Wd(p26, "adj_close")]); close = close[~close.index.duplicated()].sort_index().reindex(columns=syms)
idx = [d for d in close.index if d >= pd.Timestamp("2017-07-01")]; close = close.reindex(idx)
ret = close.pct_change(fill_method=None)

# 行业日收益(等权) → 行业nav
ind_of = pd.Series({s: (imap.get(s) if isinstance(imap.get(s), str) else "UNK") for s in syms})
industries = sorted(set(ind_of.values) - {"UNK"})
sret = pd.DataFrame({g: ret[[s for s in syms if ind_of[s] == g]].mean(axis=1) for g in industries})

# 质量: 扣非ROE / 毛利 (PIT, 公告日as-of) + 市值
tf = pd.read_parquet(D / "tushare_features.parquet")[["symbol", "announce_date", "roe_dedt", "gross_margin"]].copy()
tf["symbol"] = tf["symbol"].astype(str).str.zfill(6); tf = tf[tf.symbol.isin(liq)].dropna(subset=["announce_date"]).sort_values("announce_date")
cal = pd.DataFrame({"announce_date": pd.DatetimeIndex(idx)})
def pit(col):
    out = {}
    for s, g in tf.groupby("symbol"):
        g = g.dropna(subset=[col]).sort_values("announce_date")
        if len(g): out[s] = pd.merge_asof(cal, g[["announce_date", col]], on="announce_date").set_index("announce_date")[col]
    return pd.DataFrame(out).reindex(index=pd.DatetimeIndex(idx), columns=syms)
roe = pit("roe_dedt"); gm = pit("gross_margin")
mp = pd.read_parquet(D / "market_panel.parquet"); mp["symbol"] = mp["symbol"].astype(str).str.zfill(6); mp = mp[mp.symbol.isin(liq)]
q = mp.pivot_table(index="trade_date", columns="symbol", values="total_mv").reindex(columns=syms)
mv = q.reindex(index=pd.DatetimeIndex(idx).union(q.index)).sort_index().ffill().reindex(idx)
mv26 = Wd(p26, "total_mv").reindex(columns=syms)
m26 = mv.index >= pd.Timestamp("2026-01-01"); mv.loc[m26] = mv26.reindex(mv.index[m26])

# 基准: 沪深300 + 等权
try:
    import sys; sys.path.insert(0, "/home/claudeuser/econ/quant-research-lab")
    from quantlab.data.tushare_adapter import get_pro
    hs = get_pro().index_daily(ts_code="000300.SH", start_date="20170101", end_date="20260620", fields="trade_date,close")
    hs["trade_date"] = pd.to_datetime(hs["trade_date"]); hs300 = hs.sort_values("trade_date").set_index("trade_date")["close"].pct_change().reindex(idx)
except Exception:
    hs300 = ret.mean(axis=1)
bench = pd.DataFrame({"等权全市场": ret.mean(axis=1), "沪深300": hs300}, index=idx)

close.to_parquet(OUT / "pullback_close.parquet")
ret.to_parquet(OUT / "pullback_ret.parquet")
sret.to_parquet(OUT / "pullback_sret.parquet")
roe.to_parquet(OUT / "pullback_roe.parquet")
gm.to_parquet(OUT / "pullback_gm.parquet")
mv.to_parquet(OUT / "pullback_mv.parquet")
bench.to_parquet(OUT / "pullback_bench.parquet")
ind_of.rename("industry").to_frame().to_parquet(OUT / "pullback_sector.parquet")
print(f"保存完成 -> {OUT}  股票{len(syms)} 行业{len(industries)} 日期{idx[0].date()}~{idx[-1].date()} ({len(idx)}天)")
