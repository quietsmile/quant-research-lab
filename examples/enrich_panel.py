"""给策略搜索补充: PIT 财报成长(单季利润/营收同比)、价值(1/PE_TTM)、扣非ROE(已有)。
对齐 dashboard_data/pullback_close 的 liq1500 × 日频。存 pullback_pftyoy/revyoy/ep.parquet。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib
D = pathlib.Path.home() / ".local/share/quantlab/fundamentals"
DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data")
close = pd.read_parquet(DD / "pullback_close.parquet"); close.index = pd.to_datetime(close.index)
syms = list(close.columns); idx = close.index

tf = pd.read_parquet(D / "tushare_features.parquet")[
    ["symbol", "announce_date", "net_profit_q_yoy", "revenue_q_yoy"]].copy()
tf["symbol"] = tf["symbol"].astype(str).str.zfill(6); tf = tf[tf.symbol.isin(set(syms))]
tf = tf.dropna(subset=["announce_date"]).sort_values("announce_date")
cal = pd.DataFrame({"announce_date": idx})
def pit(col):
    out = {}
    for s, g in tf.groupby("symbol"):
        g = g.dropna(subset=[col]).sort_values("announce_date")
        if len(g): out[s] = pd.merge_asof(cal, g[["announce_date", col]], on="announce_date").set_index("announce_date")[col]
    return pd.DataFrame(out).reindex(index=idx, columns=syms)
pft = pit("net_profit_q_yoy"); rev = pit("revenue_q_yoy")
pft.to_parquet(DD / "pullback_pftyoy.parquet"); rev.to_parquet(DD / "pullback_revyoy.parquet")

# 价值: PE_TTM 季末快照按日价重建 → EP=1/PE
mp = pd.read_parquet(D / "market_panel.parquet"); mp["symbol"] = mp["symbol"].astype(str).str.zfill(6); mp = mp[mp.symbol.isin(set(syms))]
def q2d(col):
    q = mp.pivot_table(index="trade_date", columns="symbol", values=col).reindex(columns=syms)
    return q.reindex(index=idx.union(q.index)).sort_index().ffill().reindex(idx)
ratio = close / q2d("adj_close")
pe = q2d("pe_ttm") / ratio                 # 日频PE
ep = (1.0 / pe).replace([np.inf, -np.inf], np.nan)
# 2026段用daily_2026真实pe
p26 = pd.read_parquet(D / "daily_2026.parquet"); p26["symbol"] = p26["symbol"].astype(str).str.zfill(6)
pe26 = p26.pivot_table(index="trade_date", columns="symbol", values="pe_ttm").reindex(columns=syms)
m26 = idx >= pd.Timestamp("2026-01-01")
ep.loc[m26] = (1.0 / pe26.reindex(idx[m26])).replace([np.inf, -np.inf], np.nan)
ep.to_parquet(DD / "pullback_ep.parquet")
print(f"成长/价值补全: pftyoy{pft.shape} revyoy{rev.shape} ep{ep.shape}")
print("覆盖率:", {"pft": round(pft.notna().mean().mean(), 2), "rev": round(rev.notna().mean().mean(), 2), "ep": round(ep.notna().mean().mean(), 2)})
