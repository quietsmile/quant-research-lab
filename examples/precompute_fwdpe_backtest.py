"""'管理层 forward PE' 策略 PIT 回测数据底座。
对每个月末×股票: 取最新管理层业绩预告(announce<=月末)→年化forward净利→forward PE、指引增速;
附 行业/周期标记/质量(扣非ROE)/月末市值/次月收益。存 dashboard_data/fwdpe_panel.parquet。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib
D = pathlib.Path.home() / ".local/share/quantlab/fundamentals"
OUT = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data"); OUT.mkdir(exist_ok=True)

ind = pd.read_parquet(D / "industry.parquet"); ind["symbol"] = ind["symbol"].astype(str).str.zfill(6)
CYC_KW = ["钢", "煤", "有色", "化工", "化纤", "农药化肥", "水泥", "玻璃", "建材", "工程机械",
          "汽车", "船舶", "航运", "航空", "养殖", "种植", "石油", "石化", "橡胶", "造纸",
          "采掘", "金属", "矿", "电力", "港口", "高速", "房地产", "银行", "证券", "保险", "饲料"]
def is_cyc(x):
    x = x if isinstance(x, str) else ""
    return any(k in x for k in CYC_KW)
ind["is_cyc"] = ind["industry"].map(is_cyc)
imap = ind.set_index("symbol")

oh = pd.read_parquet(D / "daily_ohlcv.parquet")[["trade_date", "symbol", "adj_close", "amount"]]
oh["symbol"] = oh["symbol"].astype(str).str.zfill(6)
liq = set(oh.groupby("symbol")["amount"].mean().nlargest(2500).index)
oh = oh[oh.symbol.isin(liq)]; syms = sorted(liq)
close = oh.pivot_table(index="trade_date", columns="symbol", values="adj_close").reindex(columns=syms)
me = close.resample("M").last()                      # 月末价
me = me[me.index >= "2017-01-01"]
mret_fwd = me.pct_change().shift(-1)                  # 次月收益
month_ends = me.index

# 月末市值(季末快照ffill × 价格比)
mp = pd.read_parquet(D / "market_panel.parquet"); mp["symbol"] = mp["symbol"].astype(str).str.zfill(6); mp = mp[mp.symbol.isin(liq)]
def q2d_me(col):
    q = mp.pivot_table(index="trade_date", columns="symbol", values=col).reindex(columns=syms)
    daily = q.reindex(index=close.index.union(q.index)).sort_index().ffill().reindex(close.index)
    return daily.resample("M").last().reindex(month_ends)
mv_me = q2d_me("total_mv")
px_snap = q2d_me("adj_close"); ratio = me / px_snap
mv_me = mv_me * ratio                                # 月末市值(万元)

# 质量: 扣非ROE PIT
tf = pd.read_parquet(D / "tushare_features.parquet")[["symbol", "announce_date", "roe_dedt"]].copy()
tf["symbol"] = tf["symbol"].astype(str).str.zfill(6); tf = tf[tf.symbol.isin(liq)].dropna(subset=["announce_date"]).sort_values("announce_date")
cal = pd.DataFrame({"announce_date": month_ends})
roe = {}
for s, g in tf.groupby("symbol"):
    g = g.dropna(subset=["roe_dedt"]).sort_values("announce_date")
    if len(g): roe[s] = pd.merge_asof(cal, g[["announce_date", "roe_dedt"]], on="announce_date").set_index("announce_date")["roe_dedt"]
roe = pd.DataFrame(roe).reindex(index=month_ends, columns=syms)

# 管理层业绩预告 → 年化forward净利 + 指引增速 (PIT asof)
fc = pd.read_parquet(D / "forecast.parquet").copy()
fc["symbol"] = fc["symbol"].astype(str).str.zfill(6); fc = fc[fc.symbol.isin(liq)]
fc = fc.dropna(subset=["ann_date", "end_date"]).sort_values("ann_date")
fc["np_mid"] = fc[["net_profit_min", "net_profit_max"]].mean(axis=1)
fc["g_mid"] = fc[["p_change_min", "p_change_max"]].mean(axis=1)
ann_factor = {3: 4.0, 6: 2.0, 9: 4 / 3, 12: 1.0}
fc["np_ann"] = fc["np_mid"] * fc["end_date"].dt.month.map(ann_factor)   # 年化forward净利(万元)
fc = fc.dropna(subset=["np_ann"])

# 每月末asof最新预告
rows = []
for s, g in fc.groupby("symbol"):
    g = g.sort_values("ann_date")
    m = pd.merge_asof(pd.DataFrame({"ann_date": month_ends}), g[["ann_date", "np_ann", "g_mid", "end_date"]],
                      on="ann_date", direction="backward")
    m["symbol"] = s; rows.append(m)
gpanel = pd.concat(rows, ignore_index=True).rename(columns={"ann_date": "date"})
gpanel["announce_age_d"] = np.nan  # 由merge保留的最近ann未单列, 近似用end_date新鲜度

# 组装长面板
recs = []
mv_stack = mv_me.stack(); roe_stack = roe.stack(); ret_stack = mret_fwd.reindex(month_ends).stack()
gpanel = gpanel.set_index(["date", "symbol"])
for (d, s), g in gpanel.iterrows():
    if pd.isna(g["np_ann"]) or g["np_ann"] <= 0:
        continue
    mv = mv_me.loc[d, s] if s in mv_me.columns else np.nan
    if pd.isna(mv) or mv <= 0:
        continue
    info = imap.loc[s] if s in imap.index else None
    recs.append({"date": d, "symbol": s,
                 "industry": info["industry"] if info is not None else "UNK",
                 "is_cyc": bool(info["is_cyc"]) if info is not None else False,
                 "mv": mv, "fwd_NP": g["np_ann"], "fwd_PE": mv / g["np_ann"],
                 "guided_g": g["g_mid"], "guide_end": g["end_date"],
                 "roe": roe.loc[d, s] if s in roe.columns else np.nan,
                 "ret_1m": mret_fwd.loc[d, s] if (d in mret_fwd.index and s in mret_fwd.columns) else np.nan})
panel = pd.DataFrame(recs)
# 指引新鲜度: 预告期末距当前(月)
panel["guide_age_m"] = ((panel["date"] - panel["guide_end"]).dt.days / 30).round(1)
panel.to_parquet(OUT / "fwdpe_panel.parquet", index=False)
# 基准月收益
bench = pd.DataFrame({"等权全市场": me.pct_change().mean(axis=1)}).reindex(month_ends)
bench.to_parquet(OUT / "fwdpe_bench.parquet")
print(f"面板 {panel.shape} 月份{panel.date.nunique()} 股票{panel.symbol.nunique()} 周期股占比{panel.is_cyc.mean():.2f}")
print("forward PE 分布:", panel.fwd_PE.describe(percentiles=[.1,.5,.9]).round(1).to_dict())
