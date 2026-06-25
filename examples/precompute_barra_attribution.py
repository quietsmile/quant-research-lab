"""Barra式风格归因: 把等权策略收益回归到 市场+规模(SMB)+动量+低波+价值 风格因子上,
得出风格暴露(beta)、各风格年化贡献、残余alpha(截距,带t值)、R²。供看板展示。
存 barra_attr.json (各策略归因) + barra_rolling.parquet (滚动SMB暴露)。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib, json
import statsmodels.api as sm
D = pathlib.Path.home() / ".local/share/quantlab/fundamentals"; ANN = 242

oh = pd.read_parquet(D / "daily_ohlcv.parquet")[["trade_date", "symbol", "adj_close", "amount", "turnover_rate"]]
oh["symbol"] = oh["symbol"].astype(str).str.zfill(6)
p26 = pd.read_parquet(D / "daily_2026.parquet")[["trade_date", "symbol", "adj_close", "amount", "turnover_rate"]]
p26["symbol"] = p26["symbol"].astype(str).str.zfill(6)
liq = set(oh.groupby("symbol")["amount"].mean().nlargest(2000).index)
oh = oh[oh.symbol.isin(liq)]; p26 = p26[p26.symbol.isin(liq)]; syms = sorted(liq)
def W(df, c): return df.pivot_table(index="trade_date", columns="symbol", values=c)
close = pd.concat([W(oh, "adj_close"), W(p26, "adj_close")]); close = close[~close.index.duplicated()].sort_index().reindex(columns=syms)
ret = close.pct_change(fill_method=None); idx = [d for d in ret.index if d >= pd.Timestamp("2017-01-01")]
ret = ret.reindex(idx); close = close.reindex(idx)

# 风格描述子
mp = pd.read_parquet(D / "market_panel.parquet"); mp["symbol"] = mp["symbol"].astype(str).str.zfill(6); mp = mp[mp.symbol.isin(liq)]
def q2d(col):
    q = mp.pivot_table(index="trade_date", columns="symbol", values=col).reindex(columns=syms)
    return q.reindex(index=pd.DatetimeIndex(idx).union(q.index)).sort_index().ffill().reindex(idx)
SIZE = np.log(q2d("total_mv").clip(lower=1e-6))
VALUE = (1.0 / q2d("pb"))
MOM = close / close.shift(231) - 1
LOWVOL = -ret.rolling(120).std()

def factor_ret(desc, frac=0.3):     # 多空因子收益: 前frac − 后frac (等权), 日度
    r = pd.Series(index=idx, dtype=float)
    for d in idx:
        x = desc.loc[d].dropna()
        if len(x) < 50: continue
        n = int(len(x) * frac)
        top = x.nlargest(n).index; bot = x.nsmallest(n).index
        r[d] = ret.loc[d, top].mean() - ret.loc[d, bot].mean()
    return r
SMB = factor_ret(-SIZE)            # 小−大
HML = factor_ret(VALUE)            # 便宜−贵
WML = factor_ret(MOM)              # 强动量−弱
LMH = factor_ret(LOWVOL)           # 低波−高波
# 市场因子
nm = pd.read_parquet(D / "north_money.parquet") if (D / "north_money.parquet").exists() else None
from quantlab.data.tushare_adapter import get_pro
try:
    pro = get_pro(); hs = pro.index_daily(ts_code="000300.SH", start_date="20170101", end_date="20260620", fields="trade_date,close")
    hs["trade_date"] = pd.to_datetime(hs["trade_date"]); MKT = hs.sort_values("trade_date").set_index("trade_date")["close"].pct_change().reindex(idx)
except Exception:
    MKT = ret.mean(axis=1)
F = pd.DataFrame({"市场": MKT, "规模SMB": SMB, "价值HML": HML, "动量WML": WML, "低波LMH": LMH}).reindex(idx)

sv = pd.read_parquet(D / "strat_viz.parquet"); sv.index = pd.to_datetime(sv.index)
targets = ["每日再平衡等权", "月度再平衡等权(扣成本)"]
res = {}
for t in targets:
    dat = pd.concat([sv[t].rename("y"), F], axis=1).dropna()
    X = sm.add_constant(dat[F.columns]); m = sm.OLS(dat["y"], X).fit()
    betas = {k: float(m.params[k]) for k in F.columns}
    contrib = {k: float(m.params[k] * dat[k].mean() * ANN) for k in F.columns}   # 年化贡献
    res[t] = {"betas": betas, "alpha_ann": float(m.params["const"] * ANN), "alpha_t": float(m.tvalues["const"]),
              "r2": float(m.rsquared), "contrib": contrib, "factor_t": {k: float(m.tvalues[k]) for k in F.columns}}
    print(f"[{t}] R²={m.rsquared:.2f} alpha年化{m.params['const']*ANN*100:+.1f}%(t={m.tvalues['const']:.1f}) betas={ {k:round(v,2) for k,v in betas.items()} }")
json.dump(res, open(D / "barra_attr.json", "w"), ensure_ascii=False, indent=1)
# 滚动SMB暴露(1年窗)
roll = {}
for t in targets:
    dat = pd.concat([sv[t].rename("y"), F], axis=1).dropna()
    b = []
    Xall = sm.add_constant(dat[F.columns])
    for i in range(ANN, len(dat)):
        w = dat.iloc[i-ANN:i]
        mm = sm.OLS(w["y"], sm.add_constant(w[F.columns])).fit()
        b.append((dat.index[i], mm.params.get("规模SMB", np.nan), mm.params.get("市场", np.nan)))
    rr = pd.DataFrame(b, columns=["date", f"{t}_SMB", f"{t}_MKT"]).set_index("date")
    roll[t] = rr
rolldf = pd.concat(roll.values(), axis=1)
rolldf.to_parquet(D / "barra_rolling.parquet")
print("saved barra_attr.json + barra_rolling.parquet")
