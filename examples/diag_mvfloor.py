"""诊断: 热门板块+主力资金策略对'市值下限'敏感, 是真小盘alpha / 持股太少退化 / 还是bug?
对比不同市值下限的 CAGR/夏普/平均持股/持仓中位市值; 并对策略收益做 Barra(市场+SMB)回归看规模暴露。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib
import statsmodels.api as sm
DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data"); ANN = 242
g = lambda n: pd.read_parquet(DD / f"pullback_{n}.parquet")
close = g("close"); ret = g("ret"); sret = g("sret"); mv = g("mv"); bench = g("bench"); sector = g("sector")["industry"]
for x in [close, ret, sret, mv, bench]: x.index = pd.to_datetime(x.index)
zl = pd.read_parquet(DD / "moneyflow_zlrate.parquet"); zl.index = pd.to_datetime(zl.index)
syms = list(close.columns); zl = zl.reindex(columns=syms)
idx = close.index[close.index >= "2020-01-01"]
close, ret, sret, mv = [x.reindex(idx) for x in (close, ret, sret, mv)]; zl = zl.reindex(idx)

# 默认参数
snav = (1 + sret.fillna(0)).cumprod()
hot = (snav / snav.shift(30) - 1).rank(axis=1, ascending=False) <= 5
smap = sector.reindex(syms).values; sh = hot.reindex(columns=smap); sh.columns = syms; sh = sh.reindex(idx).fillna(False)
ma = close.rolling(20, min_periods=10).mean()
trend = (close >= ma) & (ma > ma.shift(5))
mf = zl.rolling(5, min_periods=1).mean() > 0
rfwd = ret.shift(-1).clip(-0.21, 0.21); rebal = 3
logmv = np.log(mv.clip(lower=1))

def run(min_mv):
    elig = sh & trend & mf & (mv > min_mv) & ret.notna()
    rb = np.zeros(len(idx), bool); rb[::rebal] = True
    held = elig.astype(float).where(pd.Series(rb, index=idx), np.nan).ffill().fillna(0.0)
    w = held.div(held.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    port = (w * rfwd).sum(axis=1) - (w - w.shift(1)).abs().sum(axis=1) * 0.001
    nh = (w > 0).sum(axis=1)
    med_mc = (w.replace(0, np.nan) * 0 + mv).where(w > 0).median(axis=1).median() / 1e4   # 持仓中位市值(亿)
    return port, nh.mean(), med_mc

print("市值下限敏感性 (默认: 热门top5板块+价>20MA上行+主力流入, 3日调仓):")
print(f"{'下限':>8} {'年化':>7} {'夏普':>6} {'平均持股':>8} {'持仓中位市值(亿)':>14}")
ports = {}
for f_yi in [0, 30, 50, 80, 100, 150, 200]:
    port, nh, medmc = run(f_yi * 1e4)
    ports[f_yi] = port
    p = port.fillna(0); nav = (1 + p).cumprod(); cagr = nav.iloc[-1] ** (ANN / len(p)) - 1
    shp = p.mean() / (p.std() + 1e-12) * np.sqrt(ANN)
    print(f"{f_yi:>6}亿 {cagr*100:>+6.0f}% {shp:>+6.2f} {nh:>8.1f} {medmc:>14.0f}")

# Barra: SMB(小-大 tercile LS) + 市场, 回归50亿版策略
def smb_daily(d):
    x = logmv.loc[d].dropna()
    if len(x) < 60: return np.nan
    n = int(len(x) / 3)
    return rfwd.loc[d, x.nsmallest(n).index].mean() - rfwd.loc[d, x.nlargest(n).index].mean()
SMB = pd.Series({d: smb_daily(d) for d in idx})
MKT = bench["等权全市场"].reindex(idx)
print("\nBarra回归 (策略=50亿下限版; 因子=市场MKT + 规模SMB[小-大]):")
for tag, fy in [("50亿", 50), ("100亿", 100)]:
    dat = pd.concat([ports[fy].rename("y"), MKT.rename("MKT"), SMB.rename("SMB")], axis=1).dropna()
    m = sm.OLS(dat["y"], sm.add_constant(dat[["MKT", "SMB"]])).fit()
    print(f"  [{tag}] R²={m.rsquared:.2f} 市场beta={m.params['MKT']:+.2f} SMB beta={m.params['SMB']:+.2f}(t={m.tvalues['SMB']:.1f}) alpha年化={m.params['const']*ANN*100:+.1f}%(t={m.tvalues['const']:.1f})")
