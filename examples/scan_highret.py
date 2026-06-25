"""在 min_mv=50亿 下扫参数, 找出'年化70%+'的配置, 看它们是不是靠极少持股/个别小盘暴涨(过拟合/运气)。
完全复刻看板 render_hotmoney 的计算逻辑。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib, itertools
DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data"); ANN = 242
g = lambda n: pd.read_parquet(DD / f"pullback_{n}.parquet")
close = g("close"); ret = g("ret"); sret = g("sret"); roe = g("roe"); mv = g("mv"); bench = g("bench"); sector = g("sector")["industry"]
for x in [close, ret, sret, roe, mv, bench]: x.index = pd.to_datetime(x.index)
zl = pd.read_parquet(DD / "moneyflow_zlrate.parquet"); zl.index = pd.to_datetime(zl.index)
syms = list(close.columns); zl = zl.reindex(columns=syms)
idx = close.index[close.index >= "2020-01-01"]
close, ret, sret, roe, mv = [x.reindex(idx) for x in (close, ret, sret, roe, mv)]; zl = zl.reindex(idx)
rfwd = ret.shift(-1).clip(-0.21, 0.21)
smap = sector.reindex(syms).values

def bt(hot_win, topK, maw, tol, rising, mf_win, mf_thr, rebal, min_mv=500000):
    snav = (1 + sret.fillna(0)).cumprod()
    hot = (snav / snav.shift(hot_win) - 1).rank(axis=1, ascending=False) <= topK
    sh = hot.reindex(columns=smap); sh.columns = syms; sh = sh.reindex(idx).fillna(False)
    ma = close.rolling(maw, min_periods=max(3, maw // 2)).mean()
    trend = (close >= ma * (1 - tol)) & ((ma > ma.shift(5)) if rising else True)
    mf = zl.rolling(mf_win, min_periods=1).mean() > mf_thr
    elig = sh & trend & mf & (mv > min_mv) & ret.notna()
    rb = np.zeros(len(idx), bool); rb[::rebal] = True
    held = elig.astype(float).where(pd.Series(rb, index=idx), np.nan).ffill().fillna(0.0)
    w = held.div(held.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    port = (w * rfwd).sum(axis=1) - (w - w.shift(1)).abs().sum(axis=1) * 0.001
    nh = (w > 0).sum(axis=1)
    p = port.fillna(0); nav = (1 + p).cumprod(); cagr = nav.iloc[-1] ** (ANN / len(p)) - 1
    sh_ = p.mean() / (p.std() + 1e-12) * np.sqrt(ANN)
    by = {d.year: round(((1 + p[p.index.year == d.year]).prod() - 1) * 100) for d in p.index[::240]}
    return cagr, sh_, nh.mean(), by

print("复刻看板默认配置(hot30,topK5,ma20,tol2%,rising,mf5,thr0,rebal3):")
print("  ", bt(30, 5, 20, 0.02, True, 5, 0.0, 3)[:3])
print("\n扫描找'年化>=50%'的配置 (min_mv=50亿):")
grid = itertools.product([20, 30], [3, 5, 8], [10, 20], [0.0, 0.03, 0.06], [True, False], [1, 5], [0.0, 0.05], [1, 3])
hits = []
for hw, tk, mw, tol, ri, mfw, mft, rb in grid:
    cagr, shp, nh, by = bt(hw, tk, mw, tol, ri, mfw, mft, rb)
    if cagr > 0.50:
        hits.append((cagr, shp, nh, dict(hot=hw, topK=tk, ma=mw, tol=tol, rising=ri, mfwin=mfw, thr=mft, rebal=rb), by))
hits.sort(key=lambda x: x[0], reverse=True)
print(f"找到 {len(hits)} 个年化>50%的配置。展示Top6(看持股数和逐年):")
for cagr, shp, nh, cfg, by in hits[:6]:
    print(f"  年化{cagr*100:.0f}% 夏普{shp:.2f} 平均持股{nh:.1f} | {cfg} | 逐年{by}")
