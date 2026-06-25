"""搜索迭代2: 保留'财报质量+板块资金流'的优势, 但强制分散(持股≥15)看edge是否还在。
低持股(F1=7,F3=4)的高夏普不可信; 这里只认 持股≥15 的稳健配置。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib, json
import statsmodels.api as sm
DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data"); ANN = 242
L = lambda n: pd.read_parquet(DD / f"pullback_{n}.parquet")
close = L("close"); ret = L("ret"); sret = L("sret"); roe = L("roe"); mv = L("mv"); bench = L("bench"); sector = L("sector")["industry"]
pft = L("pftyoy"); rev = L("revyoy"); ep = L("ep")
for x in [close, ret, sret, roe, mv, bench, pft, rev, ep]: x.index = pd.to_datetime(x.index)
zl = pd.read_parquet(DD / "moneyflow_zlrate.parquet"); zl.index = pd.to_datetime(zl.index)
syms = list(close.columns); zl = zl.reindex(columns=syms)
idx = close.index[close.index >= "2020-01-01"]
close, ret, sret, roe, mv, pft, rev, ep = [x.reindex(idx) for x in (close, ret, sret, roe, mv, pft, rev, ep)]
zl = zl.reindex(idx); rfwd = ret.shift(-1).clip(-0.21, 0.21)
smap = sector.reindex(syms); logmv = np.log(mv.clip(lower=1)); vol20 = ret.rolling(20).std()
snav = (1 + sret.fillna(0)).cumprod(); sec_mom = snav / snav.shift(20) - 1
zl_sec = pd.DataFrame({g: zl[[s for s in syms if smap.get(s) == g]].mean(axis=1) for g in sret.columns}).reindex(idx)
zl_sec20 = zl_sec.rolling(20, min_periods=5).mean()
ma20 = close.rolling(20, min_periods=10).mean(); trend = (close >= ma20) & (ma20 > ma20.shift(5))
zl20 = zl.rolling(20, min_periods=5).mean()
def map_sec(b): o = b.reindex(columns=smap.values); o.columns = syms; return o.reindex(idx).fillna(False)
hot_mom = lambda k: map_sec(sec_mom.rank(axis=1, ascending=False) <= k)
hot_flow = lambda k: map_sec(zl_sec20.rank(axis=1, ascending=False) <= k)

def run(select, score, topn, rebal=5, vol_target=None, cost=0.0015):
    rb = np.zeros(len(idx), bool); rb[::rebal] = True
    scc = score.where(select & ret.notna(), np.nan)
    chosen = pd.DataFrame(False, index=idx, columns=syms)
    for d in idx[rb]:
        r = scc.loc[d].dropna()
        if len(r): chosen.loc[d, r.nlargest(topn).index] = True
    held = chosen.where(pd.Series(rb, index=idx), np.nan).ffill().fillna(False).astype(bool)
    w = held.astype(float); w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    port = (w * rfwd).sum(axis=1) - (w - w.shift(1)).abs().sum(axis=1) * cost
    if vol_target:
        rv = port.rolling(20).std().shift(1) * np.sqrt(ANN); port = port * (vol_target / rv).clip(upper=1.5).fillna(0.0)
    return port, (w > 0).sum(axis=1)

SMB = pd.Series({d: (rfwd.loc[d, logmv.loc[d].dropna().nsmallest(int(logmv.loc[d].count() / 3)).index].mean()
                     - rfwd.loc[d, logmv.loc[d].dropna().nlargest(int(logmv.loc[d].count() / 3)).index].mean()) for d in idx})
MKT = bench["等权全市场"].reindex(idx)
def metrics(p, nh):
    p = p.fillna(0); nav = (1 + p).cumprod()
    def sub(m):
        x = p[m]; nv = (1 + x).cumprod(); return nv.iloc[-1] ** (ANN / len(x)) - 1, x.mean() / (x.std() + 1e-12) * np.sqrt(ANN), (nv / nv.cummax() - 1).min()
    cg, sh, dd = sub(p.index >= "2020"); _, sh1, _ = sub((p.index >= "2020") & (p.index <= "2022-12-31")); _, sh2, _ = sub(p.index >= "2023")
    by = {d.year: round(((1 + p[p.index.year == d.year]).prod() - 1) * 100) for d in p.index[::240]}
    dat = pd.concat([p.rename("y"), MKT.rename("m"), SMB.rename("s")], axis=1).dropna()
    smb = sm.OLS(dat["y"], sm.add_constant(dat[["m", "s"]])).fit().params["s"]
    return dict(cagr=cg, sharpe=sh, maxdd=dd, calmar=cg / abs(dd), sh1=sh1, sh2=sh2, worst=min(by.values()), nh=nh.mean(), smb=smb, by=by)

q = (roe > 0) & (pft > 0) & (mv > 5e5)              # 财报质量底线
comp = lambda: zl20.rank(axis=1, pct=True) + pft.rank(axis=1, pct=True) + (close / ma20).rank(axis=1, pct=True)
cands = {
 "H1 板块资金流+质量, 打分主力, top20": run(hot_flow(10) & trend & q, zl20, 20),
 "H2 板块资金流+质量, 复合打分, top20": run(hot_flow(10) & trend & q, comp(), 20),
 "H3 板块(动量∪资金流)+质量复合, top25": run((hot_mom(8) | hot_flow(8)) & trend & q, comp(), 25),
 "H4 板块动量+质量, 复合打分, top20": run(hot_mom(10) & trend & q, comp(), 20),
 "H5 H3 + vol目标22%": run((hot_mom(8) | hot_flow(8)) & trend & q, comp(), 25, vol_target=0.22),
 "H6 H3 + 月度调仓(降换手)": run((hot_mom(8) | hot_flow(8)) & trend & q, comp(), 25, rebal=20),
}
print(f"{'策略':36} {'年化':>6} {'夏普':>5} {'回撤':>6} {'Calmar':>6} {'夏普2022':>8} {'夏普2326':>8} {'最差年':>6} {'持股':>5} {'SMB':>5}")
out = {}
for name, (p, nh) in cands.items():
    m = metrics(p, nh); out[name] = m
    print(f"{name:36} {m['cagr']*100:>+5.0f}% {m['sharpe']:>+5.2f} {m['maxdd']*100:>+5.0f}% {m['calmar']:>6.2f} {m['sh1']:>+8.2f} {m['sh2']:>+8.2f} {m['worst']:>+5.0f}% {m['nh']:>5.0f} {m['smb']:>+5.2f}")
b = MKT.fillna(0); bn = (1 + b).cumprod()
print(f"{'等权基准':36} {bn.iloc[-1]**(ANN/len(b))*100-100:>+5.0f}% {b.mean()/b.std()*np.sqrt(ANN):>+5.2f} {(bn/bn.cummax()-1).min()*100:>+5.0f}%")
json.dump(out, open(DD / "search2_results.json", "w"), default=float, ensure_ascii=False, indent=1)
print("\n冠军逐年:", out[max(out, key=lambda k: out[k]['calmar'] if out[k]['nh']>=15 else -9)]['by'])
PY = None
