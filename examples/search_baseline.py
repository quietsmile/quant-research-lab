"""搜索更强、更稳健的基线: 在'热门板块+趋势+主力资金'基础上, 系统比较多个策略族
(财报质量过滤/价值过滤/板块资金流/复合打分/风控仓位/规模中性)。
判据=稳健: 夏普/最大回撤/Calmar/逐年一致性/SMB暴露/持股数 + 子区间(20-22 vs 23-26)。
全部强制分散(板块≥6, 持股≥15, 5日调仓, 50亿下限)以防集中度运气。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib
import statsmodels.api as sm
DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data"); ANN = 242
L = lambda n: pd.read_parquet(DD / f"pullback_{n}.parquet")
close = L("close"); ret = L("ret"); sret = L("sret"); roe = L("roe"); mv = L("mv")
bench = L("bench"); sector = L("sector")["industry"]
pft = L("pftyoy"); rev = L("revyoy"); ep = L("ep")
for x in [close, ret, sret, roe, mv, bench, pft, rev, ep]: x.index = pd.to_datetime(x.index)
zl = pd.read_parquet(DD / "moneyflow_zlrate.parquet"); zl.index = pd.to_datetime(zl.index)
syms = list(close.columns); zl = zl.reindex(columns=syms)
idx = close.index[close.index >= "2020-01-01"]
close, ret, sret, roe, mv, pft, rev, ep = [x.reindex(idx) for x in (close, ret, sret, roe, mv, pft, rev, ep)]
zl = zl.reindex(idx)
rfwd = ret.shift(-1).clip(-0.21, 0.21)
smap = sector.reindex(syms)
logmv = np.log(mv.clip(lower=1))
vol20 = ret.rolling(20).std()

# ---- 公共信号 ----
snav = (1 + sret.fillna(0)).cumprod()
sec_mom = snav / snav.shift(20) - 1                         # 板块20日动量
# 板块主力资金流(成分股主力净流入率均值)
zl_sec = pd.DataFrame({g: zl[[s for s in syms if smap.get(s) == g]].mean(axis=1) for g in sret.columns}).reindex(idx)
zl_sec_20 = zl_sec.rolling(20, min_periods=5).mean()
ma20 = close.rolling(20, min_periods=10).mean()
trend = (close >= ma20) & (ma20 > ma20.shift(5))
zl20 = zl.rolling(20, min_periods=5).mean()                 # 个股主力20日(持续吸筹)
zl5 = zl.rolling(5, min_periods=1).mean()

def map_sec(sec_bool):
    o = sec_bool.reindex(columns=smap.values); o.columns = syms; return o.reindex(idx).fillna(False)

def hot_by_mom(topK=6): return map_sec(sec_mom.rank(axis=1, ascending=False) <= topK)
def hot_by_flow(topK=6): return map_sec(zl_sec_20.rank(axis=1, ascending=False) <= topK)

# ---- 回测: 给每日选择(bool)或打分(float), 等权/风险加权, 含成本 ----
def run(select=None, score=None, topn=25, rebal=5, weight="ew", vol_target=None, cap=0.12, cost=0.0015):
    rb = np.zeros(len(idx), bool); rb[::rebal] = True
    if score is not None:
        sc = score.where(select, np.nan) if select is not None else score
        chosen = pd.DataFrame(False, index=idx, columns=syms)
        for d in idx[rb]:
            r = sc.loc[d].dropna()
            if len(r): chosen.loc[d, r.nlargest(topn).index] = True
        base = chosen
    else:
        base = select & ret.notna()
    held = base.astype(float).where(pd.Series(rb, index=idx), np.nan).ffill().fillna(0.0).astype(bool)
    if weight == "ew":
        w = held.astype(float)
    else:  # 逆波动加权
        w = held.astype(float) * (1.0 / vol20.replace(0, np.nan))
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if cap: w = w.clip(upper=cap); w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    port = (w * rfwd).sum(axis=1) - (w - w.shift(1)).abs().sum(axis=1) * cost
    if vol_target:
        rv = port.rolling(20).std().shift(1) * np.sqrt(ANN)
        lev = (vol_target / rv).clip(upper=1.5).fillna(0.0)
        port = port * lev
    return port, (w > 0).sum(axis=1)

SMB = pd.Series({d: (rfwd.loc[d, logmv.loc[d].dropna().nsmallest(int(logmv.loc[d].count()/3)).index].mean()
                     - rfwd.loc[d, logmv.loc[d].dropna().nlargest(int(logmv.loc[d].count()/3)).index].mean())
                 for d in idx})
MKT = bench["等权全市场"].reindex(idx)

def metrics(port, nh):
    p = port.fillna(0); nav = (1 + p).cumprod()
    def sub(mask):
        x = p[mask]; nv = (1 + x).cumprod()
        cg = nv.iloc[-1] ** (ANN / len(x)) - 1; sh = x.mean() / (x.std() + 1e-12) * np.sqrt(ANN)
        dd = (nv / nv.cummax() - 1).min(); return cg, sh, dd
    cg, sh, dd = sub(p.index >= "2020-01-01")
    cg1, sh1, dd1 = sub((p.index >= "2020-01-01") & (p.index <= "2022-12-31"))
    cg2, sh2, dd2 = sub(p.index >= "2023-01-01")
    by = {d.year: round(((1 + p[p.index.year == d.year]).prod() - 1) * 100) for d in p.index[::240]}
    posyr = np.mean([v > 0 for v in by.values()])
    dat = pd.concat([p.rename("y"), MKT.rename("m"), SMB.rename("s")], axis=1).dropna()
    smb_b = sm.OLS(dat["y"], sm.add_constant(dat[["m", "s"]])).fit().params["s"]
    return dict(cagr=cg, sharpe=sh, maxdd=dd, calmar=cg / abs(dd) if dd < 0 else np.nan,
                sh_2022=sh1, sh_2326=sh2, worst_yr=min(by.values()), pos_yr=posyr,
                nh=nh.mean(), smb=smb_b, by=by)

# ---- 策略族 ----
strats = {}
strats["F0 基线(板块动量+趋势+主力)"] = run(select=hot_by_mom(6) & trend & (zl5 > 0) & (mv > 5e5))
strats["F1 +财报质量(ROE>0&利润增>0)"] = run(select=hot_by_mom(6) & trend & (zl20 > 0) & (mv > 5e5) & (roe > 0) & (pft > 0))
strats["F2 +质量+价值(剔最贵20%)"] = run(select=hot_by_mom(6) & trend & (zl20 > 0) & (mv > 5e5) & (roe > 0) & (pft > 0) & (ep.rank(axis=1, pct=True) > 0.2))
strats["F3 板块按主力资金流选热门"] = run(select=hot_by_flow(6) & trend & (zl20 > 0) & (mv > 5e5) & (roe > 0))
strats["F4 复合打分(主力+成长+趋势)"] = run(
    select=hot_by_mom(8) & trend & (mv > 5e5),
    score=zl20.rank(axis=1, pct=True) + pft.rank(axis=1, pct=True) + (close / ma20).rank(axis=1, pct=True), topn=25)
strats["F5 复合打分+逆波动+vol目标25%"] = run(
    select=hot_by_mom(8) & trend & (mv > 5e5) & (roe > 0),
    score=zl20.rank(axis=1, pct=True) + pft.rank(axis=1, pct=True), topn=25, weight="iv", vol_target=0.25)
# F6 规模中性: 在每个市值三分组内各取打分最高, 合并(剥离SMB)
def size_neutral():
    sc = zl20.rank(axis=1, pct=True) + pft.rank(axis=1, pct=True) + (close / ma20).rank(axis=1, pct=True)
    cand = hot_by_mom(8) & trend & (mv > 5e5)
    scc = sc.where(cand, np.nan)
    chosen = pd.DataFrame(False, index=idx, columns=syms)
    rb = np.zeros(len(idx), bool); rb[::5] = True
    for d in idx[rb]:
        r = scc.loc[d].dropna()
        if len(r) < 6: continue
        sz = logmv.loc[d, r.index]
        for grp in [r[sz <= sz.quantile(.33)], r[(sz > sz.quantile(.33)) & (sz <= sz.quantile(.67))], r[sz > sz.quantile(.67)]]:
            if len(grp): chosen.loc[d, grp.nlargest(8).index] = True
    held = chosen.where(pd.Series(rb, index=idx), np.nan).ffill().fillna(False).astype(bool)
    w = held.astype(float); w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    port = (w * rfwd).sum(axis=1) - (w - w.shift(1)).abs().sum(axis=1) * 0.0015
    return port, (w > 0).sum(axis=1)
strats["F6 规模中性复合打分"] = size_neutral()

print(f"{'策略':30} {'年化':>6} {'夏普':>5} {'回撤':>6} {'Calmar':>6} {'夏普20-22':>9} {'夏普23-26':>9} {'最差年':>6} {'持股':>5} {'SMB':>5}")
rows = {}
for name, (port, nh) in strats.items():
    m = metrics(port, nh); rows[name] = m
    print(f"{name:30} {m['cagr']*100:>+5.0f}% {m['sharpe']:>+5.2f} {m['maxdd']*100:>+5.0f}% {m['calmar']:>6.2f} "
          f"{m['sh_2022']:>+9.2f} {m['sh_2326']:>+9.2f} {m['worst_yr']:>+5.0f}% {m['nh']:>5.0f} {m['smb']:>+5.2f}")
b = bench["等权全市场"].reindex(idx).fillna(0); bnav = (1 + b).cumprod()
print(f"{'等权基准':30} {bnav.iloc[-1]**(ANN/len(b))*100-100:>+5.0f}% {b.mean()/b.std()*np.sqrt(ANN):>+5.2f} {(bnav/bnav.cummax()-1).min()*100:>+5.0f}%")
import json
json.dump({k: {kk: (vv if kk != 'by' else vv) for kk, vv in v.items()} for k, v in rows.items()},
          open(DD / "search_results.json", "w"), default=float, ensure_ascii=False, indent=1)
