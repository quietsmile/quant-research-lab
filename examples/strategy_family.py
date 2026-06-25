"""策略族(每个带可调参数空间) + 网格搜索 + 参数敏感性 + Barra 风格暴露。

在'热门板块+趋势+主力资金+质量'主题上设计 6 个策略，每个有显式参数空间；
对每个策略网格搜索，强制分散(topn≥15)，按**最差子区间夏普**挑稳健最优，
报告参数敏感性(全网格夏普中位/最差)，并对最优配置做 **Barra 多因子暴露**。

数据：dashboard_data/(2017-2026, liq1500, 含主力资金 2020 起)；回测 2020 起、含成本。
跑法：python examples/strategy_family.py
"""
import warnings; warnings.filterwarnings("ignore")
import itertools, json, pathlib
import numpy as np, pandas as pd

from quantlab import barra

DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data"); ANN = 242
L = lambda n: pd.read_parquet(DD / f"pullback_{n}.parquet")
close, ret, sret, roe, mv, bench = L("close"), L("ret"), L("sret"), L("roe"), L("mv"), L("bench")
sector = L("sector")["industry"]; pft, ep = L("pftyoy"), L("ep")
for x in [close, ret, sret, roe, mv, bench, pft, ep]: x.index = pd.to_datetime(x.index)
zl = pd.read_parquet(DD / "moneyflow_zlrate.parquet"); zl.index = pd.to_datetime(zl.index)
syms = list(close.columns); zl = zl.reindex(columns=syms)
idx = close.index[close.index >= "2020-01-01"]
close, ret, sret, roe, mv, pft, ep = [x.reindex(idx) for x in (close, ret, sret, roe, mv, pft, ep)]
zl = zl.reindex(idx); rfwd = ret.shift(-1).clip(-0.21, 0.21)
smap = sector.reindex(syms); logmv = np.log(mv.clip(lower=1))
ma20 = close.rolling(20, 10).mean(); trend = (close >= ma20) & (ma20 > ma20.shift(5))
zl20 = zl.rolling(20, min_periods=5).mean()
snav = (1 + sret.fillna(0)).cumprod(); sec_mom = snav / snav.shift(20) - 1
zl_sec = pd.DataFrame({g: zl[[s for s in syms if smap.get(s) == g]].mean(axis=1) for g in sret.columns}).reindex(idx)
zl_sec20 = zl_sec.rolling(20, min_periods=5).mean()
def map_sec(b): o = b.reindex(columns=smap.values); o.columns = syms; return o.reindex(idx).fillna(False)
hot_mom = lambda k: map_sec(sec_mom.rank(axis=1, ascending=False) <= k)
hot_flow = lambda k: map_sec(zl_sec20.rank(axis=1, ascending=False) <= k)
q = (roe > 0) & (pft > 0) & (mv > 5e5)
mom = lambda w: close / close.shift(w) - 1
vol = lambda w: ret.rolling(w).std()
rk = lambda d: d.rank(axis=1, pct=True)
comp = rk(zl20) + rk(pft) + rk(close / ma20)


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


MKT = bench["等权全市场"].reindex(idx)
def metr(p, nh):
    p = p.fillna(0); nav = (1 + p).cumprod()
    sub = lambda m: ((1 + p[m]).prod() ** (ANN / max(m.sum(), 1)) - 1,
                     p[m].mean() / (p[m].std() + 1e-12) * np.sqrt(ANN),
                     ((1 + p[m]).cumprod() / (1 + p[m]).cumprod().cummax() - 1).min())
    cg, sh, dd = sub(p.index >= "2020"); _, sh1, _ = sub(p.index <= "2022-12-31"); _, sh2, _ = sub(p.index >= "2023")
    by = {d.year: round(((1 + p[p.index.year == d.year]).prod() - 1) * 100) for d in p.index[::240]}
    return dict(cagr=cg, sharpe=sh, maxdd=dd, calmar=cg / abs(dd), sh1=sh1, sh2=sh2,
                worst=min(by.values()) if by else 0, nh=float(nh.mean()), by=by)


# ---------- 6 个策略，每个一套参数空间 ----------
def strat(name, sel_fn, score_fn):
    return dict(name=name, sel=sel_fn, score=score_fn)

STRATS = {
 "S1 质量+动量":   (lambda P: trend & q, lambda P: mom(P["mom_win"]),
                   dict(topn=[15,20,25,30], rebal=[5,10,20], mom_win=[20,40,60])),
 "S2 质量+价值EP": (lambda P: trend & q, lambda P: ep,
                   dict(topn=[15,20,25,30], rebal=[5,10,20])),
 "S3 质量+低波":   (lambda P: trend & q, lambda P: -vol(P["vol_win"]),
                   dict(topn=[15,20,25,30], rebal=[5,10,20], vol_win=[20,40])),
 "S4 板块轮动+质量+复合": (lambda P: (hot_mom(P["hot_k"]) | hot_flow(P["hot_k"])) & trend & q, lambda P: comp,
                   dict(hot_k=[6,8,10], topn=[20,25,30], rebal=[5,20])),
 "S5 规模中性+质量+复合": (lambda P: trend & q, lambda P: comp,   # 实际打分在 main 里用 size_neutral 覆盖
                   dict(topn=[20,25,30], rebal=[5,20])),
 "S6 多因子融合":  (lambda P: trend & q, lambda P: rk(mom(40)) + rk(ep) + rk(-vol(20)) + rk(zl20) + rk(pft),
                   dict(topn=[20,25,30], rebal=[5,20])),
}


def size_neutral(score):
    # 截面对 logmv 回归取残差(规模中性)
    out = score.copy()
    for d in idx:
        s = score.loc[d]; x = logmv.loc[d]
        m = s.notna() & x.notna()
        if m.sum() > 30:
            b = np.polyfit(x[m], s[m], 1); out.loc[d, m.index[m]] = s[m] - (b[1] + b[0] * x[m])
    return out


def grid(space):
    keys = list(space);
    return [dict(zip(keys, v)) for v in itertools.product(*[space[k] for k in keys])]


def main():
    # Barra 风格因子(日频)
    style = barra.build_style_factors(rfwd, market=MKT, logmv=logmv, ep=ep,
                                      mom=mom(60), vol=vol(20), growth=pft)
    print("Barra 风格因子:", list(style.columns), "| 期数", len(style), "\n")

    results = {}
    for name, (sel_fn, score_fn, space) in STRATS.items():
        configs = [c for c in grid(space) if c["topn"] >= 15]   # 强制分散
        rows = []
        for P in configs:
            sel = sel_fn(P)
            sc = score_fn(P)
            if name.startswith("S5"):
                sc = size_neutral(comp)
            p, nh = run(sel, sc, P["topn"], P.get("rebal", 5))
            if nh.mean() < 15:
                continue
            m = metr(p, nh); m["P"] = P; m["port"] = p
            rows.append(m)
        if not rows:
            continue
        # 稳健最优：最差子区间夏普最大
        best = max(rows, key=lambda r: min(r["sh1"], r["sh2"]))
        shs = [r["sharpe"] for r in rows]
        b = barra.barra_exposure(best["port"], style)
        results[name] = dict(
            n_configs=len(rows), best_param=best["P"],
            cagr=best["cagr"], sharpe=best["sharpe"], maxdd=best["maxdd"], calmar=best["calmar"],
            sh1=best["sh1"], sh2=best["sh2"], worst=best["worst"], nh=best["nh"], by=best["by"],
            grid_sharpe_median=float(np.median(shs)), grid_sharpe_min=float(np.min(shs)),
            grid_robust_pct=float(np.mean([(r["sh1"] > 0) and (r["sh2"] > 0) for r in rows])),
            barra=b)
        print(f"{name}: 最优{best['P']}")
        print(f"  年化{best['cagr']*100:+.0f}% 夏普{best['sharpe']:.2f} 回撤{best['maxdd']*100:+.0f}% Calmar{best['calmar']:.2f} "
              f"子区间{best['sh1']:.2f}/{best['sh2']:.2f} 持股{best['nh']:.0f}")
        print(f"  参数敏感: 全网格夏普中位{np.median(shs):.2f} 最差{np.min(shs):.2f} 两子区间均正占比{results[name]['grid_robust_pct']:.0%}")
        print("  " + barra.format_exposure(b) + "\n")

    # 等权基准 + 存盘
    bn = (1 + MKT.fillna(0)).cumprod()
    bench_m = dict(cagr=bn.iloc[-1] ** (ANN / len(MKT)) - 1,
                   sharpe=float(MKT.mean() / MKT.std() * np.sqrt(ANN)),
                   maxdd=float((bn / bn.cummax() - 1).min()),
                   barra=barra.barra_exposure(MKT, style))
    out = {"strategies": {k: {kk: vv for kk, vv in v.items() if kk != "port"} for k, v in results.items()},
           "benchmark": {k: v for k, v in bench_m.items()}}
    json.dump(out, open(DD / "strategy_family.json", "w"), ensure_ascii=False, default=float)
    print(f"等权基准: 年化{bench_m['cagr']*100:+.0f}% 夏普{bench_m['sharpe']:.2f} 回撤{bench_m['maxdd']*100:+.0f}%")
    print("结果已存 dashboard_data/strategy_family.json")


if __name__ == "__main__":
    main()
