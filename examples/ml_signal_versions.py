"""新旧信号版本对比：IC/ICIR + 策略表现 + Barra 暴露。

- 旧版: 月度扩张窗口, 原始标签(未来10日收益), 单视野
- 新版: 月度扩张窗口, 标签横截面rank + 多视野(5/10/20)集成  [生产]
两者同一套后处理(simulate hold=10, 真实撮合)与同一 Barra 风格因子,横向对比。

跑法：python examples/ml_signal_versions.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from examples.ml_alpha import build_features
from examples.ml_trade import simulate, metrics, _xs_rank, syms, ANN
from examples.strategy_family import idx, MKT, mv, ep, pft
from quantlab import barra


def train(recipe, close, F, step=2):
    """recipe: 'old'(原始标签/单视野) 或 'new'(rank标签/多视野5,10,20)。返回 (pred_panel, ic_series)。"""
    feats = list(F)
    horizons = (10,) if recipe == "old" else (5, 10, 20)
    use_rank = recipe == "new"
    purge = max(horizons)
    ic_label = (close.shift(-10) / close - 1).clip(-0.5, 0.5)        # IC 基准:未来10日收益
    days = [d for d in idx if d >= pd.Timestamp("2018-01-01")]
    tr_days = days[::step]
    datas = {}
    for h in horizons:
        lab = (close.shift(-h) / close - 1).clip(-0.5, 0.5)
        if use_rank:
            lab = _xs_rank(lab)
        rows = []
        for d in tr_days:
            df = pd.DataFrame({k: v.loc[d] for k, v in F.items()})
            df["y"] = lab.loc[d]; df["date"] = d
            rows.append(df)
        datas[h] = pd.concat(rows).dropna(subset=["y"])
    pred = pd.DataFrame(np.nan, index=idx, columns=syms)
    ipos = {d: i for i, d in enumerate(idx)}
    months = sorted({(d.year, d.month) for d in idx if d.year >= 2021})
    rank_ic, pear_ic = [], []
    for (Y, M) in months:
        te = [d for d in idx if d.year == Y and d.month == M]
        if not te:
            continue
        ci = ipos[te[0]] - purge
        if ci <= 0:
            continue
        cut = idx[ci]; models = []
        for h in horizons:
            tr = datas[h][datas[h]["date"] <= cut]
            if len(tr) < 4000:
                continue
            med = tr[feats].median()
            m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.03, subsample=0.8,
                                  colsample_bytree=0.7, min_child_samples=100, n_jobs=4, verbosity=-1)
            m.fit(tr[feats].fillna(med), tr["y"]); models.append((m, med))
        if not models:
            continue
        for d in te:
            base = pd.DataFrame({k: v.loc[d] for k, v in F.items()})
            p = pd.concat([pd.Series(mm.predict(base.fillna(md)[feats]), index=syms) for mm, md in models], axis=1).mean(axis=1)
            pred.loc[d] = p.values
            ric = p.corr(ic_label.loc[d], method="spearman")      # RankIC
            pic = p.corr(ic_label.loc[d], method="pearson")       # IC(Pearson)
            if pd.notna(ric):
                rank_ic.append(ric)
            if pd.notna(pic):
                pear_ic.append(pic)
    return pred, pd.Series(rank_ic), pd.Series(pear_ic)


def ic_stats(s):
    m, sd, n = s.mean(), s.std(), len(s)
    return m, sd, m / (sd + 1e-12), (s > 0).mean(), (m / (sd + 1e-12)) * np.sqrt(n)


def main():
    print("构建因子 ...", flush=True)
    F, _l, close = build_features()
    F = {k: v.reindex(index=idx, columns=syms) for k, v in F.items()}
    ret_d = close.pct_change(fill_method=None)
    style = barra.build_style_factors(ret_d.shift(-1).clip(-0.21, 0.21), market=MKT,
                                      logmv=np.log(mv.clip(lower=1)), ep=ep, mom=close / close.shift(60) - 1,
                                      vol=ret_d.rolling(20).std(), growth=pft)
    out = {}
    for recipe, name in [("old", "旧版(原始标签/单视野)"), ("new", "新版(rank/多视野)[生产]")]:
        print(f"\n训练 {name} ...", flush=True)
        pred, ric, pic = train(recipe, close, F)
        rm, rsd, ricir, rwin, rt = ic_stats(ric)        # RankIC(Spearman)
        pm, psd, picir, pwin, pt = ic_stats(pic)        # IC(Pearson)
        port, tr = simulate(pred, hold=10, realistic=True, exclude_st=True)
        mt = metrics(port)
        b = barra.barra_exposure(port, style)
        out[name] = dict(rankic=rm, rankic_std=rsd, rankicir=ricir, rankwin=rwin, rankt=rt,
                         ic=pm, ic_std=psd, icir=picir, win=pwin, t=pt, port=port, tr=tr, mt=mt, b=b)
        print(f"  RankIC {rm:.4f}(ICIR {ricir:.3f}) | IC(Pearson) {pm:.4f}(IR {picir:.3f}) | "
              f"年化 {mt['cagr']*100:+.0f}% 夏普 {mt['sharpe']:.2f} 回撤 {mt['maxdd']*100:+.0f}%")

    print("\n" + "=" * 92)
    print(f"{'指标':18s} {'旧版(原始/单视野)':>22s} {'新版(rank/多视野)':>22s}")
    o = out["旧版(原始标签/单视野)"]; n = out["新版(rank/多视野)[生产]"]
    rows = [("RankIC均值(Spearman)", f"{o['rankic']:.4f}", f"{n['rankic']:.4f}"),
            ("RankIC标准差", f"{o['rankic_std']:.4f}", f"{n['rankic_std']:.4f}"),
            ("RankICIR", f"{o['rankicir']:.3f}", f"{n['rankicir']:.3f}"),
            ("RankIC>0胜率", f"{o['rankwin']*100:.0f}%", f"{n['rankwin']*100:.0f}%"),
            ("RankIC t值", f"{o['rankt']:.1f}", f"{n['rankt']:.1f}"),
            ("IC均值(Pearson)", f"{o['ic']:.4f}", f"{n['ic']:.4f}"),
            ("IC标准差(Pearson)", f"{o['ic_std']:.4f}", f"{n['ic_std']:.4f}"),
            ("ICIR(Pearson)", f"{o['icir']:.3f}", f"{n['icir']:.3f}"),
            ("IC t值(Pearson)", f"{o['t']:.1f}", f"{n['t']:.1f}"),
            ("策略年化", f"{o['mt']['cagr']*100:+.0f}%", f"{n['mt']['cagr']*100:+.0f}%"),
            ("策略夏普", f"{o['mt']['sharpe']:.2f}", f"{n['mt']['sharpe']:.2f}"),
            ("最大回撤", f"{o['mt']['maxdd']*100:+.0f}%", f"{n['mt']['maxdd']*100:+.0f}%"),
            ("Calmar", f"{o['mt']['calmar']:.2f}", f"{n['mt']['calmar']:.2f}"),
            ("Barra α年化", f"{o['b']['alpha_ann']*100:+.1f}%", f"{n['b']['alpha_ann']*100:+.1f}%"),
            ("Barra α t值", f"{o['b']['alpha_t']:.2f}", f"{n['b']['alpha_t']:.2f}"),
            ("Barra R²", f"{o['b']['r2']*100:.0f}%", f"{n['b']['r2']*100:.0f}%")]
    for lab, a, c in rows:
        print(f"{lab:16s} {a:>22s} {c:>22s}")
    print("\n--- 旧版 Barra 暴露 ---\n " + barra.format_exposure(o["b"]))
    print("--- 新版 Barra 暴露 ---\n " + barra.format_exposure(n["b"]))
    # 存净值供看板对比
    pd.DataFrame({"旧版": (1 + o["port"].fillna(0)).cumprod(),
                  "新版": (1 + n["port"].fillna(0)).cumprod()}).to_parquet(
        "/home/claudeuser/econ/quant-research-lab/dashboard_data/ml_signal_versions_nav.parquet")
    print("\n净值存 ml_signal_versions_nav.parquet")


if __name__ == "__main__":
    main()
