"""幅度感知目标搜索：为 Top-N 策略找比 rank/raw 都好的训练目标。

对比对象(参照): 旧版raw单视野(策略夏普0.75) / 新版rank多视野(0.72)。
候选(都多视野5/10/20集成): raw / xs-zscore(横截面标准化) / signed-sqrt。
评价改以「策略指标(夏普/Calmar) + Pearson IC」为主,RankIC 仅供参考。

跑法：python examples/ml_target_search.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from examples.ml_alpha import build_features
from examples.ml_trade import simulate, metrics, syms
from examples.strategy_family import idx, MKT, mv, ep, pft
from quantlab import barra

HORIZONS = (5, 10, 20)


def tf_raw(r):
    return r.clip(-0.5, 0.5)


def tf_zscore(r):
    r = r.clip(-0.5, 0.5)
    z = r.sub(r.mean(axis=1), axis=0).div(r.std(axis=1) + 1e-9, axis=0)
    return z.clip(-3, 3)


def tf_ssqrt(r):
    r = r.clip(-0.5, 0.5)
    return np.sign(r) * np.sqrt(r.abs())


TFS = {"raw": tf_raw, "xs-zscore": tf_zscore, "signed-sqrt": tf_ssqrt}


def train(transform, close, F, step=2):
    feats = list(F)
    purge = max(HORIZONS)
    ic_label = (close.shift(-10) / close - 1).clip(-0.5, 0.5)
    days = [d for d in idx if d >= pd.Timestamp("2018-01-01")]; tr_days = days[::step]
    datas = {}
    for h in HORIZONS:
        lab = transform((close.shift(-h) / close - 1))
        rows = []
        for d in tr_days:
            df = pd.DataFrame({k: v.loc[d] for k, v in F.items()})
            df["y"] = lab.loc[d]; df["date"] = d
            rows.append(df)
        datas[h] = pd.concat(rows).dropna(subset=["y"])
    pred = pd.DataFrame(np.nan, index=idx, columns=syms)
    ipos = {d: i for i, d in enumerate(idx)}
    months = sorted({(d.year, d.month) for d in idx if d.year >= 2021})
    ric, pic = [], []
    for (Y, M) in months:
        te = [d for d in idx if d.year == Y and d.month == M]
        if not te:
            continue
        ci = ipos[te[0]] - purge
        if ci <= 0:
            continue
        cut = idx[ci]; models = []
        for h in HORIZONS:
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
            a, b = p.corr(ic_label.loc[d], method="spearman"), p.corr(ic_label.loc[d], method="pearson")
            if pd.notna(a):
                ric.append(a)
            if pd.notna(b):
                pic.append(b)
    return pred, pd.Series(ric), pd.Series(pic)


def main():
    print("构建因子 ...", flush=True)
    F, _l, close = build_features()
    F = {k: v.reindex(index=idx, columns=syms) for k, v in F.items()}
    ret_d = close.pct_change(fill_method=None)
    style = barra.build_style_factors(ret_d.shift(-1).clip(-0.21, 0.21), market=MKT,
                                      logmv=np.log(mv.clip(lower=1)), ep=ep, mom=close / close.shift(60) - 1,
                                      vol=ret_d.rolling(20).std(), growth=pft)
    print(f"\n{'目标(多视野)':14s} {'RankIC':>7s} {'PearsonIC':>9s} {'年化':>6s} {'夏普':>6s} {'回撤':>7s} {'Calmar':>7s} {'αt':>5s}", flush=True)
    print(f"{'[参照]旧raw单视野':14s} {'0.0477':>7s} {'0.0571':>9s} {'+14%':>6s} {'0.75':>6s} {'-29%':>7s} {'0.49':>7s} {'0.81':>5s}")
    print(f"{'[参照]新rank多视野':14s} {'0.0837':>7s} {'0.0534':>9s} {'+10%':>6s} {'0.72':>6s} {'-31%':>7s} {'0.33':>7s} {'1.16':>5s}")
    best = None
    for nm, tf in TFS.items():
        pred, ric, pic = train(tf, close, F)
        port, tr = simulate(pred, hold=10, realistic=True, exclude_st=True)
        m = metrics(port); b = barra.barra_exposure(port, style)
        print(f"{nm:14s} {ric.mean():>7.4f} {pic.mean():>9.4f} {m['cagr']*100:>+5.0f}% {m['sharpe']:>6.2f} "
              f"{m['maxdd']*100:>+6.0f}% {m['calmar']:>7.2f} {b['alpha_t']:>5.2f}", flush=True)
        if best is None or m["sharpe"] > best[1]:
            best = (nm, m["sharpe"], pred)
    print(f"\n最佳(按策略夏普): {best[0]} 夏普 {best[1]:.2f}", flush=True)
    best[2].to_parquet("/home/claudeuser/econ/quant-research-lab/dashboard_data/ml_signal_best.parquet")
    print("最佳信号存 ml_signal_best.parquet")


if __name__ == "__main__":
    main()
