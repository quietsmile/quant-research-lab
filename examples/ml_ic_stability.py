"""IC 稳定性对比：除 IC 均值外，报 IC标准差、ICIR(=均值/标准差,稳定性)、年化ICIR、
IC>0 胜率、t值，并在不同模型/配方间横向对比。

口径：季度扩张窗口+purge(20),逐日 IC=Spearman(预测, 未来10日收益)。保留逐日 IC 序列以算 ICIR。
跑法：python examples/ml_ic_stability.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from examples.ml_alpha import build_features
from examples.strategy_family import idx, mv

syms = list(mv.columns)
ANN = 242


def xs_rank(panel):
    return panel.rank(axis=1, pct=True)


def stack(Fdict, label_panel, step):
    days = [d for d in idx if d >= pd.Timestamp("2018-01-01")]
    rows = []
    for d in days[::step]:
        df = pd.DataFrame({k: v.loc[d] for k, v in Fdict.items()})
        df["y"] = label_panel.loc[d]; df["date"] = d
        rows.append(df)
    return pd.concat(rows).dropna(subset=["y"])


def ic_series(Fdict, label_panels, ic_label, step=3, purge=20, extra_feat=None):
    """label_panels: 面板或list(多视野集成)。extra_feat: 额外因子面板dict(并入X,测增量)。返回逐日IC的Series。"""
    if not isinstance(label_panels, (list, tuple)):
        label_panels = [label_panels]
    Fd = dict(Fdict)
    if extra_feat:
        Fd.update(extra_feat)
    feats = list(Fd)
    datas = [stack(Fd, lp, step) for lp in label_panels]
    ipos = {d: i for i, d in enumerate(idx)}
    periods = sorted({(d.year, (d.month - 1) // 3) for d in idx if d.year >= 2021})
    P = dict(n_estimators=200, num_leaves=31, learning_rate=0.03, subsample=0.8,
             colsample_bytree=0.7, min_child_samples=100, n_jobs=4, verbosity=-1)
    recs = []
    for (Y, Q) in periods:
        te_days = [d for d in idx if d.year == Y and (d.month - 1) // 3 == Q]
        if not te_days:
            continue
        cut_i = ipos[te_days[0]] - purge
        if cut_i <= 0:
            continue
        cut = idx[cut_i]
        models = []
        for data in datas:
            tr = data[data["date"] <= cut]
            if len(tr) < 4000:
                continue
            med = tr[feats].median()
            m = lgb.LGBMRegressor(**P); m.fit(tr[feats].fillna(med), tr["y"])
            models.append((m, med))
        if not models:
            continue
        for d in te_days:
            base = pd.DataFrame({k: v.loc[d] for k, v in Fd.items()})
            preds = [pd.Series(m.predict(base.fillna(md)[feats]), index=syms) for m, md in models]
            pred = pd.concat(preds, axis=1).mean(axis=1)
            ic = pred.corr(ic_label.loc[d], method="spearman")
            if pd.notna(ic):
                recs.append((d, ic))
    s = pd.Series({d: v for d, v in recs}).sort_index()
    return s


def stats(s):
    m, sd, n = s.mean(), s.std(), len(s)
    icir = m / (sd + 1e-12)
    return dict(IC=m, ICstd=sd, ICIR=icir, ICIR_ann=icir * np.sqrt(ANN), win=(s > 0).mean(),
                t=icir * np.sqrt(n), n=n)


def main():
    print("构建因子 ...", flush=True)
    F, _label, close = build_features()
    F = {k: v.reindex(index=idx, columns=syms) for k, v in F.items()}
    lab10 = (close.shift(-10) / close - 1).clip(-0.5, 0.5).reindex(index=idx, columns=syms)
    lab5 = (close.shift(-5) / close - 1).clip(-0.5, 0.5).reindex(index=idx, columns=syms)
    lab20 = (close.shift(-20) / close - 1).clip(-0.5, 0.5).reindex(index=idx, columns=syms)
    r5, r10, r20 = xs_rank(lab5), xs_rank(lab10), xs_rank(lab20)

    models = {
        "M0 baseline(原始特征/原始标签)": ic_series(F, lab10, lab10),
        "M1 标签rank(单视野10d)": ic_series(F, r10, lab10),
        "M2 多视野rank集成(5/10/20)[生产]": ic_series(F, [r5, r10, r20], lab10),
    }
    print(f"\n{'模型':32s} {'IC均值':>7s} {'IC标准差':>8s} {'ICIR':>6s} {'年化ICIR':>8s} {'胜率':>6s} {'t值':>6s}", flush=True)
    for name, s in models.items():
        st = stats(s)
        print(f"{name:32s} {st['IC']:>7.4f} {st['ICstd']:>8.4f} {st['ICIR']:>6.3f} "
              f"{st['ICIR_ann']:>8.2f} {st['win']*100:>5.0f}% {st['t']:>6.1f}", flush=True)
    # 存逐日IC供看板画稳定性
    out = pd.DataFrame({k: v for k, v in models.items()})
    out.to_parquet("/home/claudeuser/econ/quant-research-lab/dashboard_data/ic_stability.parquet")
    print("\n逐日IC序列存 ic_stability.parquet (供看板对比)", flush=True)


if __name__ == "__main__":
    main()
