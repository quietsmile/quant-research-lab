"""第二轮 IC 实验：锁定「原始特征 + 标签横截面rank」基座(第一轮胜出),继续调优。

测：高斯rank标签、超参(lr/trees/leaves/正则)、多种子集成、多视野(5/10/20日)集成。
口径同第一轮：季度扩张窗口+purge(10),IC=当日 Spearman(预测, 未来10日收益)。

跑法：python examples/ml_ic_experiments2.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import ndtri                       # 逆正态CDF,用于高斯rank
from examples.ml_alpha import build_features
from examples.strategy_family import idx, mv

syms = list(mv.columns)


def xs_rank(panel):
    return panel.rank(axis=1, pct=True)


def gauss_rank(panel):                                # 横截面rank → 高斯分位(平滑目标)
    r = panel.rank(axis=1, pct=True)
    return r.apply(lambda row: pd.Series(ndtri(row.clip(0.001, 0.999).values), index=row.index), axis=1)


def stack(Fdict, label_panel, step):
    days = [d for d in idx if d >= pd.Timestamp("2018-01-01")]
    rows = []
    for d in days[::step]:
        df = pd.DataFrame({k: v.loc[d] for k, v in Fdict.items()})
        df["y"] = label_panel.loc[d]; df["date"] = d
        rows.append(df)
    return pd.concat(rows).dropna(subset=["y"])


def run(Fdict, label_panels, ic_label, step=3, purge=10, params=None, seeds=(0,)):
    """label_panels: 单个面板或面板list(多视野集成,各训一模型平均预测)。seeds: 多种子集成。"""
    if not isinstance(label_panels, (list, tuple)):
        label_panels = [label_panels]
    feats = list(Fdict)
    datas = [stack(Fdict, lp, step) for lp in label_panels]
    ipos = {d: i for i, d in enumerate(idx)}
    periods = sorted({(d.year, (d.month - 1) // 3) for d in idx if d.year >= 2021})
    P = dict(n_estimators=300, num_leaves=31, learning_rate=0.03, subsample=0.8,
             colsample_bytree=0.7, min_child_samples=100, n_jobs=4, verbosity=-1)
    if params:
        P.update(params)
    ics = []
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
            for sd in seeds:
                m = lgb.LGBMRegressor(random_state=sd, **P)
                m.fit(tr[feats].fillna(med), tr["y"])
                models.append((m, med))
        if not models:
            continue
        for d in te_days:
            base = pd.DataFrame({k: v.loc[d] for k, v in Fdict.items()})
            preds = [pd.Series(m.predict(base.fillna(md)[feats]), index=syms) for m, md in models]
            pred = pd.concat(preds, axis=1).mean(axis=1)
            ic = pred.corr(ic_label.loc[d], method="spearman")
            if pd.notna(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan, len(ics)


def main():
    print("构建因子 ...", flush=True)
    F, _label, close = build_features()
    F = {k: v.reindex(index=idx, columns=syms) for k, v in F.items()}
    lab10 = (close.shift(-10) / close - 1).clip(-0.5, 0.5).reindex(index=idx, columns=syms)
    lab5 = (close.shift(-5) / close - 1).clip(-0.5, 0.5).reindex(index=idx, columns=syms)
    lab20 = (close.shift(-20) / close - 1).clip(-0.5, 0.5).reindex(index=idx, columns=syms)
    r10, r5, r20 = xs_rank(lab10), xs_rank(lab5), xs_rank(lab20)
    g10 = gauss_rank(lab10)

    exps = [
        ("D 基座: 标签rank(10d)", dict(label_panels=r10)),
        ("J 高斯rank标签", dict(label_panels=g10)),
        ("K 低lr0.02/600树", dict(label_panels=r10, params=dict(learning_rate=0.02, n_estimators=600))),
        ("L 浅leaves15", dict(label_panels=r10, params=dict(num_leaves=15))),
        ("M 深leaves63/minchild300", dict(label_panels=r10, params=dict(num_leaves=63, min_child_samples=300))),
        ("N 强正则reg5/alpha1", dict(label_panels=r10, params=dict(reg_lambda=5.0, reg_alpha=1.0))),
        ("O 3种子集成", dict(label_panels=r10, seeds=(0, 1, 2))),
        ("P 多视野集成rank(5/10/20)", dict(label_panels=[r5, r10, r20])),
        ("Q 多视野+3种子", dict(label_panels=[r5, r10, r20], seeds=(0, 1, 2))),
        ("R 高斯rank+低lr0.02/600+3种子", dict(label_panels=g10, params=dict(learning_rate=0.02, n_estimators=600), seeds=(0, 1, 2))),
    ]
    print(f"{'变体':34s} {'样本外IC':>9s} {'天数':>6s}", flush=True)
    res = {}
    for name, kw in exps:
        ic, n = run(F, ic_label=lab10, **kw)
        res[name] = ic
        print(f"{name:34s} {ic:>9.4f} {n:>6d}", flush=True)
    best = max(res, key=lambda k: res[k] if pd.notna(res[k]) else -9)
    print(f"\n最佳(季度代理): {best} = {res[best]:.4f}  (基座 D {res[list(res)[0]]:.4f})")


if __name__ == "__main__":
    main()
