"""提升样本外 IC 的受控实验：逐个测试横截面标准化/中性化/特征/集成等技巧。

口径：季度扩张窗口 walk-forward + purge(10日),严格 PIT;IC = 当日 Spearman(预测, 未来10日收益)。
为快速排序技巧用 step=3 采样、季度重训作代理;胜出组合再用全口径(月度)确认。

跑法：python examples/ml_ic_experiments.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from examples.ml_alpha import build_features
from examples.strategy_family import idx, mv

syms = list(mv.columns)
LOGMV = np.log(mv.clip(lower=1)).reindex(index=idx, columns=syms)


def xs_rank(panel):                      # 当日横截面 rank → [0,1]
    return panel.rank(axis=1, pct=True)


def xs_z(panel):                         # 当日横截面 zscore
    return panel.sub(panel.mean(axis=1), axis=0).div(panel.std(axis=1) + 1e-9, axis=0)


def xs_neutralize_size(panel):           # 当日对 logmv 回归取残差(去市值)
    out = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    for d in panel.index:
        y = panel.loc[d]; x = LOGMV.loc[d]
        m = y.notna() & x.notna()
        if m.sum() < 30:
            continue
        xv = x[m].values; yv = y[m].values
        b = np.polyfit(xv, yv, 1)
        out.loc[d, m[m].index] = yv - (b[0] * xv + b[1])
    return out


def stack(Fdict, label_panel, step):
    days = [d for d in idx if d >= pd.Timestamp("2018-01-01")]
    rows = []
    for d in days[::step]:
        df = pd.DataFrame({k: v.loc[d] for k, v in Fdict.items()})
        df["y"] = label_panel.loc[d]; df["date"] = d
        rows.append(df)
    return pd.concat(rows).dropna(subset=["y"])


def run_variant(Fdict, label_panel, ic_label, step=3, purge=10, params=None):
    """季度扩张窗口;返回 (mean_ic, n_days)。ic_label=用于测 IC 的真实未来收益面板。"""
    feats = list(Fdict)
    data = stack(Fdict, label_panel, step)
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
        tr = data[data["date"] <= idx[cut_i]]
        if len(tr) < 4000:
            continue
        med = tr[feats].median()
        m = lgb.LGBMRegressor(**P); m.fit(tr[feats].fillna(med), tr["y"])
        for d in te_days:
            X = pd.DataFrame({k: v.loc[d] for k, v in Fdict.items()}).fillna(med)
            pred = pd.Series(m.predict(X[feats]), index=syms)
            ic = pred.corr(ic_label.loc[d], method="spearman")
            if pd.notna(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan, len(ics)


def main():
    print("构建因子 ...", flush=True)
    F, _label, close = build_features()
    F = {k: v.reindex(index=idx, columns=syms) for k, v in F.items()}
    lab_ret = (close.shift(-10) / close - 1).clip(-0.5, 0.5).reindex(index=idx, columns=syms)  # 真实未来10日收益(测IC基准)

    F_rank = {k: xs_rank(v) for k, v in F.items()}
    F_z = {k: xs_z(v) for k, v in F.items()}
    lab_z = xs_z(lab_ret)
    lab_rank = lab_ret.rank(axis=1, pct=True)
    lab_neut = xs_neutralize_size(lab_ret)

    exps = [
        ("0 baseline(原始特征/原始标签)", F, lab_ret, None),
        ("A 特征横截面rank", F_rank, lab_ret, None),
        ("B 特征横截面zscore", F_z, lab_ret, None),
        ("C 标签横截面zscore", F, lab_z, None),
        ("D 标签横截面rank", F, lab_rank, None),
        ("E 特征rank+标签zscore", F_rank, lab_z, None),
        ("F 特征rank+标签rank", F_rank, lab_rank, None),
        ("G 特征rank+标签去市值中性", F_rank, lab_neut, None),
        ("H 特征rank+标签z+更深(leaves63)", F_rank, lab_z, dict(num_leaves=63, min_child_samples=200)),
        ("I 特征rank+标签z+强正则", F_rank, lab_z, dict(reg_lambda=5.0, reg_alpha=1.0, min_child_samples=200)),
    ]
    print(f"{'变体':36s} {'样本外IC':>9s} {'天数':>6s}", flush=True)
    res = {}
    for name, Fd, lab, prm in exps:
        ic, n = run_variant(Fd, lab, lab_ret, step=3, params=prm)
        res[name] = ic
        print(f"{name:36s} {ic:>9.4f} {n:>6d}", flush=True)
    best = max(res, key=lambda k: res[k] if pd.notna(res[k]) else -9)
    print(f"\n最佳(季度代理口径): {best} = {res[best]:.4f}  (baseline {res[list(res)[0]]:.4f})")


if __name__ == "__main__":
    main()
