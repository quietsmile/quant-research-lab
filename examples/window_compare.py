"""(B-2) 训练窗口:扩张窗 vs 滚动48月 vs 滚动24月,比 IC/ICIR。

用户提到"过去48个月训练"。A股非平稳,滚动定长窗可能比扩张窗更跟得上风格,也可能因数据少而更差。
同一多视野rank口径、季度代理,只改训练样本的时间范围。

跑法：python examples/window_compare.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from examples.ml_alpha import build_features
from examples.ml_ic_stability import xs_rank, stats
from examples.strategy_family import idx, mv

syms = list(mv.columns)
HOR = (5, 10, 20)


def run(lookback_m, step=3, purge=20):
    """lookback_m: None=扩张窗; 否则只用预测点前 lookback_m 个月的数据。返回逐日RankIC。"""
    F, _l, close = build_features()
    F = {k: v.reindex(index=idx, columns=syms) for k, v in F.items()}
    feats = list(F)
    ic_label = (close.shift(-10) / close - 1).clip(-0.5, 0.5)
    days = [d for d in idx if d >= pd.Timestamp("2018-01-01")][::step]
    datas = {}
    for h in HOR:
        lab = xs_rank((close.shift(-h) / close - 1).clip(-0.5, 0.5))
        datas[h] = pd.concat([pd.DataFrame({**{k: v.loc[d] for k, v in F.items()}, "y": lab.loc[d], "date": d})
                              for d in days]).dropna(subset=["y"])
    ipos = {d: i for i, d in enumerate(idx)}
    periods = sorted({(d.year, (d.month - 1) // 3) for d in idx if d.year >= 2021})
    ics = []
    for (Y, Q) in periods:
        te = [d for d in idx if d.year == Y and (d.month - 1) // 3 == Q]
        if not te:
            continue
        ci = ipos[te[0]] - purge
        if ci <= 0:
            continue
        cut = idx[ci]
        lo = cut - pd.DateOffset(months=lookback_m) if lookback_m else pd.Timestamp("2000-01-01")
        models = []
        for h in HOR:
            tr = datas[h][(datas[h]["date"] <= cut) & (datas[h]["date"] > lo)]
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
            ic = p.corr(ic_label.loc[d], method="spearman")
            if pd.notna(ic):
                ics.append(ic)
    return pd.Series(ics)


def main():
    print(f"{'训练窗口':16s} {'RankIC':>7s} {'ICstd':>7s} {'ICIR':>6s} {'胜率':>5s} {'t值':>6s}", flush=True)
    for nm, lb in [("扩张窗(现状)", None), ("滚动48月", 48), ("滚动36月", 36), ("滚动24月", 24)]:
        s = run(lb); st = stats(s)
        print(f"{nm:16s} {st['IC']:>7.4f} {st['ICstd']:>7.4f} {st['ICIR']:>6.3f} "
              f"{st['win']*100:>4.0f}% {st['t']:>6.1f}", flush=True)
    print("\n判读: 若滚动窗 IC/ICIR 高于扩张窗 → 非平稳下定长窗更优,值得换;若扩张窗最高 → 多数据胜过近因。")


if __name__ == "__main__":
    main()
