"""(A) 乐观度拆解:同一信号,逐步松绑严格项,看年化如何虚高。

回答「30-70%年化里有多少是air/beta、多少是真本事」。每步在前一步基础上再松一档:
严格基线 → 去真实撮合/ST → 去成本 → 日频调仓 → 下沉小盘 → 样本内(泄漏)。

跑法：python examples/optimism_budget.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from examples.ml_trade import load_signal, simulate, metrics, syms, _xs_rank, HORIZONS
from examples.ml_alpha import build_features
from examples.strategy_family import idx, mv

mvr = mv.reindex(index=idx, columns=syms).rank(axis=1, pct=True)   # 市值百分位


def train_insample(step=2):
    """样本内泄漏对照:一个模型用「全期」数据训练、预测同一期(无walk-forward)。"""
    F, _l, close = build_features()
    F = {k: v.reindex(index=idx, columns=syms) for k, v in F.items()}
    feats = list(F)
    days = [d for d in idx if d >= pd.Timestamp("2018-01-01")][::step]
    preds = []
    for h in HORIZONS:
        lab = _xs_rank((close.shift(-h) / close - 1).clip(-0.5, 0.5))
        rows = [pd.DataFrame({**{k: v.loc[d] for k, v in F.items()}, "y": lab.loc[d]}) for d in days]
        data = pd.concat(rows).dropna(subset=["y"])
        med = data[feats].median()
        m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.03, subsample=0.8,
                              colsample_bytree=0.7, min_child_samples=100, n_jobs=4, verbosity=-1)
        m.fit(data[feats].fillna(med), data["y"])              # 全期训练(含未来)
        pp = pd.DataFrame(np.nan, index=idx, columns=syms)
        for d in idx:
            pp.loc[d] = m.predict(pd.DataFrame({k: v.loc[d] for k, v in F.items()}).fillna(med)[feats])
        preds.append(pp)
    return sum(preds) / len(preds)


def main():
    pred = load_signal()
    pred_small = pred.where(mvr <= 0.5)                        # 只保留小市值半区
    print("构建样本内泄漏信号(全期训练) ...", flush=True)
    pred_in = train_insample()
    pred_in_small = pred_in.where(mvr <= 0.5)

    steps = [
        ("0 严格基线(OOS/大中盘/真实撮合/扣成本/T+1开盘/hold10)",
         dict(pred=pred, hold=10, realistic=True, exclude_st=True, cost=0.0015, use_fund=True)),
        ("1 +理想撮合(去涨跌停/停牌/ST约束)",
         dict(pred=pred, hold=10, realistic=False, exclude_st=False, cost=0.0015, use_fund=True)),
        ("2 +去成本(毛收益)",
         dict(pred=pred, hold=10, realistic=False, exclude_st=False, cost=0.0, use_fund=True)),
        ("3 +日频调仓(hold=1)",
         dict(pred=pred, hold=1, realistic=False, exclude_st=False, cost=0.0, use_fund=True)),
        ("4 +下沉小盘(liq1500内小市值半区)",
         dict(pred=pred_small, hold=1, realistic=False, exclude_st=False, cost=0.0, use_fund=True)),
        ("5 +样本内信号(全期训练,泄漏)",
         dict(pred=pred_in_small, hold=1, realistic=False, exclude_st=False, cost=0.0, use_fund=True)),
    ]
    print(f"\n{'松绑步骤':46s} {'年化':>6s} {'夏普':>6s} {'回撤':>7s}", flush=True)
    base = None
    for name, kw in steps:
        port, tr = simulate(top_n=20, **kw)
        m = metrics(port)
        tag = "" if base is None else f"  (累计×{(1+m['cagr'])/(1+base):.1f})"
        if base is None:
            base = m["cagr"]
        print(f"{name:46s} {m['cagr']*100:>+5.0f}% {m['sharpe']:>6.2f} {m['maxdd']*100:>+6.0f}%{tag}", flush=True)
    print("\n判读: 从基线到「样本内+小盘+日频+无成本」的年化跳升,几乎全是 air/beta/泄漏。"
          "\n      别人 30-70% 的年化,大概率就是停在了这条链的某个松绑档位上。")


if __name__ == "__main__":
    main()
