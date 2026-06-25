"""决定性检验:信号的多空(dollar-neutral)收益,判断 RankIC 0.08 是否「真」。

Top-20 多头夏普 0.7 ≠ 信号没用——多头丢了 breadth、扛 beta。正确检验是看
宽截面多空: ① 日频 rank 加权多空 ② 10日调仓十分位价差 ③ 五分位单调性。
若多空夏普高 → IC 真实,只是 Top-N 多头 harvest 得差;若也低 → IC 确实虚。

跑法：python examples/ml_signal_longshort.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from examples.ml_trade import load_signal, _ohlc, syms
from examples.strategy_family import idx
ANN = 242


def sharpe(p):
    p = p.dropna()
    return p.mean() / (p.std() + 1e-12) * np.sqrt(ANN) if len(p) else np.nan


def main():
    pred = load_signal()
    _o = _ohlc(); cl = _o[3]
    ret = cl.pct_change(fill_method=None)               # 日收益
    oos = [d for d in idx if d.year >= 2021]
    pred = pred.reindex(oos); ret = ret.reindex(oos)

    # ① 日频 rank 加权多空(dollar-neutral, gross=1)
    r = pred.rank(axis=1, pct=True).sub(0.5)            # 截面去均值
    w = r.div(r.abs().sum(axis=1) + 1e-12, axis=0)      # sum|w|=1 → 多/空各 0.5
    pnl = (w.shift(1) * ret).sum(axis=1)                # 次日生效,无前视
    turn = (w - w.shift(1)).abs().sum(axis=1) / 2
    cost = 0.0015
    pnl_net = pnl - turn * cost * 2
    print("===== 信号多空检验(生产信号 rank/多视野, OOS 2021–2026) =====")
    print(f"① 日频 rank 加权多空: 夏普(毛) {sharpe(pnl):.2f} | 夏普(净,扣双边{cost*100:.2f}%×2) {sharpe(pnl_net):.2f} "
          f"| 年化(毛) {pnl.mean()*ANN*100:+.0f}% | 日均换手 {turn.mean()*100:.0f}%")

    # ② 10日调仓 十分位价差(top10% - bottom10%, 等权, 持有10日)
    H = 10; fwd = cl.shift(-H) / cl - 1                  # 未来10日收益
    rb = oos[::H]
    longs, shorts = [], []
    for d in rb:
        s = pred.loc[d].dropna(); f = fwd.loc[d]
        if len(s) < 50:
            continue
        q = s.rank(pct=True)
        lo = f.reindex(q[q >= 0.9].index).mean()
        sh = f.reindex(q[q <= 0.1].index).mean()
        if pd.notna(lo) and pd.notna(sh):
            longs.append(lo); shorts.append(sh)
    longs, shorts = np.array(longs), np.array(shorts)
    spread = longs - shorts
    sp_sharpe = spread.mean() / (spread.std() + 1e-12) * np.sqrt(ANN / H)   # 每H日一次
    print(f"② 10日调仓 十分位价差(top10%-bot10%): 每期均值 {spread.mean()*100:+.2f}% | "
          f"胜率 {(spread>0).mean()*100:.0f}% | 年化夏普 {sp_sharpe:.2f} | top {longs.mean()*100:+.2f}% bot {shorts.mean()*100:+.2f}%")

    # ③ 五分位单调性(各组未来10日平均收益,应单调递增)
    means = [[] for _ in range(5)]
    for d in rb:
        s = pred.loc[d].dropna(); f = fwd.loc[d]
        if len(s) < 50:
            continue
        grp = pd.qcut(s.rank(method="first"), 5, labels=False)
        for g in range(5):
            v = f.reindex(s.index[grp == g]).mean()
            if pd.notna(v):
                means[g].append(v)
    q_means = [np.mean(m) * 100 for m in means]
    print(f"③ 五分位未来10日平均收益(Q1低→Q5高): {[round(x,2) for x in q_means]}  (单调递增=信号有序)")
    print(f"\n判读: 若①日频多空夏普≫Top20多头(0.72) → IC真实,只是Top-N多头harvest差(丢breadth+扛beta);"
          f"\n      若多空夏普也低(~0.7) → IC确实虚。A股个股做空受限,多空主要作信号质量诊断(实盘走指数增强/期指对冲)。")


if __name__ == "__main__":
    main()
