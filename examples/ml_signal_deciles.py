"""分层单调性检验(可落地、不依赖做空):区分 Top-20 夏普低是「breadth不足」还是「头部信号失效」。

- 全截面按预测分 10 层,看各层未来10日平均收益是否单调递增(每10日非重叠调仓,时间平均)。
- 关键看头部:Top-20≈top1.3%,住在极头部,故把头部细分 top1%/2%/5%/10%——
  若 top1% 仍是最高 → 头部健康,Top-20 低纯属 breadth(扩持仓可救);
  若 top1% < top10%(头部 roll over)→ Top-20 选的恰是信号最不可靠区,扩 breadth 也救不了。
- 纯信号 与 基本面池(trend&q,Top-20实际用的)各看一遍。

跑法：python examples/ml_signal_deciles.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from examples.ml_trade import load_signal, _ohlc, syms
from examples.strategy_family import idx, trend, q

H = 10


def layered(pred, cl, pool=None):
    fwd = cl.shift(-H) / cl - 1
    oos = [d for d in idx if d.year >= 2021]
    rb = oos[::H]
    dec = [[] for _ in range(10)]
    head = {k: [] for k in ["top1%", "top2%", "top5%", "top10%", "bot10%"]}
    base = []
    for d in rb:
        s = pred.loc[d].dropna()
        if pool is not None:
            ok = (trend.loc[d].reindex(s.index).fillna(False)) & (q.loc[d].reindex(s.index).fillna(False))
            s = s[ok]
        f = fwd.loc[d]
        if len(s) < 100:
            continue
        base.append(f.reindex(s.index).mean())
        r = s.rank(pct=True)
        g = np.clip((r.values * 10).astype(int), 0, 9)
        for j in range(10):
            v = f.reindex(s.index[g == j]).mean()
            if pd.notna(v):
                dec[j].append(v)
        for k, thr in [("top1%", 0.99), ("top2%", 0.98), ("top5%", 0.95), ("top10%", 0.90)]:
            v = f.reindex(s.index[r >= thr]).mean()
            if pd.notna(v):
                head[k].append(v)
        v = f.reindex(s.index[r <= 0.10]).mean()
        if pd.notna(v):
            head["bot10%"].append(v)
    dm = [np.mean(x) * 100 for x in dec]
    hm = {k: np.mean(v) * 100 for k, v in head.items()}
    return dm, hm, np.mean(base) * 100


def main():
    pred = load_signal(); cl = _ohlc()[3]
    for tag, pool in [("纯信号(全截面)", None), ("基本面池内(trend&q, Top-20实际用)", True)]:
        dm, hm, base = layered(pred, cl, pool)
        print(f"\n===== 分层单调性 · {tag} (未来{H}日平均收益%, 每{H}日非重叠调仓, 2021–2026) =====")
        print("  10层(D1低→D10高): " + " ".join(f"{x:+.2f}" for x in dm))
        mono = all(dm[i] <= dm[i + 1] + 0.05 for i in range(9))
        print(f"  截面基准(平均) {base:+.2f}% | 单调? {'是' if mono else '否(有逆转)'} | D10−D1 价差 {dm[-1]-dm[0]:+.2f}%")
        print(f"  头部细分: top10% {hm['top10%']:+.2f} | top5% {hm['top5%']:+.2f} | top2% {hm['top2%']:+.2f} | "
              f"top1% {hm['top1%']:+.2f}  (Top-20≈top1.3%)")
        if hm["top1%"] >= hm["top10%"]:
            print(f"  → 头部健康(top1% {hm['top1%']:+.2f} ≥ top10% {hm['top10%']:+.2f}): Top-20低夏普主因是 breadth不足 → 扩持仓可救 ✅")
        else:
            print(f"  → 头部 roll over(top1% {hm['top1%']:+.2f} < top10% {hm['top10%']:+.2f}): Top-20选到信号最不可靠区 → 扩breadth救不了 ⚠️")


if __name__ == "__main__":
    main()
