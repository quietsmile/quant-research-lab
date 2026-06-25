"""对 S3/S6/S2 做正确的样本外确认：dev 选参数 → 冻结 Test 只评一次 + Deflated Sharpe。

要点(避免之前'全样本选参再看全样本'的偏差)：
- 切 dev(2020~2024-06) / 冻结 Test(2024-07~2026)；
- 每个策略**只在 dev 上网格搜参**(按 dev 夏普)，定死后**在 Test 跑一次**；
- DSR 的 n_trials = 三策略网格里试过的总配置数(多重检验校正)，trials_sr_std 用各配置 dev 的每期夏普离散度；
- 报 Test 完整面板 + PSR/DSR/bootstrap CI。
跑法：python examples/confirm_strategies.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# 复用策略族引擎与数据
from examples.strategy_family import (run, grid, STRATS, size_neutral, comp, idx, ANN, MKT)
from quantlab import report as rp, eval as ev

TEST_START = pd.Timestamp("2024-07-01")
PICK = ["S2 质量+价值EP", "S3 质量+低波", "S6 多因子融合"]
dev_idx = idx[idx < TEST_START]; test_idx = idx[idx >= TEST_START]


def daily_sr(p):
    p = p.dropna()
    return p.mean() / (p.std() + 1e-12)


def main():
    print(f"dev {dev_idx[0].date()}~{dev_idx[-1].date()}({len(dev_idx)}d) | "
          f"冻结Test {test_idx[0].date()}~{test_idx[-1].date()}({len(test_idx)}d)\n")
    all_dev_sr = []          # 所有尝试配置的 dev 每期夏普(算 DSR 的 trials_std/n_trials)
    chosen = {}
    for name in PICK:
        sel_fn, score_fn, space = STRATS[name]
        best, best_dev = None, -9
        for P in [c for c in grid(space) if c["topn"] >= 15]:
            sc = size_neutral(comp) if name.startswith("S5") else score_fn(P)
            p, nh = run(sel_fn(P), sc, P["topn"], P.get("rebal", 5))
            if nh.mean() < 15:
                continue
            dsr_dev = daily_sr(p.reindex(dev_idx))     # 只用 dev 选参
            all_dev_sr.append(dsr_dev)
            if dsr_dev > best_dev:
                best_dev, best = dsr_dev, (P, p)
        chosen[name] = best

    n_trials = len(all_dev_sr)
    trials_std = float(np.std(all_dev_sr, ddof=1))
    print(f"多重检验：三策略共试 {n_trials} 个配置；dev 每期夏普离散度 {trials_std:.4f}\n")

    print(f"{'策略(dev选定参数)':30}{'devSharpe':>10}{'TestSharpe':>11}{'TestCAGR':>9}{'Test回撤':>8}{'PSR':>6}{'DSR':>6}")
    for name in PICK:
        P, p = chosen[name]
        dp = p.reindex(dev_idx).fillna(0); tp = p.reindex(test_idx).fillna(0)
        m = rp.performance_report(tp, periods=ANN, n_trials=n_trials, trials_sr_std=trials_std)
        dev_sh = ev.sharpe(dp, ANN); test_sh = m["sharpe"]
        print(f"{name:30}{dev_sh:>10.2f}{test_sh:>11.2f}{m['cagr']*100:>+8.0f}%{m['max_drawdown']*100:>+7.0f}%"
              f"{m['psr_vs0']:>6.0%}{m.get('dsr',float('nan')):>6.0%}  P={P}")
        ci = m.get("boot_sharpe_ci")
        if ci:
            print(f"{'  └ Test bootstrap 夏普95%CI':30} [{ci[0]:.2f}, {ci[1]:.2f}] | Sortino {m['sortino']:.2f} | Calmar {m['calmar']:.2f}")
    # 基准 Test
    bt = MKT.reindex(test_idx).fillna(0)
    print(f"\n等权基准 Test: 夏普 {ev.sharpe(bt,ANN):.2f} | CAGR {((1+bt).prod()**(ANN/len(bt))-1)*100:+.0f}%")
    print("\n判读：Test 夏普是否仍稳、DSR 是否>~90%(校正多重检验后)、bootstrap CI 是否离 0 —— 三者一致才算确认。")


if __name__ == "__main__":
    main()
