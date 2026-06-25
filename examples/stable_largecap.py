"""大盘股稳健策略族：限定大中盘(结构上压低小盘暴露)、防御因子、宽分散、看绝对收益。

针对'要稳定、对大多数股票成立、别压小盘、关注总收益'：
- 股票池：每个调仓日按市值取 **Top mv_top**(剔掉小/微盘) & 趋势 & 财报质量；
- 因子：价值(EP)/低波/质量(ROE)/动量 的组合，打分**再做规模中性**(双重压 SMB)；
- 宽分散(topn≥30)、月度调仓、含成本；
- 评估：**绝对** CAGR/夏普/回撤/Calmar + 子区间 + 冻结Test + DSR + Barra(看 SMB) + 对比**沪深300**(非等权小盘)。
跑法：python examples/stable_largecap.py
"""
import warnings; warnings.filterwarnings("ignore")
import itertools, json, pathlib
import numpy as np, pandas as pd
from examples.strategy_family import (run, metr, idx, ANN, ret, mv, ep, roe, pft, close, ma20,
                                      trend, q, vol, mom, size_neutral, rk, MKT)
from quantlab import barra, eval as ev, report as rp

DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data")
HS300 = pd.read_parquet(DD / "pullback_bench.parquet"); HS300.index = pd.to_datetime(HS300.index)
HS300 = HS300["沪深300"].reindex(idx)
TEST_START = pd.Timestamp("2024-07-01")
dev_i, test_i = idx[idx < TEST_START], idx[idx >= TEST_START]

# 大盘掩码：市值排名前 mv_top 比例（剔小盘）
def largecap(mv_top):
    return mv.rank(axis=1, ascending=False, pct=True) <= mv_top

# 风格因子(算 Barra)
style = barra.build_style_factors(ret.shift(-1).clip(-0.21, 0.21), market=MKT,
                                  logmv=np.log(mv.clip(lower=1)), ep=ep, mom=mom(60), vol=vol(20), growth=pft)

STRATS = {
 "L1 大盘价值EP":   (lambda P: ep, dict(mv_top=[0.3, 0.4, 0.5], topn=[30, 40, 50], rebal=[10, 20])),
 "L2 大盘低波":     (lambda P: -vol(20), dict(mv_top=[0.3, 0.4, 0.5], topn=[30, 40, 50], rebal=[10, 20])),
 "L3 大盘质量ROE":  (lambda P: roe, dict(mv_top=[0.3, 0.4, 0.5], topn=[30, 40, 50], rebal=[10, 20])),
 "L4 大盘多因子(价值+低波+质量+动量)": (lambda P: rk(ep) + rk(-vol(20)) + rk(roe) + rk(mom(40)),
                   dict(mv_top=[0.3, 0.5], topn=[30, 40, 50], rebal=[10, 20])),
 "L5 大盘价值+低波": (lambda P: rk(ep) + rk(-vol(20)), dict(mv_top=[0.3, 0.5], topn=[30, 40, 50], rebal=[10, 20])),
}


def grid(s): k = list(s); return [dict(zip(k, v)) for v in itertools.product(*[s[k] for k in k])]


def absmetr(p):
    p = p.fillna(0); nav = (1 + p).cumprod()
    cagr = nav.iloc[-1] ** (ANN / len(p)) - 1
    return cagr, p.mean() / (p.std() + 1e-12) * np.sqrt(ANN), (nav / nav.cummax() - 1).min()


def main():
    print(f"大盘稳健策略族 | dev {dev_i[0].date()}~{dev_i[-1].date()} / 冻结Test {test_i[0].date()}~{test_i[-1].date()}")
    print("(股票池=大中盘&趋势&质量, 打分规模中性, 宽分散, 月度优先; 绝对收益 + 对比沪深300)\n")
    all_dev_sr, results = [], {}
    for name, (score_fn, space) in STRATS.items():
        best, best_dev = None, -9
        for P in [c for c in grid(space) if c["topn"] >= 30]:
            sel = trend & q & largecap(P["mv_top"])
            sc = size_neutral(score_fn(P))                      # 打分规模中性
            p, nh = run(sel, sc, P["topn"], P["rebal"])
            if nh.mean() < 25:
                continue
            ds = p.reindex(dev_i).fillna(0)
            dsr_dev = ds.mean() / (ds.std() + 1e-12)
            all_dev_sr.append(dsr_dev)
            if dsr_dev > best_dev:
                best_dev, best = dsr_dev, (P, p, nh)
        if best is None:
            continue
        P, p, nh = best
        cg, sh, dd = absmetr(p); m = metr(p, nh)
        tp = p.reindex(test_i).fillna(0); tcg, tsh, tdd = absmetr(tp)
        b = barra.barra_exposure(p, style)
        results[name] = dict(P=P, cagr=cg, sharpe=sh, maxdd=dd, calmar=cg / abs(dd),
                             sh1=m["sh1"], sh2=m["sh2"], nh=m["nh"], by=m["by"],
                             test_cagr=tcg, test_sharpe=tsh, test_maxdd=tdd,
                             smb=b["exposure"].get("SIZE(小)"), alpha=b["alpha_ann"], r2=b["r2"], barra=b["exposure"])
        print(f"{name} 最优{P}")
        print(f"  全期: 年化{cg*100:+.0f}% 夏普{sh:.2f} 回撤{dd*100:+.0f}% Calmar{cg/abs(dd):.2f} 子区间{m['sh1']:.2f}/{m['sh2']:.2f} 持股{m['nh']:.0f}")
        print(f"  冻结Test: 年化{tcg*100:+.0f}% 夏普{tsh:.2f} 回撤{tdd*100:+.0f}% | SMB暴露{b['exposure'].get('SIZE(小)'):+.2f} α{b['alpha_ann']*100:+.0f}%")
    # 沪深300 对比
    for lab, b in [("全期", HS300), ("Test", HS300.reindex(test_i))]:
        cg, sh, dd = absmetr(b.fillna(0)); print(f"沪深300 {lab}: 年化{cg*100:+.0f}% 夏普{sh:.2f} 回撤{dd*100:+.0f}%")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "barra"} | {"barra": v["barra"]}
               for k, v in results.items()}, open(DD / "stable_largecap.json", "w"), ensure_ascii=False, default=float)
    print("\n结果存 dashboard_data/stable_largecap.json")


if __name__ == "__main__":
    main()
