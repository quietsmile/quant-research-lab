"""全市场多因子分层回测（价值+质量+成长），PIT 正确、含退市、无幸存者偏差。

因子（每个调仓截面现算 z-score 后等权合成）：
- 价值 value : 盈利收益率 1/pe_ttm、账面市值比 1/pb（来自市场面板）
- 质量 quality: ROE（来自 point_in_time 的 PIT 财务）
- 成长 growth : 归母净利单季同比 net_profit_q_yoy（PIT、去季节性）

流程：季度末用 point_in_time 取"截至当日已公告"的财务（防前视）+ 面板估值，
合成打分→分 5 层→看分层收益单调性、多空(Q1-Q5)、IC。价格用后复权，
股票池含退市股（截面取当日实际可交易者）。

跑法：python examples/multifactor_backtest.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data import point_in_time
from quantlab.data.tushare_adapter import load_market_panel
from quantlab.factors import winsorize, zscore
from quantlab.stats import metrics


def main() -> None:
    panel = load_market_panel().sort_values("trade_date")
    dates = sorted(panel["trade_date"].unique())
    aclose = panel.pivot_table(index="trade_date", columns="symbol", values="adj_close")
    print(f"[面板] {len(dates)} 个季度末 | {aclose.shape[1]} 只 | "
          f"{pd.Timestamp(dates[0]).date()}~{pd.Timestamp(dates[-1]).date()}\n")

    NQ = 5
    q_rets = {q: [] for q in range(1, NQ + 1)}
    ls_rets, ics, n_cs = [], [], []
    cost = 0.003   # 多空换手粗略成本

    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        cs = panel[panel["trade_date"] == t].set_index("symbol")
        fwd = aclose.loc[t1] / aclose.loc[t] - 1.0          # 下一季度收益
        snap = point_in_time(pd.Timestamp(t).strftime("%Y-%m-%d")).set_index("symbol")

        # 规模过滤：剔除最小 20%（去微盘噪声）
        mv_floor = cs["total_mv"].quantile(0.2)
        keep = cs.index[cs["total_mv"] >= mv_floor]

        df = pd.DataFrame(index=keep)
        df["ey"] = (1.0 / cs["pe_ttm"]).replace([np.inf, -np.inf], np.nan).where(cs["pe_ttm"] > 0)
        df["bm"] = (1.0 / cs["pb"]).replace([np.inf, -np.inf], np.nan).where(cs["pb"] > 0)
        df["roe"] = snap["roe"].reindex(keep)
        df["growth"] = snap["net_profit_q_yoy"].reindex(keep)
        df["fwd"] = fwd.reindex(keep)
        df = df.dropna(subset=["fwd"])
        if len(df) < 50:
            continue

        # 每因子截面 winsorize+zscore，等权合成（容忍部分缺失）
        zs = []
        for c in ["ey", "bm", "roe", "growth"]:
            z = zscore(winsorize(df[c]))
            zs.append(z)
        df["score"] = pd.concat(zs, axis=1).mean(axis=1, skipna=True)
        df = df.dropna(subset=["score"])
        if len(df) < 50:
            continue

        # IC（截面打分 vs 下季收益，spearman）
        ics.append(df["score"].corr(df["fwd"], method="spearman"))
        # 分 5 层：qcut 升序赋标签 → label 1=最低分, NQ=最高分（Q5=高分组）
        df["q"] = pd.qcut(df["score"].rank(method="first"), NQ, labels=range(1, NQ + 1)).astype(int)
        for q in range(1, NQ + 1):
            q_rets[q].append(df.loc[df["q"] == q, "fwd"].mean())
        # 多空 = 高分(Q5) - 低分(Q1)
        ls_rets.append(df.loc[df["q"] == NQ, "fwd"].mean() - df.loc[df["q"] == 1, "fwd"].mean() - cost)
        n_cs.append(len(df))

    # ---- 汇总 ----
    ann = lambda r: (1 + pd.Series(r).dropna()).prod() ** (4 / max(len(r), 1)) - 1
    print("===== 分层年化收益(Q5=高分组 → Q1=低分组)=====")
    qa = {q: ann(q_rets[q]) for q in range(1, NQ + 1)}
    for q in range(NQ, 0, -1):
        bar = "█" * max(0, int(qa[q] * 100))
        print(f"  Q{q} {'(高分)' if q==NQ else '(低分)' if q==1 else '     '}: {qa[q]:+.1%} {bar}")
    mono = all(qa[q] >= qa[q - 1] for q in range(2, NQ + 1))
    print(f"  单调性(高分组收益更高): {'✅ 单调' if mono else '⚠ 非严格单调'}")

    ls = pd.Series(ls_rets).dropna()
    ic = pd.Series(ics).dropna()
    print("\n===== 多空 & IC =====")
    print(f"  多空(Q5-Q1) 年化 {ann(ls_rets):+.1%} | 季度胜率 {(ls>0).mean():.0%} | "
          f"夏普 {metrics.sharpe_ratio(ls, periods=4):.2f}")
    print(f"  IC 均值 {ic.mean():+.3f} | IC_IR(均值/标准差) {ic.mean()/ic.std():.2f} | "
          f"|IC|>0.02 占比 {(ic.abs()>0.02).mean():.0%}")
    print(f"  调仓期数 {len(ls)} | 每期股票数中位 ~{int(np.median(n_cs)) if n_cs else 0}")
    print("\n注：PIT防前视+含退市股；季度调仓，多空已扣粗略成本。中途退市股下季无价→该期剔除"
          "(轻微低估退市损失，后续可用退市收益精修)。")


if __name__ == "__main__":
    main()
