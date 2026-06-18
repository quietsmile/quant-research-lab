"""事件驱动策略回测：好财报 + 价格突破 → 买入，持有到下次财报前卖出。

用户思路操作化：
- 基本面可以：最新已公告财报 ROE>0
- 财报预期明显超过当前：单季归母净利同比 > 阈值(默认 20%)（缺分析师预期，用高增长代理）
- 触发：3 日累计涨幅 ≥ 8%
- 卖出：下一个公告日前一交易日
- 超预期则继续：每个"好财报周期"各一笔，连续达标=滚动持有

变体对比（看每个条件是否加分）：
  A 完整(基本面+突破)  B 仅突破(不看基本面)  C 仅基本面(好财报周期开头买,不等突破)
全程 PIT；入场用突破"次日"收盘(防同bar前视)；扣双边成本；excess=相对等权市场。

跑法：python examples/event_strategy.py [同比阈值] [突破阈值]
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys

import numpy as np
import pandas as pd

from quantlab.data.tushare_adapter import load_daily_prices, TS_FEATURES_FILE
from quantlab import eval as ev

COST = 0.003   # 双边成本


def run(value_ratio=1.5, breakout=0.08, fair_pe=15.0):
    px = load_daily_prices().pivot_table(index="trade_date", columns="symbol",
                                         values="adj_close").sort_index()
    dates = px.index
    pos = {d: i for i, d in enumerate(dates)}
    cum3 = px / px.shift(3) - 1.0                       # 3 日累计收益
    mktidx = (1 + px.pct_change(fill_method=None).mean(axis=1)).cumprod()   # 等权市场指数

    # 季度末市值(万元)→ 每股按公告日 asof 取最近一期
    from quantlab.data.tushare_adapter import load_market_panel
    mvq = load_market_panel().pivot_table(index="trade_date", columns="symbol", values="total_mv").sort_index()

    feat = pd.read_parquet(TS_FEATURES_FILE)
    feat["announce_date"] = pd.to_datetime(feat["announce_date"])
    feat = feat.dropna(subset=["announce_date"]).sort_values(["symbol", "announce_date"])

    trades = {"A_full": [], "B_breakout": [], "C_fundamental": [], "D_good_dip": []}
    for sym, g in feat.groupby("symbol"):
        if sym not in px.columns:
            continue
        s = px[sym]; c3 = cum3[sym]
        mv_s = mvq[sym].dropna() if sym in mvq.columns else pd.Series(dtype=float)
        recs = g[["announce_date", "roe", "net_profit_q_yoy", "net_profit_ttm"]].values
        for k in range(len(recs) - 1):
            a0, roe, yoy, ttm = recs[k]
            a1 = recs[k + 1][0]
            # 隐含价值 = TTM归母净利 × 合理PE × (1+增长) ；与当前市值(万元→元)比
            mv_at = mv_s.loc[:a0]
            mcap = mv_at.iloc[-1] * 1e4 if len(mv_at) else np.nan
            # "对未来的预期"：用增长把盈利向前外推 2 年(g 截断在[0,1])，再×合理PE
            g = min(max(yoy, 0.0), 1.0) if (yoy is not None and not np.isnan(yoy)) else 0.0
            implied = ttm * fair_pe * (1 + g) ** 2 if (ttm is not None and not np.isnan(ttm)) else np.nan
            good = ((roe is not None and roe > 0) and (ttm is not None and ttm > 0)
                    and not np.isnan(mcap) and mcap > 0 and (implied / mcap) > value_ratio)
            # 周期内交易日 (a0, a1)
            win = dates[(dates > a0) & (dates < a1)]
            if len(win) < 4:
                continue
            exit_d = win[-1]                              # 下次财报前最后一交易日
            # 突破日：周期内首个 3 日累计≥阈值
            bdays = win[c3.reindex(win).values >= breakout]
            entry_b = None
            if len(bdays):
                bp = pos[bdays[0]] + 1                    # 次日收盘入场(防前视)
                if bp < pos[exit_d]:
                    entry_b = dates[bp]
            # 对照:好财报 + 3日回调≥阈值(抄跌)
            ddays = win[c3.reindex(win).values <= -breakout]
            entry_d = None
            if len(ddays):
                dp = pos[ddays[0]] + 1
                if dp < pos[exit_d]:
                    entry_d = dates[dp]

            def rec(entry):
                if entry is None or pd.isna(s.get(entry)) or pd.isna(s.get(exit_d)):
                    return None
                r = s[exit_d] / s[entry] - 1 - COST
                m = mktidx[exit_d] / mktidx[entry] - 1
                return {"entry": entry, "exit": exit_d, "ret": r, "excess": r - m,
                        "hold": pos[exit_d] - pos[entry], "year": entry.year}

            if good and entry_b is not None:
                t = rec(entry_b);  trades["A_full"].append(t) if t else None
            if entry_b is not None:
                t = rec(entry_b);  trades["B_breakout"].append(t) if t else None
            if good:
                t = rec(win[0]);   trades["C_fundamental"].append(t) if t else None
            if good and entry_d is not None:
                t = rec(entry_d);  trades["D_good_dip"].append(t) if t else None
    return {k: pd.DataFrame(v) for k, v in trades.items()}


def summarize(df, label):
    if df.empty:
        print(f"  {label}: 无交易"); return
    ex = df["excess"]
    t = ex.mean() / ex.std() * np.sqrt(len(ex))
    print(f"  {label:<16} 笔数 {len(df):>5} | 平均超额 {ex.mean():>+6.2%} | 胜率 {(ex>0).mean():>4.0%} | "
          f"t {t:>5.1f} | 持有中位 {int(df['hold'].median())}日 | 原始均收益 {df['ret'].mean():+.2%}")


def main():
    vr = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    bk = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
    pe = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    print(f"事件策略：隐含价值(TTM净利×{pe:.0f}×(1+增长)) > 当前市值×{vr} 且 ROE>0  +  3日涨≥{bk:.0%}"
          f" → 持有到下次财报前\n")
    tr = run(vr, bk, pe)

    print("===== 全样本（每笔=一个财报周期的一次交易，超额=相对等权市场）=====")
    for k, lab in [("A_full", "A 完整(好财报+追涨)"), ("B_breakout", "B 仅追涨"),
                   ("C_fundamental", "C 仅好财报"), ("D_good_dip", "D 好财报+抄跌")]:
        summarize(tr[k], lab)

    print("\n===== 样本外检验（完整策略 A：按入场年份切 dev≤2023 / Test≥2024）=====")
    A = tr["A_full"]
    for lab, sub in [("开发(≤2023)", A[A.year <= 2023]), ("冻结Test(≥2024)", A[A.year >= 2024])]:
        if len(sub):
            ex = sub["excess"]; t = ex.mean()/ex.std()*np.sqrt(len(ex))
            print(f"  {lab:<14} 笔数 {len(sub):>5} | 平均超额 {ex.mean():+.2%} | 胜率 {(ex>0).mean():.0%} | t {t:.1f}")
    # bootstrap 超额均值 CI（按笔自助）
    test = A[A.year >= 2024]["excess"].to_numpy()
    if len(test) > 30:
        rng = np.random.default_rng(0)
        means = [rng.choice(test, len(test), replace=True).mean() for _ in range(3000)]
        print(f"  Test 平均超额 95%CI: [{np.percentile(means,2.5):+.2%}, {np.percentile(means,97.5):+.2%}]")

    print("\n判读：比 A vs B 看'基本面过滤'是否加分；A vs C 看'突破择时'是否加分；"
          "Test 的 t 值与 CI 是否离 0 决定 OOS 是否可信。")


if __name__ == "__main__":
    main()
