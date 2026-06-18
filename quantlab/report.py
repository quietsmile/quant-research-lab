"""完整策略评估面板（不止看收益）。

覆盖六大类：① 收益(CAGR/累计/超额)；② 风险(最大回撤/回撤时长/最差期/CVaR)；
③ 风险调整(Sharpe/Sortino/Calmar/信息比)；④ 稳定性(分年/滚动Sharpe/正收益年占比/
收益集中度)；⑤ 尾部(偏度/峰度/VaR/CVaR/最长连亏)；⑥ 可信度(t值/PSR/DSR/bootstrap CI)。

核心理念：样本外、扣费后能否持续盈利，且回撤可承受——单一高 Sharpe 不可信。
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from quantlab import eval as ev


def _cagr(r: pd.Series, periods: int) -> float:
    n = len(r)
    if n == 0:
        return 0.0
    g = float((1 + r).prod())
    return g ** (periods / n) - 1 if g > 0 else -1.0


def _mdd_duration(equity: pd.Series):
    peak = equity.cummax()
    dd = equity / peak - 1
    mdd = float(dd.min())
    # 最长水下时长（连续未创新高的期数）
    under = dd < 0
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return mdd, longest


def _max_consec_neg(r: pd.Series) -> int:
    longest = cur = 0
    for x in r:
        cur = cur + 1 if x < 0 else 0
        longest = max(longest, cur)
    return longest


def performance_report(returns, periods: int = 12, benchmark=None, rf: float = 0.0,
                       n_trials: int = 1, trials_sr_std: float = 0.0) -> dict:
    """周期收益序列 → 全套指标 dict。benchmark 同频时给出超额/信息比。"""
    r = pd.Series(returns).dropna()
    if len(r) < 3:
        return {"n": len(r)}
    eq = (1 + r).cumprod()
    mdd, dd_dur = _mdd_duration(eq)
    downside = r[r < 0].std(ddof=1)
    var5 = float(r.quantile(0.05))
    m = {
        "n": len(r), "periods_per_year": periods,
        # ① 收益
        "cum_return": float(eq.iloc[-1] - 1), "cagr": _cagr(r, periods),
        "ann_vol": float(r.std(ddof=1) * sqrt(periods)),
        # ② 风险
        "max_drawdown": mdd, "max_dd_periods": dd_dur,
        "worst_period": float(r.min()), "var5": var5,
        "cvar5": float(r[r <= var5].mean()) if (r <= var5).any() else var5,
        "max_consec_loss": _max_consec_neg(r),
        # ③ 风险调整
        "sharpe": ev.sharpe(r, periods),
        "sortino": float(r.mean() / downside * sqrt(periods)) if downside and downside > 0 else float("nan"),
        "calmar": (_cagr(r, periods) / abs(mdd)) if mdd < 0 else float("nan"),
        # ⑤ 尾部
        "skew": float(r.skew()), "kurtosis": float(r.kurt()),
        # ⑥ 可信度
        "t_stat": float(r.mean() / r.std(ddof=1) * sqrt(len(r))) if r.std(ddof=1) else 0.0,
        "psr_vs0": ev.probabilistic_sharpe_ratio(r, 0.0, periods),
    }
    # ④ 稳定性（需 DatetimeIndex 才能分年/滚动）
    if isinstance(r.index, pd.DatetimeIndex):
        yearly = (1 + r).groupby(r.index.year).prod() - 1
        m["pct_positive_years"] = float((yearly > 0).mean())
        m["worst_year"] = float(yearly.min())
        m["best_year"] = float(yearly.max())
        m["yearly"] = yearly
        # 去掉最好的一年后是否仍正
        if len(yearly) > 1:
            m["cum_ex_best_year"] = float((1 + r[r.index.year != yearly.idxmax()]).prod() - 1)
        if len(r) >= periods + 2:
            rs = r.rolling(periods).apply(lambda x: ev.sharpe(x, periods), raw=False).dropna()
            if len(rs):
                m["rolling_sharpe_min"] = float(rs.min())
                m["rolling_sharpe_median"] = float(rs.median())
    # 收益集中度：最好的 k 期贡献占累计对数收益比例
    lr = np.log1p(r)
    if lr.sum() != 0:
        k = max(1, int(len(r) * 0.1))
        m["top10pct_periods_contrib"] = float(lr.nlargest(k).sum() / lr.sum())
    # 基准相关
    if benchmark is not None:
        b = pd.Series(benchmark).reindex(r.index).dropna()
        a = (r - b).dropna()
        if len(a) > 2 and a.std(ddof=1):
            m["excess_cagr"] = _cagr(r.loc[a.index], periods) - _cagr(b.loc[a.index], periods)
            m["info_ratio"] = float(a.mean() / a.std(ddof=1) * sqrt(periods))
    # 多重检验
    if n_trials > 1 and trials_sr_std > 0:
        m["dsr"] = ev.deflated_sharpe_ratio(r, n_trials, trials_sr_std, periods)["dsr"]
    m.update({f"boot_{k}": v for k, v in ev.block_bootstrap_ci(r, periods=periods).items()})
    return m


def trade_stats(trade_returns) -> dict:
    """逐笔收益 → 胜率/期望/盈亏比/Profit Factor/集中度（胜率只是辅助）。"""
    t = pd.Series(trade_returns).dropna()
    if len(t) == 0:
        return {}
    win, loss = t[t > 0], t[t < 0]
    gross_win, gross_loss = win.sum(), -loss.sum()
    return {
        "n_trades": len(t), "win_rate": float((t > 0).mean()),
        "avg_win": float(win.mean()) if len(win) else 0.0,
        "avg_loss": float(loss.mean()) if len(loss) else 0.0,
        "payoff": float(win.mean() / -loss.mean()) if len(loss) and loss.mean() != 0 else float("nan"),
        "expectancy": float(t.mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "top10_trade_contrib": float(t.nlargest(max(1, int(len(t) * 0.01))).sum() / t.sum()) if t.sum() != 0 else float("nan"),
    }


def format_report(m: dict, title: str = "策略评估面板") -> str:
    if m.get("n", 0) < 3:
        return f"{title}: 样本不足"
    L = [f"===== {title}（{m['n']} 期，年化基准 {m['periods_per_year']}）====="]
    L.append("① 收益   : CAGR {cagr:+.1%} | 累计 {cum_return:+.1%}".format(**m)
             + (f" | 超额CAGR {m['excess_cagr']:+.1%}" if "excess_cagr" in m else ""))
    L.append("② 风险   : 最大回撤 {max_drawdown:.1%} | 回撤时长 {max_dd_periods}期 | 最差单期 {worst_period:.1%} | CVaR5% {cvar5:.1%} | 最长连亏 {max_consec_loss}期".format(**m))
    L.append("③ 风险调整: Sharpe {sharpe:.2f} | Sortino {sortino:.2f} | Calmar {calmar:.2f}".format(**m)
             + (f" | 信息比 {m['info_ratio']:.2f}" if "info_ratio" in m else "") + f" | 年化波动 {m['ann_vol']:.1%}")
    if "pct_positive_years" in m:
        L.append("④ 稳定性 : 正收益年占比 {pct_positive_years:.0%} | 最差年 {worst_year:+.1%} | 最好年 {best_year:+.1%}".format(**m)
                 + (f" | 去最好年后累计 {m['cum_ex_best_year']:+.1%}" if "cum_ex_best_year" in m else "")
                 + (f" | 滚动Sharpe中位 {m['rolling_sharpe_median']:.2f}(最低 {m['rolling_sharpe_min']:.2f})" if "rolling_sharpe_median" in m else ""))
    if "top10pct_periods_contrib" in m:
        L.append("           最好10%期贡献 {top10pct_periods_contrib:.0%} 的累计收益（越高越依赖少数时段）".format(**m))
    L.append("⑤ 尾部   : 偏度 {skew:+.2f} | 峰度 {kurtosis:+.2f} | VaR5% {var5:.1%}".format(**m))
    rel = "⑥ 可信度 : t值 {t_stat:.1f} | PSR {psr_vs0:.0%}".format(**m)
    if "dsr" in m:
        rel += f" | Deflated SR {m['dsr']:.0%}"
    if "boot_sharpe_ci" in m:
        rel += f" | bootSharpe95%CI [{m['boot_sharpe_ci'][0]:.2f},{m['boot_sharpe_ci'][1]:.2f}]"
    L.append(rel)
    return "\n".join(L)
