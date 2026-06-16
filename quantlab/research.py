"""跨标的稳健性研究：把同一套规则跑遍标的池，看它是不是只在个别票上好看。

这是防过拟合的关键一步——单票上调出来的好看结果几乎没有意义；一个规则
要在**多数标的**上都站得住，才谈得上"可能有效"。
"""
from __future__ import annotations

import pandas as pd

from quantlab.backtest import Backtester
from quantlab.data import load_prices
from quantlab.stats import metrics
from quantlab.strategies import BuyHoldStrategy


def universe_backtest(
    symbols: list[str],
    strategy_factory,
    start: str,
    end: str,
    *,
    backtester: Backtester | None = None,
    source: str = "auto",
) -> dict:
    """对标的池逐只回测同一策略，汇总稳健性。

    Parameters
    ----------
    symbols : list[str]
        标的代码列表。
    strategy_factory : Callable[[], Strategy]
        无参工厂，每只标的构造一个新策略实例。
    start, end : str
    backtester : Backtester
        默认含 A 股成本的回测器。
    source : str
        数据源（建议先用 build_dataset 落好离线缓存，再用 "auto"/"offline"）。

    Returns
    -------
    dict: {"per_symbol": DataFrame, "portfolio_returns": Series, "summary": dict}
    """
    bt = backtester or Backtester()
    rows = []
    strat_ret_cols: dict[str, pd.Series] = {}

    for sym in symbols:
        try:
            prices = load_prices(sym, start, end, source=source)
        except Exception as e:  # noqa: BLE001
            rows.append({"symbol": sym, "error": type(e).__name__})
            continue

        strat = strategy_factory()
        res = bt.run(prices, strat)
        bh = bt.run(prices, BuyHoldStrategy())

        sharpe = metrics.sharpe_ratio(res.returns)
        bh_sharpe = metrics.sharpe_ratio(bh.returns)
        rows.append({
            "symbol": sym,
            "is_real": bool(prices.attrs.get("is_real")),
            "cum_return": metrics.cumulative_return(res.returns),
            "sharpe": sharpe,
            "max_dd": metrics.max_drawdown(res.returns),
            "turnover": res.turnover,
            "bh_sharpe": bh_sharpe,
            "beats_bh": sharpe > bh_sharpe,
        })
        strat_ret_cols[sym] = res.returns

    per_symbol = pd.DataFrame(rows)
    valid = per_symbol[per_symbol.get("sharpe").notna()] if "sharpe" in per_symbol else per_symbol

    # 等权组合：按日对齐取各标的策略收益的均值（简单稳健性代理）
    if strat_ret_cols:
        ret_df = pd.DataFrame(strat_ret_cols).sort_index()
        portfolio_returns = ret_df.mean(axis=1).dropna()
    else:
        portfolio_returns = pd.Series(dtype=float)

    summary = {
        "n_symbols": int(len(valid)),
        "median_sharpe": float(valid["sharpe"].median()) if len(valid) else float("nan"),
        "pct_positive_sharpe": float((valid["sharpe"] > 0).mean()) if len(valid) else float("nan"),
        "pct_beat_buyhold": float(valid["beats_bh"].mean()) if len(valid) else float("nan"),
        "portfolio_sharpe": metrics.sharpe_ratio(portfolio_returns) if len(portfolio_returns) else float("nan"),
        "portfolio_cum_return": metrics.cumulative_return(portfolio_returns) if len(portfolio_returns) else float("nan"),
        "portfolio_max_dd": metrics.max_drawdown(portfolio_returns) if len(portfolio_returns) else float("nan"),
    }
    return {"per_symbol": per_symbol, "portfolio_returns": portfolio_returns, "summary": summary}
