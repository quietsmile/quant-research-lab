"""绩效与统计指标 —— 单一可信实现。

约定：
- ``returns`` 为**简单日收益率**序列（不是价格、不是对数收益）。
- 年化交易日数默认 252。
- 无风险利率默认 0（A 股研究常用近似；需要时显式传入年化值）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_returns(prices: pd.Series) -> pd.Series:
    """价格序列 → 简单日收益率（首日为 NaN，已剔除）。"""
    return prices.pct_change().dropna()


def cumulative_return(returns: pd.Series) -> float:
    """累计收益率（区间总收益）。"""
    if len(returns) == 0:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """几何年化收益率。"""
    n = len(returns)
    if n == 0:
        return 0.0
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods / n) - 1.0


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """年化波动率（样本标准差，ddof=1）。"""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """年化夏普比率。

    risk_free 为**年化**无风险利率，内部按 periods 折算到每期。
    """
    if len(returns) < 2:
        return 0.0
    rf_per = risk_free / periods
    excess = returns - rf_per
    std = excess.std(ddof=1)
    # 常数收益序列的标准差在浮点下约为 1e-19 而非精确 0：用极小阈值
    # 判定"无波动"，避免夏普被放大成天文数字。
    if not np.isfinite(std) or std < 1e-12:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """最大回撤（负值，如 -0.35 表示 -35%）。"""
    if len(returns) == 0:
        return 0.0
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Calmar = 年化收益 / |最大回撤|。"""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return annualized_return(returns, periods) / abs(mdd)


def correlation(a: pd.Series, b: pd.Series) -> float:
    """两条收益序列的皮尔逊相关系数（按索引对齐取交集）。"""
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(df) < 2:
        return float("nan")
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def linear_regression(y: pd.Series, x: pd.Series) -> dict:
    """最简单的一元线性回归 y = alpha + beta * x（OLS）。

    返回 alpha / beta / r_squared / n。常用于算策略对基准的 alpha/beta。
    """
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    if len(df) < 2:
        raise ValueError("回归样本不足（<2）")
    yy = df.iloc[:, 0].to_numpy()
    xx = df.iloc[:, 1].to_numpy()
    beta, alpha = np.polyfit(xx, yy, 1)
    y_hat = alpha + beta * xx
    ss_res = float(np.sum((yy - y_hat) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"alpha": float(alpha), "beta": float(beta), "r_squared": r2, "n": len(df)}


def summary_stats(
    returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS
) -> dict:
    """一次性给出常用指标，便于回测结果体检。"""
    return {
        "cumulative_return": cumulative_return(returns),
        "annualized_return": annualized_return(returns, periods),
        "annualized_volatility": annualized_volatility(returns, periods),
        "sharpe": sharpe_ratio(returns, risk_free, periods),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods),
        "n_periods": len(returns),
    }
