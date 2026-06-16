"""仪表盘核心逻辑：策略注册表、回测封装、Plotly 图表。纯函数，可单测。"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from quantlab.backtest import Backtester, BacktestResult
from quantlab.costs import AShareCostModel
from quantlab.data import load_prices
from quantlab.stats import metrics
from quantlab.strategies import (
    MACrossStrategy,
    DonchianBreakoutStrategy,
    BollingerReversionStrategy,
    BuyHoldStrategy,
)

# 策略注册表：名称 → (构造函数, 默认参数)。新增策略只需在此登记。
STRATEGY_REGISTRY = {
    "均线趋势 MA": (MACrossStrategy, {"fast": 20, "slow": 60}),
    "通道突破 Donchian": (DonchianBreakoutStrategy, {"entry": 20, "exit": 10}),
    "均值回归 Bollinger": (BollingerReversionStrategy, {"window": 20, "k": 2.0}),
    "买入持有 BuyHold": (BuyHoldStrategy, {}),
}


def build_strategy(name: str, params: dict):
    """按注册表构造策略实例。"""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {name}")
    ctor, defaults = STRATEGY_REGISTRY[name]
    merged = {**defaults, **(params or {})}
    return ctor(**merged)


def run(symbol: str, start: str, end: str, strategy_name: str,
        params: dict | None = None, *, initial_cash: float = 1_000_000,
        source: str = "auto") -> dict:
    """跑一次回测，返回仪表盘所需的全部数据（含买入持有基准）。"""
    prices = load_prices(symbol, start, end, source=source)
    bt = Backtester(initial_cash=initial_cash, cost_model=AShareCostModel())
    res = bt.run(prices, build_strategy(strategy_name, params))
    bench = bt.run(prices, BuyHoldStrategy())
    return {
        "prices": prices,
        "result": res,
        "benchmark": bench,
        "is_real": bool(prices.attrs.get("is_real")),
        "source": prices.attrs.get("source"),
        "stats": res.stats(),
        "bench_stats": bench.stats(),
    }


def equity_figure(result: BacktestResult, benchmark: BacktestResult | None = None) -> go.Figure:
    """净值曲线（策略 vs 买入持有），归一化到 1。"""
    fig = go.Figure()
    eq = result.equity / result.equity.iloc[0]
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=result.strategy_name,
                             line=dict(width=2)))
    if benchmark is not None:
        be = benchmark.equity / benchmark.equity.iloc[0]
        fig.add_trace(go.Scatter(x=be.index, y=be.values, name="买入持有",
                                 line=dict(width=1, dash="dash", color="gray")))
    fig.update_layout(title="净值曲线（归一）", height=300,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


def drawdown_figure(result: BacktestResult) -> go.Figure:
    """回撤曲线。"""
    eq = (1 + result.returns).cumprod()
    dd = eq / eq.cummax() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                             line=dict(color="crimson", width=1), name="回撤"))
    fig.update_layout(title="回撤", height=220, yaxis_tickformat=".0%",
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def position_figure(result: BacktestResult) -> go.Figure:
    """仓位曲线。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.positions.index, y=result.positions.values,
                             fill="tozeroy", line=dict(color="steelblue", width=1),
                             name="仓位"))
    fig.update_layout(title="仓位", height=200, yaxis_tickformat=".0%",
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def price_trades_figure(prices: pd.DataFrame, result: BacktestResult) -> go.Figure:
    """收盘价 + 买卖点。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices["close"], name="收盘",
                             line=dict(color="black", width=1)))
    if not result.trades.empty:
        buys = result.trades[result.trades["side"] == "buy"]
        sells = result.trades[result.trades["side"] == "sell"]
        fig.add_trace(go.Scatter(x=buys.index, y=buys["price"], mode="markers",
                                 name="买", marker=dict(color="red", symbol="triangle-up", size=8)))
        fig.add_trace(go.Scatter(x=sells.index, y=sells["price"], mode="markers",
                                 name="卖", marker=dict(color="green", symbol="triangle-down", size=8)))
    fig.update_layout(title="价格与成交点", height=300,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig
