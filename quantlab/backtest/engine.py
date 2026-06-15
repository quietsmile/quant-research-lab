"""日频回测引擎。

设计要点（每一条都是为了"不骗自己"）：

1. **信号错位执行**：t 日收盘生成的信号，在 t+1 日**开盘**成交。
   引擎对信号做 shift(1)，从根本上杜绝前视偏差。
2. **真实成本**：每笔成交都过 :class:`AShareCostModel`，佣金/印花税/
   过户费/滑点全扣。
3. **T+1**：每日只在开盘调一次仓，且当日买入的份额次日才可卖出 ——
   "每日一次开盘再平衡"本身就满足 T+1（详见 README）。
4. **涨跌停**：开盘价触及涨停则买单不成交、触及跌停则卖单不成交，
   仓位维持不变。
5. **整手**：按 100 股一手取整（A 股规则）。

输出 :class:`BacktestResult`，自带诚实的体检 :meth:`summary`。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.costs import AShareCostModel
from quantlab.stats import metrics
from quantlab.strategies.base import Strategy

LOT_SIZE = 100


@dataclass
class BacktestResult:
    """回测结果容器。"""

    equity: pd.Series              # 每日收盘净值（绝对金额）
    returns: pd.Series             # 每日净值收益率
    positions: pd.Series           # 每日目标仓位（执行后实际权重）
    trades: pd.DataFrame           # 成交明细
    total_cost: float              # 累计交易成本（元）
    initial_cash: float
    cost_model: AShareCostModel
    strategy_name: str = "strategy"
    n_rejected: int = 0            # 因涨跌停被拒的下单次数
    meta: dict = field(default_factory=dict)

    @property
    def turnover(self) -> float:
        """年化双边换手率估计（成交额 / 平均净值，再年化）。"""
        if self.trades.empty or len(self.equity) == 0:
            return 0.0
        traded = self.trades["amount"].abs().sum()
        avg_equity = self.equity.mean()
        years = len(self.equity) / metrics.TRADING_DAYS
        if avg_equity <= 0 or years <= 0:
            return 0.0
        return float(traded / avg_equity / years)

    def stats(self, risk_free: float = 0.0) -> dict:
        s = metrics.summary_stats(self.returns, risk_free=risk_free)
        s["total_cost"] = self.total_cost
        s["cost_drag_pct"] = (
            self.total_cost / self.initial_cash if self.initial_cash else 0.0
        )
        s["turnover_annual"] = self.turnover
        s["n_trades"] = int(len(self.trades))
        s["n_rejected_by_limit"] = self.n_rejected
        return s

    def summary(self, risk_free: float = 0.0) -> str:
        """打印一份带"自我怀疑"的体检报告。"""
        s = self.stats(risk_free)
        lines = [
            f"===== 回测体检：{self.strategy_name} =====",
            f"  期间             : {self.equity.index[0].date()} ~ {self.equity.index[-1].date()}  ({s['n_periods']} 交易日)",
            f"  累计收益         : {s['cumulative_return']:+.2%}",
            f"  年化收益         : {s['annualized_return']:+.2%}",
            f"  年化波动         : {s['annualized_volatility']:.2%}",
            f"  夏普 (rf={risk_free:.1%})   : {s['sharpe']:.2f}",
            f"  最大回撤         : {s['max_drawdown']:.2%}",
            f"  Calmar           : {s['calmar']:.2f}",
            "  ---- 成本与摩擦 ----",
            f"  累计交易成本     : {s['total_cost']:,.0f} 元  (占初始资金 {s['cost_drag_pct']:.2%})",
            f"  年化换手率       : {s['turnover_annual']:.1f}x",
            f"  成交笔数         : {s['n_trades']}",
            f"  涨跌停被拒下单   : {s['n_rejected_by_limit']} 次",
            "  ---- ⚠ 先怀疑再相信 ----",
            "  · 这是单标的、样本内结果，夏普极易高估，请看 walk-forward 的 OOS 指标。",
            "  · 若年化换手率高，成本拖累会随实盘费率上升而放大。",
            "  · 参数若来自网格搜索，存在多重检验/过拟合风险。",
        ]
        return "\n".join(lines)


class Backtester:
    """日频、开盘成交、强制扣成本的回测引擎。"""

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        cost_model: AShareCostModel | None = None,
        lot_size: int = LOT_SIZE,
    ):
        if initial_cash <= 0:
            raise ValueError("initial_cash 必须为正")
        self.initial_cash = initial_cash
        self.cost_model = cost_model or AShareCostModel()
        self.lot_size = lot_size

    def run(self, prices: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        """运行回测。

        Parameters
        ----------
        prices : DataFrame
            OHLCV，DatetimeIndex，建议先经 clean_prices。
        strategy : Strategy
            产出目标仓位 [0,1] 的策略。
        """
        for col in ("open", "close"):
            if col not in prices.columns:
                raise ValueError(f"行情缺少必要列: {col}")
        if len(prices) < 2:
            raise ValueError("行情过短，至少需要 2 个交易日")

        # 目标仓位 -> 错位一日执行（杜绝前视）
        target = strategy.generate_signals(prices).shift(1).fillna(0.0)

        opens = prices["open"].to_numpy()
        closes = prices["close"].to_numpy()
        prev_closes = prices["close"].shift(1).to_numpy()
        index = prices.index

        cash = self.initial_cash
        shares = 0
        total_cost = 0.0
        n_rejected = 0

        equity = np.empty(len(prices))
        realized_pos = np.empty(len(prices))
        trade_records: list[dict] = []

        for i in range(len(prices)):
            exec_price = opens[i]
            tgt = float(target.iloc[i])

            # 仅在有合法成交价时尝试调仓
            if exec_price > 0 and not np.isnan(exec_price):
                portfolio_value = cash + shares * exec_price
                desired_value = tgt * portfolio_value
                desired_shares = int(desired_value // (exec_price * self.lot_size)) * self.lot_size
                delta = desired_shares - shares

                # 买入时为交易成本预留现金，避免现金变负后次日产生虚假再平衡。
                # 可买上限 ≈ (现金 - 最低佣金) / (价格 × (1 + 买入费率))。
                if delta > 0:
                    buy_rate = (
                        self.cost_model.commission_rate
                        + self.cost_model.transfer_fee_rate
                        + self.cost_model.slippage_rate
                    )
                    budget = max(cash - self.cost_model.min_commission, 0.0)
                    affordable = int(
                        budget / (exec_price * (1 + buy_rate)) // self.lot_size
                    ) * self.lot_size
                    delta = min(delta, affordable)

                if delta != 0:
                    prev_close = prev_closes[i]
                    blocked = False
                    if delta > 0 and self.cost_model.hit_upper_limit(prev_close, exec_price):
                        blocked = True  # 涨停买不进
                    elif delta < 0 and self.cost_model.hit_lower_limit(prev_close, exec_price):
                        blocked = True  # 跌停卖不出

                    if blocked:
                        n_rejected += 1
                    else:
                        side = "buy" if delta > 0 else "sell"
                        amount = abs(delta) * exec_price
                        tc = self.cost_model.cost(amount=amount, side=side)
                        # 现金变动：买入支出 = 金额 + 成本；卖出收入 = 金额 - 成本
                        if delta > 0:
                            cash -= amount + tc.total
                        else:
                            cash += amount - tc.total
                        shares += delta
                        total_cost += tc.total
                        trade_records.append(
                            {
                                "date": index[i],
                                "side": side,
                                "price": exec_price,
                                "shares": delta,
                                "amount": amount,
                                "cost": tc.total,
                                "commission": tc.commission,
                                "stamp_tax": tc.stamp_tax,
                                "transfer_fee": tc.transfer_fee,
                                "slippage": tc.slippage,
                            }
                        )

            # 当日收盘估值
            eq = cash + shares * closes[i]
            equity[i] = eq
            realized_pos[i] = (shares * closes[i] / eq) if eq > 0 else 0.0

        equity_s = pd.Series(equity, index=index, name="equity")
        returns_s = equity_s.pct_change().fillna(0.0)
        positions_s = pd.Series(realized_pos, index=index, name="position")
        trades_df = (
            pd.DataFrame(trade_records).set_index("date")
            if trade_records
            else pd.DataFrame(
                columns=["side", "price", "shares", "amount", "cost",
                         "commission", "stamp_tax", "transfer_fee", "slippage"]
            )
        )

        return BacktestResult(
            equity=equity_s,
            returns=returns_s,
            positions=positions_s,
            trades=trades_df,
            total_cost=total_cost,
            initial_cash=self.initial_cash,
            cost_model=self.cost_model,
            strategy_name=strategy.name,
            n_rejected=n_rejected,
        )
