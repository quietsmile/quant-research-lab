"""样本外验证：train/test 切分、walk-forward 滚动验证、过拟合诊断。

为什么重要：一个策略在历史上好看（样本内）几乎没有意义；它必须在
**没见过的数据**（样本外 / OOS）上仍然站得住。这里提供三件工具：

- :func:`train_test_split` —— 时间序列切分（绝不打乱顺序）。
- :func:`walk_forward`     —— 滚动窗口反复"用过去测未来"。
- :func:`overfitting_report` —— 把样本内 vs 样本外的差距量化成一句人话。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantlab.backtest import Backtester
from quantlab.stats import metrics
from quantlab.strategies.base import Strategy


def train_test_split(
    prices: pd.DataFrame, train_ratio: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间顺序切分（前 train_ratio 为训练，其余为测试）。

    时间序列**绝不能随机打乱**，否则会用未来信息训练 = 前视偏差。
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须在 (0,1)")
    n = len(prices)
    cut = int(n * train_ratio)
    if cut < 1 or cut >= n:
        raise ValueError("数据太短，无法切分")
    return prices.iloc[:cut], prices.iloc[cut:]


@dataclass
class WalkForwardResult:
    """walk-forward 汇总结果。"""

    folds: pd.DataFrame      # 每个 fold 的 OOS 指标
    oos_returns: pd.Series   # 拼接后的样本外收益序列
    n_folds: int

    def oos_stats(self) -> dict:
        return metrics.summary_stats(self.oos_returns)

    def is_robust(self, min_positive_ratio: float = 0.5) -> bool:
        """OOS 是否稳健：至少 min_positive_ratio 的 fold 夏普为正。"""
        if self.folds.empty:
            return False
        positive = (self.folds["sharpe"] > 0).mean()
        return positive >= min_positive_ratio


def walk_forward(
    prices: pd.DataFrame,
    strategy: Strategy,
    *,
    n_folds: int = 5,
    backtester: Backtester | None = None,
) -> WalkForwardResult:
    """滚动样本外验证（anchored walk-forward）。

    把时间轴切成 n_folds 段，逐段把"该段"作为样本外回测。每段都只用
    在它之前的数据所定义的策略 —— 这里策略参数固定，因此等价于在每个
    时间段上检验同一套规则的 OOS 表现，用来观察表现是否随时间稳定，
    而不是只在某一段历史里好看。

    Returns
    -------
    WalkForwardResult
    """
    if n_folds < 2:
        raise ValueError("n_folds 至少为 2")
    bt = backtester or Backtester()

    n = len(prices)
    if n < n_folds * 3:
        raise ValueError(f"数据太短（{n}），不足以切成 {n_folds} 段")

    bounds = [int(n * k / n_folds) for k in range(n_folds + 1)]
    rows = []
    oos_pieces = []
    for k in range(n_folds):
        seg = prices.iloc[bounds[k]: bounds[k + 1]]
        if len(seg) < 3:
            continue
        try:
            res = bt.run(seg, strategy)
        except ValueError:
            continue
        st = metrics.summary_stats(res.returns)
        rows.append(
            {
                "fold": k + 1,
                "start": seg.index[0].date(),
                "end": seg.index[-1].date(),
                "n": len(seg),
                "ann_return": st["annualized_return"],
                "sharpe": st["sharpe"],
                "max_drawdown": st["max_drawdown"],
            }
        )
        oos_pieces.append(res.returns)

    folds_df = pd.DataFrame(rows)
    oos_returns = (
        pd.concat(oos_pieces) if oos_pieces else pd.Series(dtype=float)
    )
    return WalkForwardResult(folds=folds_df, oos_returns=oos_returns, n_folds=len(rows))


def overfitting_report(
    prices: pd.DataFrame,
    strategy: Strategy,
    *,
    train_ratio: float = 0.7,
    backtester: Backtester | None = None,
) -> str:
    """对比样本内 vs 样本外夏普，给出一句人话的过拟合判断。"""
    bt = backtester or Backtester()
    train, test = train_test_split(prices, train_ratio)

    is_res = bt.run(train, strategy)
    oos_res = bt.run(test, strategy)
    is_sharpe = metrics.sharpe_ratio(is_res.returns)
    oos_sharpe = metrics.sharpe_ratio(oos_res.returns)

    decay = is_sharpe - oos_sharpe
    if is_sharpe <= 0:
        verdict = "样本内本身就不赚钱，谈不上过拟合，先回去想逻辑。"
    elif oos_sharpe <= 0:
        verdict = "⚠ 高度可疑：样本内赚钱、样本外亏钱，典型过拟合特征。"
    elif decay > is_sharpe * 0.5:
        verdict = "⚠ 警惕：样本外夏普大幅衰减（>50%），可能过拟合。"
    else:
        verdict = "样本内外较一致，过拟合迹象较弱（但样本仍少，别高兴太早）。"

    return (
        f"===== 过拟合诊断：{strategy.name} =====\n"
        f"  样本内夏普 (IS) : {is_sharpe:.2f}  [{train.index[0].date()} ~ {train.index[-1].date()}]\n"
        f"  样本外夏普 (OOS): {oos_sharpe:.2f}  [{test.index[0].date()} ~ {test.index[-1].date()}]\n"
        f"  衰减            : {decay:.2f}\n"
        f"  判断            : {verdict}"
    )
