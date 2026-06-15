"""量化研究方法论（最重要、最容易被跳过）。

对应 todo.txt 能力点（第一位）：过拟合、样本内/外、walk-forward、
多重检验意识。这里把"方法论"从知识变成**默认执行的函数**。
"""
from quantlab.validation.walk_forward import (
    train_test_split,
    walk_forward,
    overfitting_report,
)

__all__ = ["train_test_split", "walk_forward", "overfitting_report"]
