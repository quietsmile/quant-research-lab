"""数据储备：把一批高流动性 A 股的真实日线落到本地缓存，并生成质量报告。

这是"数据基建"的第一块砖：在 QMT/xtquant 到位前，先用免费真实源把
一个可复现的高质量数据集攒下来，供多标的、多时段的稳健性检验使用。

特点：
- 走 load_prices(source="auto")，带本地缓存，重复跑只补缺失。
- 礼貌限速（默认每只间隔 1.2s），避免 Yahoo 429。
- 产出 data_manifest.csv：每只的天数/区间/是否真实/缺口/异常跳变，
  一眼看清数据质量。

跑法：
    python examples/build_dataset.py [开始] [结束] [间隔秒]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from quantlab.data import load_prices
from quantlab.data.universe import LIQUID_LEADERS


def _quality(df: pd.DataFrame) -> dict:
    """简单数据质量指标：交易日数、缺口、单日异常跳变（|日涨跌|>20%）。"""
    rets = df["close"].pct_change()
    # 期望交易日数 vs 实际（用工作日近似，缺口比例反映停牌/缺失）
    span_bdays = len(pd.bdate_range(df.index[0], df.index[-1]))
    coverage = len(df) / span_bdays if span_bdays else float("nan")
    abnormal = int((rets.abs() > 0.20).sum())  # A股主板单日理论上限 ~10%，>20%多为数据错误/除权未复权
    return {
        "n_days": len(df),
        "start": df.index[0].date(),
        "end": df.index[-1].date(),
        "coverage": round(coverage, 3),
        "abnormal_jumps": abnormal,
    }


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "2015-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2024-12-31"
    delay = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2

    rows = []
    n = len(LIQUID_LEADERS)
    for i, (code, name) in enumerate(LIQUID_LEADERS.items(), 1):
        try:
            df = load_prices(code, start, end, source="auto")
            q = _quality(df)
            real = bool(df.attrs.get("is_real"))
            rows.append({"code": code, "name": name, "source": df.attrs.get("source"),
                         "is_real": real, **q})
            flag = "✅" if real else "⚠合成"
            print(f"[{i:>2}/{n}] {code} {name:<6} {flag} {q['n_days']}天 "
                  f"{q['start']}~{q['end']} 覆盖{q['coverage']:.0%} 异常{q['abnormal_jumps']}")
        except Exception as e:
            print(f"[{i:>2}/{n}] {code} {name:<6} ❌ {type(e).__name__}: {str(e)[:40]}")
            rows.append({"code": code, "name": name, "source": "error", "is_real": False})
        if i < n:
            time.sleep(delay)   # 礼貌限速，避免 429

    manifest = pd.DataFrame(rows)
    out = "data_manifest.csv"
    manifest.to_csv(out, index=False)
    real_n = int(manifest["is_real"].sum())
    print(f"\n清单已写入 {out}：{real_n}/{n} 只拿到真实数据。")
    print("提示：合成/失败的标的（如北交所或被限流）等 QMT/xtquant 接入后再补真实数据。")


if __name__ == "__main__":
    main()
