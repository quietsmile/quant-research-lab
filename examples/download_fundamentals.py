"""下载 A 股全市场财务数据（业绩报表 + 公告日），建 point-in-time 财务库。

一次调用一个季度即覆盖全市场所有公司，按季度迭代即可。可反复跑、逐步
拉长年份，数据按 (symbol, report_period) 去重累积。

跑法：
    python examples/download_fundamentals.py [起始年] [结束年] [每季间隔秒]
    例：python examples/download_fundamentals.py 2018 2024 0.5
"""
from __future__ import annotations

import sys

from quantlab.data.fundamentals import (
    download_performance, quarter_ends, coverage, FUND_DIR,
)


def main() -> None:
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2020
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    sleep = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    periods = quarter_ends(start_year, end_year)
    print(f"财务库目录：{FUND_DIR}")
    print(f"下载区间：{start_year}~{end_year}，共 {len(periods)} 个季度\n")

    merged = download_performance(periods, save=True, sleep=sleep)
    if merged.empty:
        print("未下载到数据（检查网络/数据源）。")
        return

    print(f"\n财务库累计 {len(merged)} 条 / {merged['symbol'].nunique()} 只公司")
    cov = coverage()
    print("\n各报告期覆盖（部分）：")
    print(cov.tail(8).to_string(index=False))
    print(f"\n用法：from quantlab.data.fundamentals import point_in_time")
    print(f"      point_in_time('2024-06-15')  # 只返回截至该日已公告的最新财报")


if __name__ == "__main__":
    main()
