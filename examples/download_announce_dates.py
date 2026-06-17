"""为全市场 A 股抓取定期报告的真实公告日（cninfo），建 PIT 公告日库。

公告日是 PIT 的基准。本脚本逐只抓取沪深京 A 股的年报/半年报/一/三季报
首发公告时间，可反复跑、断点累积（按 (symbol,报告期) 去重）。

跑法：
    python examples/download_announce_dates.py [起始YYYYMMDD] [结束YYYYMMDD] [每只间隔秒] [scope]
    scope: all = 财务库里全部A股(默认) | universe = 仅 20 只标的池
    例：python examples/download_announce_dates.py 20150101 20251231 0.2 all
"""
from __future__ import annotations

import sys

from quantlab.data.fundamentals import (
    download_announce_dates, a_share_symbols_from_store, FUND_DIR,
)
from quantlab.data.universe import default_universe


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "20150101"
    end = sys.argv[2] if len(sys.argv) > 2 else "20251231"
    sleep = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    scope = sys.argv[4] if len(sys.argv) > 4 else "all"

    if scope == "universe":
        symbols = default_universe()
    else:
        symbols = a_share_symbols_from_store()  # 需先跑 download_fundamentals 建数字库

    print(f"公告日库目录：{FUND_DIR}")
    print(f"抓取 {len(symbols)} 只 A 股的定期报告公告日，{start}~{end}\n")
    ann = download_announce_dates(symbols, start_date=start, end_date=end,
                                  sleep=sleep, verbose=True)
    print(f"\n完成：公告日库累计 {len(ann)} 条 / {ann['symbol'].nunique() if len(ann) else 0} 只。")
    print("现在 point_in_time(date) 会优先用真实公告日，缺失的用法定截止日兜底。")


if __name__ == "__main__":
    main()
