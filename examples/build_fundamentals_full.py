"""全市场财务数据回补：数字(全公司,按季度) → 公告日(全A股) → PIT 概览。

一个长驻任务。带 socket 超时保护，避免个别请求挂死。

跑法：
    python examples/build_fundamentals_full.py [起始年] [结束年]
"""
from __future__ import annotations

import socket
import sys

socket.setdefaulttimeout(25)  # 防个别请求挂死

from quantlab.data.fundamentals import (  # noqa: E402
    download_performance, download_announce_dates, quarter_ends,
    a_share_symbols_from_store, build_pit_table, FUND_DIR,
)


def main() -> None:
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2024

    print(f"=== 财务库目录：{FUND_DIR} ===", flush=True)
    print(f"=== 阶段1/2：下载财务数字 {start_year}~{end_year}（全市场，按季度）===", flush=True)
    perf = download_performance(quarter_ends(start_year, end_year), sleep=0.4, verbose=True)
    print(f"数字库：{len(perf)} 条 / {perf['symbol'].nunique()} 只", flush=True)

    symbols = a_share_symbols_from_store()
    print(f"\n=== 阶段2/2：抓取公告日（{len(symbols)} 只 A 股）===", flush=True)
    ann = download_announce_dates(
        symbols, start_date=f"{start_year}0101", end_date=f"{end_year}1231",
        sleep=0.15, verbose=True)
    print(f"公告日库：{len(ann)} 条 / {ann['symbol'].nunique() if len(ann) else 0} 只", flush=True)

    pit = build_pit_table()
    real = pit["announce_is_real"].mean() if "announce_is_real" in pit else 0
    print(f"\n=== 完成 ===", flush=True)
    print(f"PIT 表：{len(pit)} 条，真实公告日覆盖 {real:.0%}（其余用法定截止日兜底）", flush=True)


if __name__ == "__main__":
    main()
