"""离线数据同步：把标的池的真实行情**增量**下载、累积进本地 parquet 数据仓。

可以"慢慢下载"——重复跑、逐步拉长区间，数据会不断累积去重，断网后用
`load_prices(code, source="offline")` 即可离线回测。

跑法：
    python examples/sync_offline.py [开始] [结束] [间隔秒] [数据源]
    例：python examples/sync_offline.py 2010-01-01 2024-12-31 1.0 yahoo
"""
from __future__ import annotations

import sys
import time

from quantlab.data import load_prices, update_offline, list_offline, OFFLINE_DIR
from quantlab.data.universe import LIQUID_LEADERS


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "2015-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2024-12-31"
    delay = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    source = sys.argv[4] if len(sys.argv) > 4 else "auto"

    print(f"离线仓目录：{OFFLINE_DIR}")
    n = len(LIQUID_LEADERS)
    ok = 0
    for i, (code, name) in enumerate(LIQUID_LEADERS.items(), 1):
        try:
            df = load_prices(code, start, end, source=source)
            if not df.attrs.get("is_real"):
                print(f"[{i:>2}/{n}] {code} {name:<6} ⚠跳过（非真实数据，未入库）")
            else:
                merged = update_offline(code, df)
                ok += 1
                print(f"[{i:>2}/{n}] {code} {name:<6} ✅ 入库累计 {len(merged)} 天 "
                      f"({merged.index.min().date()}~{merged.index.max().date()})")
        except Exception as e:  # noqa: BLE001
            print(f"[{i:>2}/{n}] {code} {name:<6} ❌ {type(e).__name__}: {str(e)[:40]}")
        if i < n:
            time.sleep(delay)

    print(f"\n完成：{ok}/{n} 只已入离线仓。")
    inv = list_offline()
    if not inv.empty:
        print(f"离线仓现有 {len(inv)} 只，累计 {int(inv['n_days'].sum())} 条交易日记录。")
    print('用法：load_prices("600519", "2018-01-01", "2023-12-31", source="offline")')


if __name__ == "__main__":
    main()
