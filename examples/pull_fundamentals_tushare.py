"""在【固定公网 IP】的机器上运行：用 Tushare 拉全 A 股 PIT 财务，存 parquet。

为什么要在固定 IP 机器跑：Tushare 限制每 token 绑定的 IP 数；多出口 NAT 环境
每次请求换 IP 会触发"IP 数量超限"。在单 IP 机器上跑就没这问题。

产物 parquet 可直接拷到共享盘（如 /newcpfs/.../share/），本框架用
`tushare_adapter.load_pit()` / `point_in_time()` 读取，无需联网。

用法（在固定 IP 机器）：
    export TUSHARE_TOKEN=你的token
    python examples/pull_fundamentals_tushare.py [起始YYYYMMDD] [结束] [输出parquet] [scope]
    scope: universe(默认,20只) | all(全A股,约5500只,慢且耗积分)
"""
from __future__ import annotations

import sys
import time

from quantlab.data import tushare_adapter as tsa
from quantlab.data.universe import default_universe


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "20150101"
    end = sys.argv[2] if len(sys.argv) > 2 else "20251231"
    out = sys.argv[3] if len(sys.argv) > 3 else str(tsa.TS_FUND_FILE)
    scope = sys.argv[4] if len(sys.argv) > 4 else "universe"

    if scope == "all":
        pro = tsa.get_pro()
        basic = pro.stock_basic(exchange="", list_status="L", fields="symbol")
        symbols = sorted(basic["symbol"].tolist())
    else:
        symbols = default_universe()

    print(f"拉取 {len(symbols)} 只，{start}~{end} → {out}", flush=True)
    df = tsa.fundamentals_pit(symbols, start, end, sleep=0.4, verbose=True)
    if df.empty:
        print("未拉到数据（检查 token / IP 限制 / 积分）。")
        return
    import pandas as pd
    pd.DataFrame(df).to_parquet(out, index=False)
    print(f"\n完成：{len(df)} 条 / {df['symbol'].nunique()} 只 → {out}")
    print("把该 parquet 拷到共享盘后，本框架 tushare_adapter.load_pit() 即可读。")


if __name__ == "__main__":
    main()
