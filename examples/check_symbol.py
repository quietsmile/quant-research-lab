"""检查某个 A 股代码能否拿到**真实**行情（以及落到哪个数据源）。

跑法：
    python examples/check_symbol.py 600519 000858 300750 688981 830799
"""
from __future__ import annotations

import sys

from quantlab.data import load_prices


def main() -> None:
    codes = sys.argv[1:] or ["600519", "000858", "300750", "688981", "830799"]
    print(f"{'代码':<10}{'来源':<14}{'真实?':<8}{'天数':<8}末收")
    print("-" * 46)
    for code in codes:
        try:
            df = load_prices(code, "2023-06-01", "2023-09-30",
                             source="auto", use_cache=False)
            real = "✅是" if df.attrs.get("is_real") else "⚠合成"
            print(f"{code:<10}{df.attrs.get('source',''):<14}{real:<8}"
                  f"{len(df):<8}{df['close'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"{code:<10}{'-':<14}{'❌失败':<8}{type(e).__name__}: {str(e)[:30]}")
    print("\n说明：来源=synthetic 表示真实源拿不到（如北交所/被墙），用的是合成数据，"
          "\n      不能当真实回测结论。北交所(8xxxxx/920xxx)请用 AKShare（需可访问其数据源）。")


if __name__ == "__main__":
    main()
