"""Quant Research Lab 仪表盘（Streamlit）。

跑法：
    streamlit run dashboard/app.py

紧凑布局：左侧选标的/策略/参数，右侧顶部指标卡 + 多图。未来要做指标监控，
在 core.py 加图函数、这里加一个 tab 即可。
"""
from __future__ import annotations

import streamlit as st

from dashboard import core
from quantlab.data.universe import LIQUID_LEADERS

st.set_page_config(page_title="Quant Research Lab", layout="wide")
st.title("📊 Quant Research Lab · 策略回测看板")

# ---------------- 侧边栏：控制项 ----------------
with st.sidebar:
    st.header("参数")
    code = st.selectbox(
        "标的", options=list(LIQUID_LEADERS.keys()),
        format_func=lambda c: f"{c} {LIQUID_LEADERS[c]}",
    )
    strat_name = st.selectbox("策略", options=list(core.STRATEGY_REGISTRY.keys()))
    col1, col2 = st.columns(2)
    start = col1.text_input("开始", "2018-01-01")
    end = col2.text_input("结束", "2024-12-31")

    # 按策略动态出参数控件
    params: dict = {}
    if strat_name == "均线趋势 MA":
        params["fast"] = st.slider("快线", 5, 60, 20)
        params["slow"] = st.slider("慢线", 20, 120, 60)
    elif strat_name == "通道突破 Donchian":
        params["entry"] = st.slider("入场窗口", 5, 60, 20)
        params["exit"] = st.slider("出场窗口", 5, 40, 10)
    elif strat_name == "均值回归 Bollinger":
        params["window"] = st.slider("窗口", 5, 60, 20)
        params["k"] = st.slider("标准差倍数 k", 1.0, 3.0, 2.0, 0.1)

    run_btn = st.button("运行回测", type="primary", use_container_width=True)


@st.cache_data(show_spinner="回测中…")
def _run(code, start, end, strat_name, params):
    return core.run(code, start, end, strat_name, params)


if run_btn or True:  # 首次加载也跑一次默认
    try:
        out = _run(code, start, end, strat_name, params)
    except Exception as e:  # noqa: BLE001
        st.error(f"回测失败：{e}")
        st.stop()

    s, bs = out["stats"], out["bench_stats"]
    tag = "✅ 真实行情" if out["is_real"] else "⚠️ 合成兜底(非真实)"
    st.caption(f"数据来源：{out['source']} · {tag}　|　"
               f"{out['prices'].index[0].date()} ~ {out['prices'].index[-1].date()}　|　"
               f"{len(out['prices'])} 交易日")

    # ---- 指标卡（策略 vs 买入持有 delta）----
    c = st.columns(5)
    c[0].metric("累计收益", f"{s['cumulative_return']:+.1%}",
                f"{s['cumulative_return']-bs['cumulative_return']:+.1%} vs 持有")
    c[1].metric("年化", f"{s['annualized_return']:+.1%}")
    c[2].metric("夏普", f"{s['sharpe']:.2f}", f"{s['sharpe']-bs['sharpe']:+.2f} vs 持有")
    c[3].metric("最大回撤", f"{s['max_drawdown']:.1%}",
                f"{s['max_drawdown']-bs['max_drawdown']:+.1%} vs 持有", delta_color="inverse")
    c[4].metric("年换手", f"{s['turnover_annual']:.1f}x")

    # ---- 图表 ----
    left, right = st.columns(2)
    left.plotly_chart(core.equity_figure(out["result"], out["benchmark"]),
                      use_container_width=True)
    right.plotly_chart(core.price_trades_figure(out["prices"], out["result"]),
                       use_container_width=True)
    left.plotly_chart(core.drawdown_figure(out["result"]), use_container_width=True)
    right.plotly_chart(core.position_figure(out["result"]), use_container_width=True)

    with st.expander("成交明细"):
        st.dataframe(out["result"].trades, use_container_width=True)
