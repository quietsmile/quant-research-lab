"""前瞻事件策略 · Test(2024–2025) 操作可视化看板。

直观看：策略在 Test 上做了哪些操作——净值 vs 大盘、持仓数量随时间、逐笔买卖(入场时间×
超额收益)、明细表。数据由 examples/event_viz_data.py 预计算。
跑法：streamlit run dashboard/event_app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quantlab.data.tushare_adapter import _FUND_DIR
from quantlab import report as rp

st.set_page_config(page_title="前瞻事件策略 · Test 操作", layout="wide")
st.title("📈 前瞻事件策略 · 样本外(2024–2025) 操作可视化")
st.caption("规则：正向业绩预告(预增/扭亏等) 且 预告增长后 forward PE **低于所属行业中位**(行业相对便宜) "
           "→ 预告次日买入，持有到该期正式财报前最后一天卖出。**仓位：按预告增速取 Top-20 等权、单只≤5%，"
           "其余现金**。下图为**冻结 Test** 真实操作，收益均为**扣成本净值**。")
st.info("💡 用规范仓位(Top-20/单只≤5%)后，Test 策略累计 +40.4% **略输**等权大盘 +45.1%——"
        "说明早先'大幅跑赢'多来自**极端集中(淡季单押1-2只)**的人为效应；仓位一规范，净收益的超额基本消失。"
        "这正是'买入量级/持仓比例'要讲清的原因。")


@st.cache_data
def load():
    tr = pd.read_parquet(_FUND_DIR / "viz_trades.parquet")
    eq = pd.read_parquet(_FUND_DIR / "viz_equity.parquet")
    return tr, eq


try:
    trades, eq = load()
except Exception as e:  # noqa: BLE001
    st.error(f"数据未就绪，请先运行 examples/event_viz_data.py：{e}")
    st.stop()

# ---- 顶部指标卡 ----
tot_strat = eq["strategy"].iloc[-1] - 1
tot_mkt = eq["market"].iloc[-1] - 1
c = st.columns(5)
c[0].metric("Test 累计净收益", f"{tot_strat:+.1%}", f"{tot_strat - tot_mkt:+.1%} vs 大盘")
c[1].metric("等权大盘", f"{tot_mkt:+.1%}")
c[2].metric("交易笔数", f"{len(trades)}")
c[3].metric("逐笔胜率(超额)", f"{(trades['excess'] > 0).mean():.0%}")
c[4].metric("平均持有", f"{int(trades['hold_days'].median())}天")

left, right = st.columns([3, 2])

# ---- 净值曲线 ----
fig = go.Figure()
fig.add_trace(go.Scatter(x=eq["date"], y=eq["strategy"], name="策略(净值)", line=dict(width=2, color="crimson")))
fig.add_trace(go.Scatter(x=eq["date"], y=eq["market"], name="等权大盘", line=dict(width=1, dash="dash", color="gray")))
fig.update_layout(title="净值曲线（扣成本，归一）", height=320, margin=dict(l=10, r=10, t=40, b=10),
                  legend=dict(orientation="h"))
left.plotly_chart(fig, use_container_width=True)

# ---- 持仓数量 ----
figh = go.Figure()
figh.add_trace(go.Scatter(x=eq["date"], y=eq["holdings"], fill="tozeroy", line=dict(color="steelblue")))
figh.update_layout(title="持仓数量随时间", height=320, margin=dict(l=10, r=10, t=40, b=10))
right.plotly_chart(figh, use_container_width=True)

# ---- 逐笔散点：入场时间 × 超额收益 ----
trades["盈亏"] = trades["excess"].apply(lambda x: "盈" if x > 0 else "亏")
figs = px.scatter(trades, x="entry", y="excess", color="盈亏",
                  color_discrete_map={"盈": "#d62728", "亏": "#2ca02c"},
                  hover_data=["name", "symbol", "exit", "hold_days", "ret"],
                  labels={"entry": "买入日期", "excess": "相对大盘超额收益"},
                  title="每一笔操作（点=一次买入→持有到财报前卖出；颜色=是否跑赢大盘）")
figs.add_hline(y=0, line_dash="dot", line_color="gray")
figs.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), yaxis_tickformat=".0%")
st.plotly_chart(figs, use_container_width=True)

# ---- 明细表（最赚 / 最亏）----
cc = st.columns(2)
show = trades.copy()
show["ret"] = (show["ret"] * 100).round(1)
show["excess"] = (show["excess"] * 100).round(1)
if "weight" in show:
    show["权重%"] = (show["weight"] * 100).round(1)
show = show.rename(columns={"name": "名称", "entry": "买入", "exit": "卖出",
                            "hold_days": "持有天", "ret": "收益%", "excess": "超额%"})
cols = ["名称", "symbol", "买入", "卖出", "持有天", "权重%", "收益%", "超额%"]
cols = [c for c in cols if c in show.columns]
cc[0].subheader("赚得最多的 10 笔"); cc[0].dataframe(show.nlargest(10, "超额%")[cols], use_container_width=True)
cc[1].subheader("亏得最多的 10 笔"); cc[1].dataframe(show.nsmallest(10, "超额%")[cols], use_container_width=True)

with st.expander("全部交易明细"):
    st.dataframe(show[cols].sort_values("买入"), use_container_width=True)
