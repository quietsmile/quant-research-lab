"""等权买入持有策略 · 全周期指标看板。读 strat_viz.parquet, plotly 可视化。
运行: python3 -m streamlit run dashboard/strategy_app.py --server.address 0.0.0.0 --server.port 14159 ...
"""
from __future__ import annotations
import numpy as np, pandas as pd, pathlib
import plotly.graph_objects as go
import streamlit as st

ANN = 242
D = pathlib.Path.home() / ".local/share/quantlab/fundamentals"
st.set_page_config(page_title="等权策略全周期", layout="wide")
st.title("📈 等权买入持有策略 · 全周期指标")
st.caption("股票池=全市场流动性Top2000 | 2017-01 ~ 2026-06 | 数据: strat_viz.parquet")


@st.cache_data
def load():
    df = pd.read_parquet(D / "strat_viz.parquet")
    df.index = pd.to_datetime(df.index)
    return df


df = load()
cols = list(df.columns)
primary = st.sidebar.selectbox("主策略", cols, index=cols.index("月度再平衡等权(扣成本)") if "月度再平衡等权(扣成本)" in cols else 0)
logy = st.sidebar.checkbox("净值对数轴", value=True)
others = st.sidebar.multiselect("对比叠加", [c for c in cols if c != primary], default=[c for c in cols if c != primary])

def metrics(r):
    r = r.dropna()
    nav = (1 + r).cumprod()
    yrs = len(r) / ANN
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(ANN)
    sh = r.mean() / (r.std() + 1e-12) * np.sqrt(ANN)
    dd = (nav / nav.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    mret = (1 + r).resample("M").prod() - 1
    winm = (mret > 0).mean()
    return dict(年化收益=cagr, 年化波动=vol, 夏普=sh, 最大回撤=dd, 卡玛=calmar, 月胜率=winm,
                累计=nav.iloc[-1] - 1, nav=nav)


m = metrics(df[primary])
c = st.columns(6)
c[0].metric("年化收益", f"{m['年化收益']*100:.1f}%")
c[1].metric("夏普", f"{m['夏普']:.2f}")
c[2].metric("最大回撤", f"{m['最大回撤']*100:.1f}%")
c[3].metric("年化波动", f"{m['年化波动']*100:.1f}%")
c[4].metric("卡玛比", f"{m['卡玛']:.2f}")
c[5].metric("月胜率", f"{m['月胜率']*100:.0f}%")

# 净值曲线
st.subheader("净值曲线")
fig = go.Figure()
fig.add_trace(go.Scatter(x=m["nav"].index, y=m["nav"], name=primary, line=dict(width=2.5, color="crimson")))
for o in others:
    nav = (1 + df[o].dropna()).cumprod()
    is_mkt = "大盘" in o
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name=o,
                             line=dict(width=1.6 if is_mkt else 1.2,
                                       dash="dash" if is_mkt else "dot",
                                       color="dimgray" if "沪深300" in o else ("silver" if "上证" in o else None))))
fig.update_layout(height=420, yaxis_type="log" if logy else "linear", legend=dict(orientation="h"), margin=dict(t=10))
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
# 回撤
with col1:
    st.subheader("回撤")
    dd = m["nav"] / m["nav"].cummax() - 1
    figd = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color="steelblue")))
    figd.update_layout(height=300, yaxis_title="回撤%", margin=dict(t=10))
    st.plotly_chart(figd, use_container_width=True)
# 年度收益
with col2:
    st.subheader("年度收益")
    yr = (1 + df[primary].dropna()).resample("Y").prod() - 1
    yr.index = yr.index.year
    figy = go.Figure(go.Bar(x=yr.index.astype(str), y=yr * 100,
                            marker_color=["crimson" if v > 0 else "seagreen" for v in yr]))
    figy.update_layout(height=300, yaxis_title="收益%", margin=dict(t=10))
    st.plotly_chart(figy, use_container_width=True)

# 滚动1年夏普
st.subheader("滚动1年夏普")
rs = df[primary].rolling(ANN).mean() / (df[primary].rolling(ANN).std() + 1e-12) * np.sqrt(ANN)
figr = go.Figure(go.Scatter(x=rs.index, y=rs, line=dict(color="darkorange")))
figr.add_hline(y=0, line_dash="dash", line_color="gray")
figr.update_layout(height=260, margin=dict(t=10))
st.plotly_chart(figr, use_container_width=True)

# 指标对比表
st.subheader("各等权/持有变体 全周期指标对比")
rows = []
for col in cols:
    mm = metrics(df[col])
    rows.append({"策略": col, "年化收益": f"{mm['年化收益']*100:.1f}%", "夏普": f"{mm['夏普']:.2f}",
                 "最大回撤": f"{mm['最大回撤']*100:.1f}%", "年化波动": f"{mm['年化波动']*100:.1f}%",
                 "卡玛": f"{mm['卡玛']:.2f}", "月胜率": f"{mm['月胜率']*100:.0f}%", "累计收益": f"{mm['累计']*100:.0f}%"})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption("说明: ①每日再平衡等权(免成本,研究基准) ②真买入持有等权(漂移) ③月度再平衡等权(扣成本,最现实) ④市值加权买入持有(≈宽基指数)。"
           "等权再平衡赚小盘溢价+再平衡溢价; 市值加权全周期弱但2026靠龙头暴涨。")

# ===================== Barra 风格归因 =====================
import json as _json
st.markdown("---")
st.header("🧬 Barra 风格归因（这策略到底赚的什么钱）")
try:
    attr = _json.load(open(D / "barra_attr.json", encoding="utf-8"))
except Exception:
    attr = {}
if attr:
    bkey = primary if primary in attr else ("月度再平衡等权(扣成本)" if "月度再平衡等权(扣成本)" in attr else list(attr)[0])
    a = attr[bkey]
    st.caption(f"归因对象: **{bkey}** | 把日收益回归到 市场/规模SMB/价值HML/动量WML/低波LMH 五风格")
    k = st.columns(3)
    k[0].metric("风格解释度 R²", f"{a['r2']*100:.0f}%")
    k[1].metric("残余 alpha(年化)", f"{a['alpha_ann']*100:+.1f}%")
    k[2].metric("alpha t值", f"{a['alpha_t']:.1f}", help="|t|>2 才算统计显著")
    cA, cB = st.columns(2)
    with cA:
        st.subheader("风格暴露 (beta)")
        bb = a["betas"]
        figb = go.Figure(go.Bar(x=list(bb.keys()), y=list(bb.values()),
                                marker_color=["crimson" if abs(v) >= 0.3 else "steelblue" for v in bb.values()],
                                text=[f"{v:+.2f}" for v in bb.values()], textposition="outside"))
        figb.update_layout(height=300, margin=dict(t=10), yaxis_title="beta")
        st.plotly_chart(figb, use_container_width=True)
    with cB:
        st.subheader("年化收益归因")
        cc = dict(a["contrib"]); cc["残余alpha"] = a["alpha_ann"]
        figc = go.Figure(go.Bar(x=[f"{v*100:+.1f}%" for v in cc.values()], y=list(cc.keys()), orientation="h",
                                marker_color=["seagreen" if v >= 0 else "indianred" for v in cc.values()],
                                text=[f"{v*100:+.1f}%" for v in cc.values()], textposition="outside"))
        figc.update_layout(height=300, margin=dict(t=10), xaxis_title="年化贡献")
        st.plotly_chart(figc, use_container_width=True)
    # 滚动SMB暴露
    try:
        rl = pd.read_parquet(D / "barra_rolling.parquet"); rl.index = pd.to_datetime(rl.index)
        smbcol = [c for c in rl.columns if bkey in c and c.endswith("SMB")]
        if smbcol:
            st.subheader("滚动1年 规模(SMB)暴露 —— 小盘倾斜随时间")
            figs = go.Figure(go.Scatter(x=rl.index, y=rl[smbcol[0]], line=dict(color="purple")))
            figs.add_hline(y=0, line_dash="dash", line_color="gray")
            figs.update_layout(height=240, margin=dict(t=10), yaxis_title="SMB beta")
            st.plotly_chart(figs, use_container_width=True)
    except Exception:
        pass
    st.info("Barra 结论: 等权策略 ~93% 收益由风格解释 —— **市场beta + 强小盘倾斜(SMB≈0.5) + 偏高波动**；"
            "残余 alpha 仅 +3~5%/年且 **t值<2 不显著**。即它的超额≈小盘溢价(一个会变的风格 regime)，**不是真 alpha**。"
            "若未来风格转向大盘龙头(如AI巨头驱动)，这条小盘腿会反噬。")
else:
    st.warning("未找到 barra_attr.json，请先运行 examples/precompute_barra_attribution.py")
