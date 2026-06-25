"""量化实验室看板 (多页): ①强势板块回撤买入(可调参) ②等权策略+Barra归因。
run: python3 -m streamlit run dashboard/lab_app.py --server.address 0.0.0.0 --server.port 14159 ...
"""
from __future__ import annotations
import json, pathlib
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st

ANN = 242
D = pathlib.Path.home() / ".local/share/quantlab/fundamentals"
DD = pathlib.Path(__file__).resolve().parent.parent / "dashboard_data"
st.set_page_config(page_title="量化实验室", layout="wide")
page = st.sidebar.radio("页面", ["🔥 热门板块·短期趋势+主力资金(可调参)", "🎯 强势板块·回撤买入(可调参)", "💰 管理层forwardPE低估(排周期股·可调参)", "📈 等权策略 + Barra归因"])


@st.cache_data
def load_pullback():
    g = lambda n: pd.read_parquet(DD / f"pullback_{n}.parquet")
    close = g("close"); close.index = pd.to_datetime(close.index)
    ret = g("ret"); ret.index = pd.to_datetime(ret.index)
    sret = g("sret"); sret.index = pd.to_datetime(sret.index)
    roe = g("roe"); roe.index = pd.to_datetime(roe.index)
    gm = g("gm"); gm.index = pd.to_datetime(gm.index)
    mv = g("mv"); mv.index = pd.to_datetime(mv.index)
    bench = g("bench"); bench.index = pd.to_datetime(bench.index)
    sector = g("sector")["industry"]
    return close, ret, sret, roe, gm, mv, bench, sector


def stat(x):
    x = x.dropna()
    if len(x) < 20: return dict(cagr=np.nan, sh=np.nan, dd=np.nan, nav=pd.Series(dtype=float))
    nav = (1 + x).cumprod()
    cagr = nav.iloc[-1] ** (ANN / len(x)) - 1
    sh = x.mean() / (x.std() + 1e-12) * np.sqrt(ANN)
    dd = (nav / nav.cummax() - 1).min()
    return dict(cagr=cagr, sh=sh, dd=dd, nav=nav)


def render_pullback():
    st.title("🎯 强势板块 · 回撤买入策略 (可调参)")
    st.caption("逻辑: 在大涨且趋势在上的板块里, 对优质股票, 每次回撤到均线就买。股票池=流动性Top1500。")
    if not (DD / "pullback_close.parquet").exists():
        st.error("数据底座未就绪, 请先运行 examples/precompute_pullback_data.py"); return
    close, ret, sret, roe, gm, mv, bench, sector = load_pullback()
    syms = list(close.columns); idx = close.index

    s = st.sidebar
    s.markdown("### 板块筛选")
    gw = s.slider("板块涨幅窗口(交易日)", 10, 60, 30, 5)
    gth = s.slider("板块涨幅阈值 %", 5, 80, 30, 5) / 100
    trend_ma = s.slider("趋势确认均线(板块, 日)", 20, 120, 60, 10)
    s.markdown("### 回撤买入")
    maw = s.slider("买入均线周期(个股, 日)", 5, 60, 20, 1)
    band = s.slider("回撤带宽 %(≤均线×(1+带))", 0, 10, 2, 1) / 100
    s.markdown("### 质量筛选")
    use_q = s.checkbox("启用质量筛选(扣非ROE>0)", True)
    q_gm = s.slider("毛利率分位下限", 0.0, 0.9, 0.3, 0.1)
    q_mv = s.slider("市值分位下限(剔微盘)", 0.0, 0.9, 0.2, 0.1)
    s.markdown("### 交易")
    rebal = s.slider("调仓间隔(日)", 1, 20, 5, 1)
    cost = s.slider("单边成本 %", 0.0, 0.3, 0.1, 0.05) / 100
    bname = s.selectbox("对比基准", list(bench.columns))

    # ---- 向量化回测 ----
    snav = (1 + sret.fillna(0)).cumprod()
    hot = (snav / snav.shift(gw) - 1) > gth
    uptrend = snav >= snav.rolling(trend_ma, min_periods=max(5, trend_ma // 2)).mean()
    sec_ok = (hot & uptrend)
    smap = sector.reindex(syms).values
    stock_sec_ok = sec_ok.reindex(columns=smap); stock_sec_ok.columns = syms
    stock_sec_ok = stock_sec_ok.fillna(False)
    ma = close.rolling(maw, min_periods=max(3, maw // 2)).mean()
    pullback = close <= ma * (1 + band)
    qual = pd.DataFrame(True, index=idx, columns=syms)
    if use_q:
        qual = (roe > 0) & (gm.rank(axis=1, pct=True) >= q_gm) & (mv.rank(axis=1, pct=True) >= q_mv)
    elig = stock_sec_ok & pullback & qual.reindex(index=idx, columns=syms).fillna(False) & ret.notna()
    target = elig.astype(float)
    rb = np.zeros(len(idx), bool); rb[::rebal] = True
    held = target.where(pd.Series(rb, index=idx), other=np.nan).ffill().fillna(0.0)
    rowsum = held.sum(axis=1).replace(0, np.nan)
    w = held.div(rowsum, axis=0).fillna(0.0)
    rfwd = ret.shift(-1).clip(-0.21, 0.21)
    gross = (w * rfwd).sum(axis=1)
    turn = (w - w.shift(1)).abs().sum(axis=1)
    port = (gross - turn * cost)
    nhold = (w > 0).sum(axis=1)

    eval_idx = idx[idx >= pd.Timestamp("2018-01-01")]
    p = port.reindex(eval_idx); b = bench[bname].reindex(eval_idx)
    sp, sb = stat(p), stat(b)
    if sp["nav"].empty or nhold.reindex(eval_idx).mean() < 1:
        st.warning("当前参数下几乎选不到股票(板块阈值太高/质量太严)。放宽参数试试。")
    c = st.columns(5)
    c[0].metric("年化收益", f"{sp['cagr']*100:.1f}%" if np.isfinite(sp['cagr']) else "—")
    c[1].metric("夏普", f"{sp['sh']:.2f}" if np.isfinite(sp['sh']) else "—")
    c[2].metric("最大回撤", f"{sp['dd']*100:.1f}%" if np.isfinite(sp['dd']) else "—")
    c[3].metric(f"超额({bname})", f"{(sp['cagr']-sb['cagr'])*100:+.1f}%" if np.isfinite(sp['cagr']) else "—")
    c[4].metric("平均持股数", f"{nhold.reindex(eval_idx).mean():.0f}")

    st.subheader("净值曲线 vs 基准")
    fig = go.Figure()
    if not sp["nav"].empty:
        fig.add_trace(go.Scatter(x=sp["nav"].index, y=sp["nav"], name="策略", line=dict(width=2.5, color="crimson")))
    fig.add_trace(go.Scatter(x=sb["nav"].index, y=sb["nav"], name=bname, line=dict(width=1.4, dash="dash", color="gray")))
    fig.update_layout(height=400, yaxis_type="log", legend=dict(orientation="h"), margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("逐年超额(vs基准)")
        py = (1 + p.fillna(0)).resample("Y").prod() - 1
        by = (1 + b.fillna(0)).resample("Y").prod() - 1
        ex = (py - by) * 100; ex.index = ex.index.year
        figy = go.Figure(go.Bar(x=ex.index.astype(str), y=ex, marker_color=["crimson" if v > 0 else "seagreen" for v in ex]))
        figy.update_layout(height=280, yaxis_title="超额%", margin=dict(t=10))
        st.plotly_chart(figy, use_container_width=True)
    with col2:
        st.subheader("每日持股数(仓位活跃度)")
        figh = go.Figure(go.Scatter(x=eval_idx, y=nhold.reindex(eval_idx), fill="tozeroy", line=dict(color="steelblue")))
        figh.update_layout(height=280, margin=dict(t=10))
        st.plotly_chart(figh, use_container_width=True)
    st.info("提示: 这是'动量中低吸'策略——板块强(大资金在)+趋势在上(临时资金入场)+个股回撤到均线买入+质量筛选。"
            "回撤买入本质是在强势里做均值回归。注意: 阈值越高/质量越严→持股越少→越集中越波动。无独立卖出规则时, 退出靠'板块转弱或股价远离均线'自然发生。")


def render_equalweight():
    st.title("📈 等权买入持有 · 全周期 + Barra归因")
    if not (D / "strat_viz.parquet").exists():
        st.error("缺 strat_viz.parquet"); return
    df = pd.read_parquet(D / "strat_viz.parquet"); df.index = pd.to_datetime(df.index)
    cols = list(df.columns)
    primary = st.sidebar.selectbox("主策略", cols, index=cols.index("月度再平衡等权(扣成本)") if "月度再平衡等权(扣成本)" in cols else 0)
    m = stat(df[primary]); mret = (1 + df[primary].dropna()).resample("M").prod() - 1
    c = st.columns(4)
    c[0].metric("年化", f"{m['cagr']*100:.1f}%"); c[1].metric("夏普", f"{m['sh']:.2f}")
    c[2].metric("最大回撤", f"{m['dd']*100:.1f}%"); c[3].metric("月胜率", f"{(mret>0).mean()*100:.0f}%")
    fig = go.Figure()
    for col in cols:
        nav = (1 + df[col].dropna()).cumprod(); mk = "大盘" in col
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name=col,
                                 line=dict(width=2.5 if col == primary else (1.6 if mk else 1.1),
                                           dash="dash" if mk else ("solid" if col == primary else "dot"),
                                           color="crimson" if col == primary else ("dimgray" if mk else None))))
    fig.update_layout(height=420, yaxis_type="log", legend=dict(orientation="h"), margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    try:
        attr = json.load(open(D / "barra_attr.json", encoding="utf-8"))
        bkey = primary if primary in attr else list(attr)[0]
        a = attr[bkey]
        st.subheader("🧬 Barra 风格归因")
        k = st.columns(3)
        k[0].metric("风格解释R²", f"{a['r2']*100:.0f}%"); k[1].metric("残余alpha年化", f"{a['alpha_ann']*100:+.1f}%")
        k[2].metric("alpha t值", f"{a['alpha_t']:.1f}")
        bb = a["betas"]
        figb = go.Figure(go.Bar(x=list(bb.keys()), y=list(bb.values()), text=[f"{v:+.2f}" for v in bb.values()], textposition="outside",
                                marker_color=["crimson" if abs(v) >= 0.3 else "steelblue" for v in bb.values()]))
        figb.update_layout(height=280, yaxis_title="beta", margin=dict(t=10)); st.plotly_chart(figb, use_container_width=True)
        st.info("等权策略~93%收益由风格解释(市场+强小盘SMB+偏高波), 残余alpha不显著。**小盘溢价2026已反转**, 风格转大盘龙头时这条腿会反噬。")
    except Exception:
        pass


@st.cache_data
def load_fwdpe():
    p = pd.read_parquet(DD / "fwdpe_panel.parquet"); p["date"] = pd.to_datetime(p["date"])
    b = pd.read_parquet(DD / "fwdpe_bench.parquet"); b.index = pd.to_datetime(b.index)
    return p, b


def render_fwdpe():
    st.title("💰 管理层 forward PE 低估策略 (历史回测 · 可调参)")
    st.markdown("**两种'未来预期'要分清**: 本页用 **管理层业绩预告**(公司自己在财报前的官方指引, PIT 干净、可回测) 算 forward PE; "
                "而 **分析师预测**(卖方一致预期, 未来1-2年EPS, 有乐观偏差)是另一个口径——当前快照见 Feishu《Forward PE低估股》文档。")
    if not (DD / "fwdpe_panel.parquet").exists():
        st.error("缺 fwdpe_panel.parquet, 先跑 examples/precompute_fwdpe_backtest.py"); return
    panel, bench = load_fwdpe()
    s = st.sidebar
    s.markdown("### 估值/成长筛选")
    pe_max = s.slider("forward PE 上限", 8, 60, 30, 2)
    g_min = s.slider("管理层指引增速下限 %", 0, 80, 20, 5)
    topn = s.slider("最多持股数(0=全部合格)", 0, 80, 30, 5)
    s.markdown("### 过滤")
    excl_cyc = s.checkbox("排除周期股", True)
    use_roe = s.checkbox("质量: 扣非ROE>0", True)
    fresh = s.slider("指引新鲜度(指引期末距今≤月)", 3, 24, 12, 3)
    min_mv = s.slider("市值下限(亿)", 0, 300, 50, 10) * 10000
    bname = s.selectbox("对比基准", list(bench.columns))

    f = panel[(panel.fwd_PE > 0) & (panel.fwd_PE < pe_max) & (panel.guided_g > g_min)
              & (panel.guide_age_m <= fresh) & (panel.mv > min_mv) & (panel.ret_1m.notna())]
    if excl_cyc: f = f[~f.is_cyc]
    if use_roe: f = f[f.roe > 0]
    rows = []
    for d, g in f.groupby("date"):
        sel = g.nsmallest(topn, "fwd_PE") if topn > 0 else g
        if len(sel): rows.append((d, sel.ret_1m.mean(), len(sel)))
    if len(rows) < 12:
        st.warning("合格股票太少, 放宽参数(提高PE上限/降低增速门槛/关掉质量过滤)。"); return
    r = pd.DataFrame(rows, columns=["date", "ret", "n"]).set_index("date")
    nav = (1 + r["ret"]).cumprod()
    b = bench[bname].reindex(r.index).fillna(0); bnav = (1 + b).cumprod()
    yrs = len(r) / 12
    cagr = nav.iloc[-1] ** (1 / yrs) - 1; bcagr = bnav.iloc[-1] ** (1 / yrs) - 1
    sh = r["ret"].mean() / (r["ret"].std() + 1e-12) * np.sqrt(12)
    mdd = (nav / nav.cummax() - 1).min()
    c = st.columns(5)
    c[0].metric("年化", f"{cagr*100:.1f}%"); c[1].metric("夏普(月)", f"{sh:.2f}")
    c[2].metric("最大回撤", f"{mdd*100:.1f}%"); c[3].metric(f"超额({bname})", f"{(cagr-bcagr)*100:+.1f}%")
    c[4].metric("月均持股", f"{r['n'].mean():.0f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name="策略", line=dict(width=2.5, color="crimson")))
    fig.add_trace(go.Scatter(x=bnav.index, y=bnav, name=bname, line=dict(width=1.4, dash="dash", color="gray")))
    fig.update_layout(height=380, yaxis_type="log", legend=dict(orientation="h"), margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("逐年超额")
        py = (1 + r["ret"]).groupby(r.index.year).prod() - 1
        bb = (1 + b).groupby(b.index.year).prod() - 1
        ex = (py - bb) * 100
        figy = go.Figure(go.Bar(x=ex.index.astype(str), y=ex, marker_color=["crimson" if v > 0 else "seagreen" for v in ex]))
        figy.update_layout(height=260, yaxis_title="超额%", margin=dict(t=10)); st.plotly_chart(figy, use_container_width=True)
    with col2:
        st.subheader("当前(最新月)持仓样例")
        last = f[f.date == f.date.max()]
        last = (last.nsmallest(topn, "fwd_PE") if topn > 0 else last)[["industry", "fwd_PE", "guided_g", "roe"]]
        last.columns = ["行业", "fwdPE", "指引增速%", "扣非ROE"]
        st.dataframe(last.round(1).head(15), use_container_width=True)
    st.info("诚实提示: 历史回测里'低forwardPE+高指引增速'多数参数下**跑输等权基准**(等权小盘太强), 但常**跑赢沪深300**(切基准看)。"
            "原因: 低fwdPE+高指引常是**价值陷阱**(市场不信指引/指引未兑现)。管理层指引也有乐观偏差。这是研究工具, 非荐股。")


@st.cache_data
def load_zlrate():
    z = pd.read_parquet(DD / "moneyflow_zlrate.parquet"); z.index = pd.to_datetime(z.index)
    return z


def render_hotmoney():
    st.title("🔥 热门板块 · 短期趋势 + 主力资金 (可调参)")
    st.caption("逻辑: 只看短期(30天内)——锁定**当期热门板块**(板块短期涨幅靠前), 在其中选**仍在上升趋势 或 回踩没破X日均线**、"
               "且**主力(大单+特大单)资金净流入**的股票。主力资金 vs 散户已按订单大小区分(数据2020起)。")
    need = [DD / "pullback_close.parquet", DD / "moneyflow_zlrate.parquet"]
    if not all(p.exists() for p in need):
        st.error("缺数据: 先跑 examples/precompute_pullback_data.py 和 examples/pull_moneyflow.py"); return
    close, ret, sret, roe, gm, mv, bench, sector = load_pullback()
    zl = load_zlrate()
    syms = list(close.columns)
    zl = zl.reindex(columns=syms)
    idx = close.index[close.index >= "2020-01-01"]   # 资金流2020起
    close, ret, sret, roe, mv = [x.reindex(idx) for x in (close, ret, sret, roe, mv)]
    zl = zl.reindex(idx)

    s = st.sidebar
    s.markdown("### 热门板块")
    hot_win = s.slider("板块短期涨幅窗口(日)", 10, 40, 30, 5)
    topK = s.slider("取最强前K个板块", 1, 15, 5, 1)
    s.markdown("### 趋势 / 均线")
    maw = s.slider("均线周期X(日)", 5, 60, 20, 1)
    tol = s.slider("跌破均线容忍 %(回踩不破)", 0, 8, 2, 1) / 100
    require_rising = s.checkbox("要求均线向上(趋势未坏)", True)
    s.markdown("### 主力资金")
    mf_win = s.slider("主力净流入回看(日)", 1, 20, 5, 1)
    mf_thr = s.slider("主力净流入率门槛 %", -5, 15, 0, 1) / 100
    s.markdown("### 卖出/退出方式")
    exit_mode = s.selectbox("反转退出(调仓间隔内每日检查)",
                            ["不额外退出(持到调仓)", "跌破均线退出", "主力转流出退出(跟着钱逃)", "两者任一"])
    mf_exit_win = s.slider("主力转流出判定回看(日)", 1, 10, 3, 1)
    s.markdown("### 其他")
    use_q = s.checkbox("质量: 扣非ROE>0", False)
    min_mv = s.slider("市值下限(亿)", 0, 500, 50, 10) * 10000
    rebal = s.slider("调仓间隔(日)", 1, 20, 3, 1)
    cost = s.slider("单边成本 %", 0.0, 0.3, 0.1, 0.05) / 100
    bname = s.selectbox("对比基准", list(bench.columns))
    s.markdown("---")
    if s.button("💾 保存当前配置(供助手复现)"):
        cfg = dict(page="hotmoney", hot_win=hot_win, topK=topK, maw=maw, tol=tol,
                   require_rising=bool(require_rising), mf_win=mf_win, mf_thr=mf_thr,
                   exit_mode=exit_mode, mf_exit_win=mf_exit_win, use_q=bool(use_q),
                   min_mv_yi=int(min_mv / 10000), rebal=rebal, cost=cost, bname=bname)
        (DD / "saved_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
        s.success("已保存! 把'已保存'告诉助手即可复现。")
        s.json(cfg)

    snav = (1 + sret.fillna(0)).cumprod()
    sec_mom = snav / snav.shift(hot_win) - 1
    hot_sec = sec_mom.rank(axis=1, ascending=False) <= topK          # 当期最强topK板块
    smap = sector.reindex(syms).values
    stock_hot = hot_sec.reindex(columns=smap); stock_hot.columns = syms; stock_hot = stock_hot.reindex(idx).fillna(False)
    ma = close.rolling(maw, min_periods=max(3, maw // 2)).mean()
    above = close >= ma * (1 - tol)
    rising = ma > ma.shift(5)
    trend_ok = above & (rising if require_rising else True)
    mf_ok = zl.rolling(mf_win, min_periods=1).mean() > mf_thr
    qual = (roe > 0) if use_q else pd.DataFrame(True, index=idx, columns=syms)
    elig = stock_hot & trend_ok & mf_ok & (mv > min_mv) & qual & ret.notna()

    rb = np.zeros(len(idx), bool); rb[::rebal] = True
    if exit_mode == "不额外退出(持到调仓)":
        held = elig.astype(float).where(pd.Series(rb, index=idx), other=np.nan).ffill().fillna(0.0)
    else:
        exit_ma = close < ma * (1 - tol)
        exit_mf = zl.rolling(mf_exit_win, min_periods=1).mean() < mf_thr
        em = {"跌破均线退出": exit_ma, "主力转流出退出(跟着钱逃)": exit_mf, "两者任一": (exit_ma | exit_mf)}[exit_mode]
        ev = elig.reindex(idx).fillna(False).values
        exv = em.reindex(idx).fillna(False).values
        hv = np.zeros((len(idx), len(syms)))
        cur = np.zeros(len(syms), bool)
        for i in range(len(idx)):
            cur = ev[i].copy() if rb[i] else (cur & (~exv[i]))
            hv[i] = cur
        held = pd.DataFrame(hv, index=idx, columns=syms)
    w = held.div(held.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    rfwd = ret.shift(-1).clip(-0.21, 0.21)
    port = (w * rfwd).sum(axis=1) - (w - w.shift(1)).abs().sum(axis=1) * cost
    nhold = (w > 0).sum(axis=1)

    p = port; b = bench[bname].reindex(idx)
    sp = stat(p); sb = stat(b)
    if sp["nav"].empty or nhold.mean() < 0.5:
        st.warning("当前参数选不到股票, 放宽: 增大topK板块 / 提高跌破容忍 / 降低主力门槛。"); return
    c = st.columns(5)
    c[0].metric("年化", f"{sp['cagr']*100:.1f}%"); c[1].metric("夏普", f"{sp['sh']:.2f}")
    c[2].metric("最大回撤", f"{sp['dd']*100:.1f}%")
    c[3].metric(f"超额({bname})", f"{(sp['cagr']-sb['cagr'])*100:+.1f}%")
    c[4].metric("平均持股", f"{nhold.mean():.0f}")
    if nhold.mean() < 5:
        st.error(f"⚠️ 平均持股仅 {nhold.mean():.1f} 只——**结果由极少数个股运气主导, 不可信(看着很高也是假象)**。请放宽参数(增大topK板块/提高跌破容忍/降低主力门槛/降市值下限)以增加持股数。")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sp["nav"].index, y=sp["nav"], name="策略", line=dict(width=2.5, color="crimson")))
    fig.add_trace(go.Scatter(x=sb["nav"].index, y=sb["nav"], name=bname, line=dict(width=1.4, dash="dash", color="gray")))
    fig.update_layout(height=380, yaxis_type="log", legend=dict(orientation="h"), margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("逐年超额")
        py = (1 + p.fillna(0)).resample("Y").prod() - 1; by = (1 + b.fillna(0)).resample("Y").prod() - 1
        ex = (py - by) * 100; ex.index = ex.index.year
        figy = go.Figure(go.Bar(x=ex.index.astype(str), y=ex, marker_color=["crimson" if v > 0 else "seagreen" for v in ex]))
        figy.update_layout(height=260, yaxis_title="超额%", margin=dict(t=10)); st.plotly_chart(figy, use_container_width=True)
    with col2:
        st.subheader("每日持股数")
        figh = go.Figure(go.Scatter(x=idx, y=nhold, fill="tozeroy", line=dict(color="darkorange")))
        figh.update_layout(height=260, margin=dict(t=10)); st.plotly_chart(figh, use_container_width=True)
    st.info("说明: 主力资金=按订单大小的特大单+大单净流入率(占成交额比), 与散户(中小单)区分。"
            "这是短线趋势+资金跟随策略——在热门板块里跟着大资金、守住均线。注意短线策略换手高, 成本敏感; 趋势/资金信号反转快, 会有whipsaw。")


if page.startswith("🔥"):
    render_hotmoney()
elif page.startswith("🎯"):
    render_pullback()
elif page.startswith("💰"):
    render_fwdpe()
else:
    render_equalweight()
