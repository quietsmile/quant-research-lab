"""Forward PE 低估股筛选: 分析师对2026/2027一致预期(机构中位数) + 现价 → forward PE、隐含增速、PEG;
结合管理层业绩预告(forecast)交叉验证; 质量/覆盖/规模过滤。输出 markdown 供飞书。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib, sys
sys.path.insert(0, "/home/claudeuser/econ/quant-research-lab")
from quantlab.data.tushare_adapter import get_pro
D = pathlib.Path.home() / ".local/share/quantlab/fundamentals"

rc = pd.read_parquet(D / "report_rc_2026.parquet")
rc["symbol"] = rc["ts_code"].str[:6]
# 一致预期: 各机构中位数, 按预测年
def consensus(qtr):
    g = rc[rc["quarter"] == qtr].groupby("ts_code")
    return pd.DataFrame({"eps": g["eps"].median(), "np_med": g["np"].median(),
                         "n_org": g["org_name"].nunique(), "name": g["name"].first()})
c26 = consensus("2026Q4"); c27 = consensus("2027Q4")
df = c26.add_suffix("_26").join(c27.add_suffix("_27"), how="inner")
df["name"] = df["name_26"]; df["n_org"] = df[["n_org_26", "n_org_27"]].min(axis=1)
df["symbol"] = [t[:6] for t in df.index]

# 现价 + 现PE + 市值 (daily_2026 最新)
p26 = pd.read_parquet(D / "daily_2026.parquet")
last_d = p26["trade_date"].max()
cur = p26[p26.trade_date == last_d].copy(); cur["symbol"] = cur["symbol"].astype(str).str.zfill(6)
cur = cur.set_index("symbol")[["close", "pe_ttm", "total_mv"]]
df = df.merge(cur, left_on="symbol", right_index=True, how="inner")

df["fwdPE26"] = df["close"] / df["eps_26"]
df["fwdPE27"] = df["close"] / df["eps_27"]
df["cur_eps"] = df["close"] / df["pe_ttm"]
df["g_26"] = df["eps_26"] / df["cur_eps"] - 1            # 当前→2026
df["g_27"] = df["eps_27"] / df["eps_26"] - 1             # 2026→2027
df["cagr2y"] = (df["eps_27"] / df["cur_eps"]).clip(lower=0) ** 0.5 - 1
df["PEG"] = df["fwdPE26"] / (df["cagr2y"] * 100)

# 管理层业绩预告(FY2025)交叉验证: 是否也指引增长
try:
    pro = get_pro(); fc = pro.forecast_vip(period="20251231") if hasattr(pro, "forecast_vip") else pro.forecast(period="20251231")
    fc["symbol"] = fc["ts_code"].str[:6]
    fc["mgmt_g"] = fc[["p_change_min", "p_change_max"]].mean(axis=1)
    mg = fc.groupby("symbol")["mgmt_g"].first()
    df = df.merge(mg.rename("mgmt_g25"), left_on="symbol", right_index=True, how="left")
except Exception as e:
    df["mgmt_g25"] = np.nan; print("forecast merge skip:", str(e)[:50])

# 筛选: 低forward PE + 高增长(成长进低估) + 有覆盖 + 正eps + 非微盘
f = df[(df.eps_26 > 0) & (df.eps_27 > 0) & (df.cur_eps > 0)
       & (df.n_org >= 3)                        # 至少3家机构覆盖
       & (df.total_mv > 1_000_000)              # >100亿市值(total_mv单位万元)
       & (df.fwdPE26 > 3) & (df.fwdPE27 < 18)   # forward PE 合理且偏低(剔除<3异常)
       & (df.cagr2y > 0.12) & (df.cagr2y < 0.55) # 2年隐含EPS CAGR 12%~55%(剔除异常高)
       & (df.g_26 < 1.2)                        # 当年增速<120%(剔数据错)
       & (df.PEG > 0) & (df.PEG < 1.0)].copy()  # PEG<1 = 增长未被充分定价
f["score"] = f["cagr2y"] * 100 / f["fwdPE27"]    # 越高=越便宜的增长
f = f.sort_values("score", ascending=False)

# 行业
ind = pd.read_parquet(D / "industry.parquet"); ind["symbol"] = ind["symbol"].astype(str).str.zfill(6)
f = f.merge(ind.set_index("symbol")["industry"], left_on="symbol", right_index=True, how="left")

f.to_parquet(D / "forward_pe_screen.parquet")
print(f"候选数: {len(f)}  (覆盖股票池 {len(df)})  最新价日 {pd.Timestamp(last_d).date()}")
cols = ["name", "industry", "close", "pe_ttm", "fwdPE26", "fwdPE27", "cagr2y", "PEG", "g_27", "n_org", "mgmt_g25", "total_mv"]
print(f.head(25)[cols].to_string())
