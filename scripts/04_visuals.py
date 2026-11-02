# AI生成
# -*- coding: utf-8 -*-
"""04_visuals.py —— 数据可视化（全 Python：Matplotlib/Seaborn，150dpi，中文字体）
模板复现 10 种 + 业务模型图 6 张 + 预测诊断图，统一命名 fig01_*.png …"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 150
sns.set_style("whitegrid", {"font.sans-serif": ["Microsoft YaHei", "SimHei"]})

ROOT = Path(__file__).resolve().parents[1]
CL = ROOT / "data" / "clean"
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
C = {"主": "#2E5BBA", "辅": "#E8743B", "绿": "#3CA775", "灰": "#8C8C8C",
     "紫": "#7B61FF", "金": "#D4A017", "红": "#C0392B", "青": "#17A2B8"}


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  保存 {name}")


def load():
    wide = pd.read_csv(CL / "wdi_wide_clean.csv")
    ts = pd.read_csv(CL / "wdi_timeseries_clean.csv")
    return wide, ts


# ===== 模板复现 =====
def fig01(wide, ts):
    d = ts[(ts["指标"] == "GDP增长率(%)")].pivot_table(index="年份", columns="对象", values="数值")
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, c in [("中国", C["主"]), ("世界", C["辅"])]:
        ax.plot(d.index, d[col], marker="o", lw=2, color=c, label=col)
    ax.set_title("图1 中国与世界GDP增长率趋势（2010-2024）", fontsize=13, weight="bold")
    ax.set_xlabel("年份"); ax.set_ylabel("GDP增长率(%)"); ax.legend()
    save(fig, "fig01_增长率趋势折线图.png")


def fig02(wide, ts):
    w = wide[wide["年份"] == 2024].sort_values("GDP总量(万亿美元)", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(w["国家"], w["GDP总量(万亿美元)"], color=C["主"])
    ax.set_title("图2 2024年GDP总量Top15（柱状图）", fontsize=13, weight="bold")
    ax.set_ylabel("GDP（万亿美元）"); plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, "fig02_GDP柱状图.png")


def fig03(wide, ts):
    w = wide[wide["年份"] == 2024].sort_values("人均GDP(美元)", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(w["国家"], w["人均GDP(美元)"], color=C["绿"])
    ax.set_title("图3 2024年人均GDP Top15（条形图）", fontsize=13, weight="bold")
    ax.set_xlabel("人均GDP（美元）")
    save(fig, "fig03_人均GDP条形图.png")


def fig04(wide, ts):
    w = wide[["国家代码", "国家", "年份", "GDP总量(美元)"]].copy()
    top = w[w["年份"] == 2024].nlargest(10, "GDP总量(美元)")["国家代码"]
    pivot = w[w["国家代码"].isin(top)].pivot_table(index="年份", columns="国家", values="GDP总量(美元)")
    share = pivot.div(pivot.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    share.plot.area(ax=ax, alpha=0.75, colormap="tab20")
    ax.set_title("图4 Top10经济体GDP结构占比变化（堆叠面积图）", fontsize=13, weight="bold")
    ax.set_ylabel("占比(%)"); ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    save(fig, "fig04_GDP结构堆叠面积图.png")


def fig05(wide, ts):
    w = wide[wide["年份"] == 2024].copy()
    region = {"亚洲": ["China", "Japan", "India", "Korea, Rep.", "Indonesia", "Thailand", "Malaysia", "Philippines", "Viet Nam", "Singapore", "Saudi Arabia", "Turkiye", "Israel"],
              "欧洲": ["Germany", "United Kingdom", "France", "Italy", "Spain", "Netherlands", "Switzerland", "Poland", "Sweden", "Belgium", "Ireland", "Norway", "Czechia", "Portugal", "Greece", "Finland"],
              "北美": ["United States", "Canada", "Mexico"], "南美": ["Brazil", "Argentina", "Colombia", "Chile", "Peru"],
              "大洋洲": ["Australia", "New Zealand"], "非洲": ["South Africa", "Nigeria", "Egypt, Arab Rep."]}
    g = []
    for r, cs in region.items():
        v = w[w["国家"].isin(cs)]["GDP总量(美元)"].sum()
        g.append((r, v))
    g = pd.DataFrame(g, columns=["区域", "GDP"]).sort_values("GDP", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(g["GDP"], labels=g["区域"], autopct="%1.1f%%", startangle=90,
           colors=sns.color_palette("Set2"), wedgeprops=dict(width=0.45))
    ax.set_title("图5 2024年GDP区域结构（环形饼图）", fontsize=13, weight="bold")
    save(fig, "fig05_GDP区域环形饼图.png")


def fig06(wide, ts):
    top = wide[wide["年份"] == 2024].nlargest(20, "GDP总量(美元)")["国家代码"]
    p = wide[wide["国家代码"].isin(top)].pivot_table(index="国家", columns="年份", values="GDP增长率(%)")
    p = p.reindex(wide[wide["国家代码"].isin(top)].drop_duplicates("国家").nlargest(20, "GDP总量(美元)")["国家"])
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(p, cmap="RdYlGn", center=0, annot=False, ax=ax, cbar_kws={"label": "GDP增长率(%)"})
    ax.set_title("图6 Top20经济体GDP增长率热力图（2010-2024）", fontsize=13, weight="bold")
    plt.setp(ax.get_yticklabels(), fontsize=8)
    save(fig, "fig06_增长率热力图.png")


def fig07(wide, ts):
    w = wide[wide["年份"] == 2024].dropna(subset=["人均GDP(美元)", "互联网普及率(%)"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(w["人均GDP(美元)"], w["互联网普及率(%)"], s=60, c=C["紫"], alpha=0.7, edgecolor="white")
    ax.set_title("图7 人均GDP与互联网普及率散点图（2024）", fontsize=13, weight="bold")
    ax.set_xlabel("人均GDP（美元）"); ax.set_ylabel("互联网普及率(%)")
    save(fig, "fig07_人均GDP_互联网散点图.png")


def fig08(wide, ts):
    chn = ts[(ts["对象"] == "中国") & (ts["指标"] == "GDP总量(万亿美元)")].sort_values("年份")
    chn = chn[chn["年份"] >= 2010]
    gr = ts[(ts["对象"] == "中国") & (ts["指标"] == "GDP增长率(%)")].sort_values("年份")
    gr = gr[gr["年份"] >= 2010]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(chn["年份"], chn["数值"], color=C["主"], alpha=0.7, label="GDP总量")
    ax1.set_ylabel("GDP（万亿美元）", color=C["主"])
    ax2 = ax1.twinx()
    ax2.plot(gr["年份"], gr["数值"], color=C["红"], marker="o", lw=2, label="增长率")
    ax2.set_ylabel("GDP增长率(%)", color=C["红"])
    ax1.set_title("图8 中国GDP总量与增长率（双轴组合图）", fontsize=13, weight="bold")
    save(fig, "fig08_中国GDP双轴组合图.png")


def fig09(wide, ts):
    countries = ["China", "United States", "Japan", "Germany", "India"]
    feats = ["人均GDP(美元)", "GDP增长率(%)", "出口占GDP(%)", "互联网普及率(%)", "外商直接投资占GDP(%)"]
    w = wide[wide["年份"] == 2024].set_index("国家")
    vals = []
    for c in countries:
        row = w.loc[c, feats].astype(float)
        vals.append((row - row.mean()) / (row.std() if row.std() > 0 else 1))
    angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, v in enumerate(vals):
        vv = v.values.tolist() + [v.values[0]]
        ang = angles + [angles[0]]
        ax.plot(ang, vv, lw=2, label=countries[i])
        ax.fill(ang, vv, alpha=0.1)
    ax.set_xticks(angles); ax.set_xticklabels(feats, fontsize=9)
    ax.set_title("图9 五国宏观指标雷达图（标准化）", fontsize=13, weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    save(fig, "fig09_五国指标雷达图.png")


def fig10(wide, ts):
    chn = ts[(ts["对象"] == "中国") & (ts["指标"] == "GDP总量(万亿美元)")].sort_values("年份")
    chn = chn[chn["年份"] >= 2015].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    base = 0
    for i, row in chn.iterrows():
        if i == 0:
            ax.bar(i, row["数值"], color=C["灰"]); base = row["数值"]
        else:
            delta = row["数值"] - chn.loc[i - 1, "数值"]
            ax.bar(i, delta, bottom=base, color=C["绿"] if delta > 0 else C["红"])
            base += delta
    ax.set_xticks(range(len(chn))); ax.set_xticklabels(chn["年份"])
    ax.set_title("图10 中国GDP逐年增量瀑布图（2015-2024）", fontsize=13, weight="bold")
    ax.set_ylabel("GDP（万亿美元）")
    save(fig, "fig10_中国GDP瀑布图.png")


# ===== 业务模型图 =====
def fig11(wide, ts):
    km = pd.read_csv(TBL / "m1_kmeans_clusters.csv")
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = [C["主"], C["辅"], C["绿"], C["紫"], C["金"]]
    for i, grp in km.groupby("聚类标签"):
        ax.scatter(grp["人均GDP(美元)"], grp["GDP增长率(%)"], s=120,
                   c=palette[int(i) % len(palette)], alpha=0.75, edgecolor="white",
                   label=f"第{int(i)+1}群（{grp.iloc[0]['群体名称']}）")
    for _, r in km.iterrows():
        ax.annotate(r["国家代码"], (r["人均GDP(美元)"], r["GDP增长率(%)"]), fontsize=7, alpha=0.8)
    ax.set_title("图11 K-Means市场分群散点图（M1）", fontsize=13, weight="bold")
    ax.set_xlabel("人均GDP（美元）"); ax.set_ylabel("GDP增长率(%)")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "fig11_KMeans聚类散点图.png")


def fig12(wide, ts):
    feats = ["人均GDP(美元)", "GDP增长率(%)", "出口占GDP(%)", "互联网普及率(%)", "CPI通胀率(%)"]
    latest = wide.sort_values("年份").groupby("国家代码").tail(1).set_index("国家代码")
    X = latest[feats].fillna(latest[feats].median())
    X["人均GDP(美元)"] = np.log1p(X["人均GDP(美元)"])
    Xs = StandardScaler().fit_transform(X)
    ks = list(range(2, 7)); sils = []
    for k in ks:
        lab = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(Xs)
        sils.append(silhouette_score(Xs, lab))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, sils, marker="o", lw=2, color=C["主"])
    ax.axvline(ks[int(np.argmax(sils))], ls="--", color=C["红"], label=f"最佳K={ks[int(np.argmax(sils))]}")
    ax.set_title("图12 K-Means肘部法则·轮廓系数图（M1）", fontsize=13, weight="bold")
    ax.set_xlabel("簇数K"); ax.set_ylabel("轮廓系数"); ax.legend()
    save(fig, "fig12_轮廓系数图.png")


def fig13(wide, ts):
    b = pd.read_csv(TBL / "m2_boston_matrix.csv")
    qcolor = {"明星市场": C["红"], "金牛市场": C["金"], "问题市场": C["青"], "瘦狗市场": C["灰"]}
    fig, ax = plt.subplots(figsize=(9, 6))
    for q, g in b.groupby("波士顿象限"):
        ax.scatter(g["GDP份额(对数)"], g["近5年CAGR(%)"], s=g["GDP总量(万亿美元)"] * 40,
                   c=qcolor.get(q, C["灰"]), alpha=0.6, edgecolor="white", label=q)
    xm, ym = b["GDP份额(对数)"].median(), b["近5年CAGR(%)"].median()
    ax.axhline(ym, ls="--", color="grey"); ax.axvline(xm, ls="--", color="grey")
    for _, r in b.iterrows():
        ax.annotate(r["国家代码"], (r["GDP份额(对数)"], r["近5年CAGR(%)"]), fontsize=7)
    ax.set_title("图13 波士顿矩阵四象限气泡图（M2）", fontsize=13, weight="bold")
    ax.set_xlabel("GDP相对份额（对数）"); ax.set_ylabel("近5年CAGR(%)"); ax.legend(fontsize=8)
    save(fig, "fig13_波士顿矩阵气泡图.png")


def fig14(wide, ts):
    p = pd.read_csv(TBL / "m3_pareto_abc.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(p)), p["GDP占比(%)"], color=[C["主"] if c == "A" else C["辅"] if c == "B" else C["灰"] for c in p["ABC分类"]])
    ax2 = ax.twinx()
    ax2.plot(range(len(p)), p["累计占比(%)"], color=C["红"], marker="o", ms=3)
    ax2.axhline(70, ls="--", color="grey", lw=1); ax2.axhline(90, ls="--", color="grey", lw=1)
    ax.set_xticks(range(len(p))); ax.set_xticklabels(p["国家"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("GDP占比(%)"); ax2.set_ylabel("累计占比(%)")
    ax.set_title("图14 帕累托/ABC分类图（M3，2024）", fontsize=13, weight="bold")
    save(fig, "fig14_帕累托ABC图.png")


def fig15(wide, ts):
    d = pd.read_csv(TBL / "m4_china_gdp_yoy.csv")
    d = d[d["年份"] >= 2010]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(d["年份"], d["同比增长率(%)"], color=C["主"], alpha=0.7, label="中国GDP增速")
    ax.plot(d["年份"], d["世界GDP增速(%)"], color=C["辅"], marker="s", lw=2, label="世界GDP增速")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("图15 中国与世界GDP增速对比（M4，同比分析）", fontsize=13, weight="bold")
    ax.set_ylabel("同比增长率(%)"); ax.legend()
    save(fig, "fig15_同比增速对比图.png")


def fig16(wide, ts):
    hist = ts[(ts["对象"] == "中国") & (ts["指标"] == "人均GDP(美元)")].sort_values("年份")
    hist = hist[hist["年份"] >= 1978]
    fitted = pd.read_csv(TBL / "m5_sarima_fitted.csv", index_col=0)
    fc = pd.read_csv(TBL / "m5_sarima_forecast.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist["年份"], hist["数值"], color=C["主"], lw=2, label="实际值")
    ax.plot(fitted.index.astype(int), fitted.iloc[:, 0], color=C["辅"], lw=1.5, label="拟合值", alpha=0.8)
    fy = list(range(2025, 2030))
    ax.plot(fy, fc["点预测(美元)"], color=C["红"], marker="o", lw=2, label="预测值")
    ax.fill_between(fy, fc["CI下限95%(美元)"], fc["CI上限95%(美元)"], color=C["红"], alpha=0.15, label="95%置信区间")
    ax.set_title("图16 中国人均GDP ARIMA预测与95%置信区间（M5）", fontsize=13, weight="bold")
    ax.set_xlabel("年份"); ax.set_ylabel("人均GDP（美元）"); ax.legend(fontsize=8)
    save(fig, "fig16_SARIMA预测置信区间图.png")


def fig17(wide, ts):
    bt = pd.read_csv(TBL / "m5_sarima_backtest.csv")
    bt["年份"] = bt["年份"].astype(int)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bt["年份"], bt["实际值(美元)"], color=C["主"], marker="o", lw=2, label="实际值")
    ax.plot(bt["年份"], bt["预测值(美元)"], color=C["红"], marker="s", lw=2, label="预测值")
    ax.set_title("图17 ARIMA回测：实际vs预测（2020-2024）", fontsize=13, weight="bold")
    ax.set_xlabel("年份"); ax.set_ylabel("人均GDP（美元）"); ax.legend()
    save(fig, "fig17_SARIMA回测对比图.png")


def main():
    print("=" * 60)
    print("步骤 4：数据可视化（17张图）")
    print("=" * 60)
    wide, ts = load()
    for fn in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10,
               fig11, fig12, fig13, fig14, fig15, fig16, fig17]:
        try:
            fn(wide, ts)
        except Exception as e:
            print(f"  [警告] {fn.__name__} 失败: {e}")
    n = len(list(FIG.glob("*.png")))
    print(f"共生成 {n} 张图片 → figures/")
    print("步骤 4 完成。")


if __name__ == "__main__":
    main()
