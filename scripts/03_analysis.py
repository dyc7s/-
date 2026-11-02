# AI生成
# -*- coding: utf-8 -*-
"""
03_analysis.py —— 商业分析模型（5 种）
模型清单：
  M1 K-Means 用户（市场）分群：45 个经济体按人均GDP/增长/开放度/数字化聚类，确定 K 用肘部法+轮廓系数
  M2 波士顿矩阵（份额×增速）：GDP 相对份额（对数）× 近5年复合增长率气泡图，中位数切分四象限
  M3 帕累托/ABC 分类：2024 年各国 GDP 总量排序，70/90% 累计销售额切分 A/B/C
  M4 同比/环比分析：中国 GDP（万亿美元）年度环比、同比增长率
  M5 SARIMA 预测：中国人均 GDP（1978-2024）训练至2019回测，再全量预测2025-2029（点预测+95%CI）
参数依据：
  - K-Means 特征标准化（z-score）：量纲差异大；K 由轮廓系数最大值确定（业务上 3~6 群可解释）
  - 波士顿矩阵切分线用中位数：样本量 45，中位数对极端值稳健
  - 帕累托 ABC 用 70%/90% 累计贡献线：教科书口径
  - SARIMA 阶数用 AIC 网格搜索定阶；回测 2020-2024 计算MAPE/RMSE
输出：outputs/tables/ 下各模型结果表（CSV）
"""
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
OUT_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42  # 抽样/聚类随机种子，全项目统一，保证可复现


def model1_kmeans(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """M1 K-Means：以 2024 年（缺失则最近年份）截面数据对 45 国聚类。"""
    feats_src = ["人均GDP(美元)", "GDP增长率(%)", "出口占GDP(%)", "互联网普及率(%)", "CPI通胀率(%)"]
    latest = (wide.sort_values("年份").groupby("国家代码").tail(1)
              .set_index("国家代码"))
    X = latest[feats_src].copy()
    X = X.fillna(X.median())
    # 人均GDP 右偏严重 → 取对数后再标准化
    X["人均GDP(美元)"] = np.log1p(X["人均GDP(美元)"])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 肘部法 + 轮廓系数选择 K（业务可解释范围 2~6）
    sil = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        labels = km.fit_predict(Xs)
        sil[k] = float(silhouette_score(Xs, labels))
    best_k = max(sil, key=sil.get)

    km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=20)
    latest["聚类标签"] = km.fit_predict(Xs)

    # 群体画像：各群规模与特征均值（还原对数）
    profile = latest.groupby("聚类标签").agg(
        国家数=("国家", "count"),
        人均GDP均值美元=("人均GDP(美元)", "median"),
        GDP增长率均值=("GDP增长率(%)", "median"),
        互联网普及率均值=("互联网普及率(%)", "median"),
    ).reset_index()
    profile["人均GDP均值美元"] = np.expm1(profile["人均GDP均值美元"])

    # 命名：按人均GDP排序给出业务标签
    profile = profile.sort_values("人均GDP均值美元").reset_index(drop=True)
    names = {i: f"第{i+1}群" for i in range(len(profile))}
    profile["群体名称"] = profile.index.map(names)
    latest = latest.reset_index().merge(
        profile[["聚类标签", "群体名称"]], on="聚类标签", how="left")

    result = latest[["国家代码", "国家", "聚类标签", "群体名称"] + feats_src].copy()
    result.to_csv(OUT_DIR / "m1_kmeans_clusters.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(OUT_DIR / "m1_kmeans_profile.csv", index=False, encoding="utf-8-sig")

    info = {"最佳K": int(best_k), "轮廓系数": {str(k): round(v, 4) for k, v in sil.items()},
            "群体画像": profile.to_dict(orient="records")}
    return result, info


def model2_boston(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """M2 波士顿矩阵：X=GDP相对份额(对数)，Y=近5年CAGR，气泡=2024年GDP。"""
    w24 = wide[wide["年份"] == 2024].copy()
    share = w24["GDP总量(美元)"] / w24["GDP总量(美元)"].sum()
    w24["GDP份额(%)"] = share * 100
    w24["GDP份额(对数)"] = np.log10(w24["GDP份额(%)"].clip(lower=1e-3))

    # 近5年复合增长率：2019→2024
    w19 = wide[wide["年份"] == 2019].set_index("国家代码")["GDP总量(美元)"]
    w24 = w24.set_index("国家代码")
    w24["2019年GDP"] = w19
    w24["近5年CAGR(%)"] = ((w24["GDP总量(美元)"] / w24["2019年GDP"]) ** (1 / 5) - 1) * 100
    w24 = w24.reset_index()

    x_med = w24["GDP份额(对数)"].median()
    y_med = w24["近5年CAGR(%)"].median()

    def quadrant(row):
        big = row["GDP份额(对数)"] >= x_med
        grow = row["近5年CAGR(%)"] >= y_med
        if big and grow:
            return "明星市场"
        if big and not grow:
            return "金牛市场"
        if not big and grow:
            return "问题市场"
        return "瘦狗市场"

    w24["波士顿象限"] = w24.apply(quadrant, axis=1)
    out = w24[["国家", "国家代码", "GDP份额(%)", "GDP份额(对数)", "近5年CAGR(%)",
               "GDP总量(万亿美元)", "波士顿象限"]]
    out.to_csv(OUT_DIR / "m2_boston_matrix.csv", index=False, encoding="utf-8-sig")
    info = {"切分线X(份额对数中位数)": round(float(x_med), 4),
            "切分线Y(CAGR中位数%)": round(float(y_med), 4),
            "象限分布": out["波士顿象限"].value_counts().to_dict()}
    return out, info


def model3_pareto(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """M3 帕累托/ABC：2024 年各国 GDP 降序累计贡献。"""
    w24 = wide[wide["年份"] == 2024].sort_values("GDP总量(美元)", ascending=False).copy()
    total = w24["GDP总量(美元)"].sum()
    w24["GDP占比(%)"] = w24["GDP总量(美元)"] / total * 100
    w24["累计占比(%)"] = w24["GDP占比(%)"].cumsum()
    w24["ABC分类"] = np.where(w24["累计占比(%)"] <= 70, "A",
                        np.where(w24["累计占比(%)"] <= 90, "B", "C"))
    # 修正：恰好越过70%线的国家属于A（含该行），保持教科书口径
    w24["ABC分类"] = np.where(w24["累计占比(%)"].shift(1, fill_value=0) <= 70, "A",
                        np.where(w24["累计占比(%)"].shift(1, fill_value=0) <= 90, "B", "C"))
    out = w24[["国家", "GDP总量(万亿美元)", "GDP占比(%)", "累计占比(%)", "ABC分类"]]
    out.to_csv(OUT_DIR / "m3_pareto_abc.csv", index=False, encoding="utf-8-sig")
    n_a = int((out["ABC分类"] == "A").sum())
    info = {"A类国家数": n_a, "A类贡献(%)": round(float(out.loc[out['ABC分类'] == 'A', 'GDP占比(%)'].sum()), 2),
            "B类国家数": int((out["ABC分类"] == "B").sum()),
            "C类国家数": int((out["ABC分类"] == "C").sum())}
    return out, info


def model4_yoy(ts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """M4 同比/环比：中国 GDP 总量（万亿美元）年度环比增量与同比增长率（2010-2024）。"""
    chn = ts[(ts["对象"] == "中国") & (ts["指标"] == "GDP总量(万亿美元)")].copy()
    chn = chn.sort_values("年份").reset_index(drop=True)
    chn["环比增量"] = chn["数值"].diff()
    chn["同比增长率(%)"] = chn["数值"].pct_change() * 100
    # 世界同期对比
    wld = ts[(ts["对象"] == "世界") & (ts["指标"] == "GDP总量(万亿美元)")].copy()
    wld = wld.sort_values("年份").reset_index(drop=True)
    wld["同比增长率(%)"] = wld["数值"].pct_change() * 100
    merged = chn[["年份", "数值", "环比增量", "同比增长率(%)"]].rename(
        columns={"数值": "中国GDP(万亿美元)"})
    merged["世界GDP增速(%)"] = wld.set_index("年份")["同比增长率(%)"].reindex(
        merged["年份"]).values
    merged = merged[merged["年份"] >= 2000]
    merged.to_csv(OUT_DIR / "m4_china_gdp_yoy.csv", index=False, encoding="utf-8-sig")
    info = {"中国近10年均增速(%)": round(float(chn.set_index('年份')['同比增长率(%)'].loc[2015:2024].mean()), 2),
            "世界近10年均增速(%)": round(float(wld.set_index('年份')['同比增长率(%)'].loc[2015:2024].mean()), 2)}
    return merged, info


def model5_sarima(ts: pd.DataFrame) -> tuple[dict, dict]:
    """M5 SARIMA：中国人均GDP（美元，1978-2024），先回测再预测。"""
    chn = ts[(ts["对象"] == "中国") & (ts["指标"] == "人均GDP(美元)")].copy()
    chn = chn.sort_values("年份").reset_index(drop=True)
    chn = chn[chn["年份"] >= 1978].dropna(subset=["数值"])
    series_full = pd.Series(chn["数值"].values,
                            index=pd.PeriodIndex(chn["年份"], freq="Y"))

    # ---- 阶数搜索：AIC 最小 ----
    # 年度频率数据无季节周期（s 必须为无），SARIMA 的季节项退化为 ARIMA(p,d,q)
    def fit_best(s: pd.Series):
        best, best_aic = None, np.inf
        for p, d, q in itertools.product(range(4), range(2), range(4)):
            try:
                m = SARIMAX(s.astype(float), order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)
                if m.aic < best_aic:
                    best, best_aic = m, m.aic
            except Exception:
                continue
        return best

    # ---- 回测：训练 1978-2019，测试 2020-2024 ----
    train = series_full.loc[: "2019"]
    test = series_full.loc["2020":]
    m_bt = fit_best(train)
    n_test = len(test)
    pred_bt = m_bt.get_forecast(steps=n_test)
    pred_mean = pred_bt.predicted_mean.values
    actual = test.values
    mape = float(np.mean(np.abs((actual - pred_mean) / actual)) * 100)
    rmse = float(np.sqrt(np.mean((actual - pred_mean) ** 2)))

    # ---- 全量拟合 + 预测 2025-2029（95% CI）----
    m_full = fit_best(series_full)
    fc = m_full.get_forecast(steps=5)
    fc_mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    forecast_df = pd.DataFrame({
        "年份": [str(i) for i in range(2025, 2030)],
        "点预测(美元)": fc_mean.values,
        "CI下限95%(美元)": ci.iloc[:, 0].values,
        "CI上限95%(美元)": ci.iloc[:, 1].values,
    })
    forecast_df.to_csv(OUT_DIR / "m5_sarima_forecast.csv", index=False, encoding="utf-8-sig")

    # 回测明细表
    bt_df = pd.DataFrame({"年份": [str(i) for i in range(2020, 2025)],
                          "实际值(美元)": actual, "预测值(美元)": pred_mean})
    bt_df.to_csv(OUT_DIR / "m5_sarima_backtest.csv", index=False, encoding="utf-8-sig")

    # 拟合历史（供画图：一步向前拟合值）
    fitted = pd.Series(m_full.fittedvalues.values, index=series_full.index.astype(str))
    fitted.to_csv(OUT_DIR / "m5_sarima_fitted.csv", encoding="utf-8-sig")

    info = {"模型阶数(全量)": f"SARIMA{m_full.model_orders}",
            "回测期": "2020-2024", "MAPE(%)": round(mape, 2), "RMSE(美元)": round(rmse, 0),
            "2029年点预测(美元)": round(float(fc_mean.values[-1]), 0)}
    return {"forecast": forecast_df, "backtest": bt_df,
            "history": pd.DataFrame({"年份": chn["年份"], "实际值": chn["数值"]}),
            "fitted": fitted}, info


def main() -> None:
    print("=" * 60)
    print("步骤 3：商业分析模型（K-Means / 波士顿 / 帕累托 / 同比环比 / SARIMA）")
    print("=" * 60)
    wide = pd.read_csv(CLEAN_DIR / "wdi_wide_clean.csv")
    ts = pd.read_csv(CLEAN_DIR / "wdi_timeseries_clean.csv")

    results = {}
    r1, i1 = model1_kmeans(wide); results["M1_KMeans"] = i1
    print(f"M1 K-Means 完成: 最佳K={i1['最佳K']}")
    r2, i2 = model2_boston(wide); results["M2_波士顿矩阵"] = i2
    print(f"M2 波士顿矩阵完成: {i2['象限分布']}")
    r3, i3 = model3_pareto(wide); results["M3_帕累托ABC"] = i3
    print(f"M3 帕累托ABC完成: A类{i3['A类国家数']}国贡献{i3['A类贡献(%)']}%")
    r4, i4 = model4_yoy(ts); results["M4_同比环比"] = i4
    print(f"M4 同比环比完成: {i4}")
    r5, i5 = model5_sarima(ts); results["M5_SARIMA"] = i5
    print(f"M5 SARIMA完成: {i5}")

    with open(OUT_DIR / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print("模型汇总 → outputs/tables/model_summary.json")
    print("步骤 3 完成。")


if __name__ == "__main__":
    main()
