# AI生成
# -*- coding: utf-8 -*-
"""
02_clean.py —— 数据清洗与治理
处理策略：
  1) 缺失值：同一国家同一指标按时间线性插值（不超过连续2年缺口），仍缺失的按国家中位数填充；
     缺失率>40%的国家整行剔除（如委内瑞拉类数据断档国家）。
  2) 异常值：对 GDP 增长率、CPI 等比率类指标做 IQR 识别（仅标记，不删除——宏观极端值真实存在）。
  3) 重复值：按 (国家代码, 年份, 指标代码) 去重。
  4) 类型统一：年份 int，数值 float，国家名/代码 str；日期列统一为 year-end。
  5) 多表对齐：长表 → 宽表主键 = (国家代码, 年份)；时间序列表主键 = (对象, 指标, 年份)。
输出：data/clean/ 下 CSV + JSON + SQLite（复用与验证）。
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

MISSING_DROP_THRESHOLD = 0.40  # 国家缺失率超过 40% 剔除


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(RAW_DIR / "wdi_panel_raw.csv")
    ts = pd.read_csv(RAW_DIR / "wdi_timeseries_raw.csv")
    return panel, ts


def clean_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {}
    n0 = len(panel)

    # 1) 去重：同一国家同一年份同一指标只保留一条
    dup_mask = panel.duplicated(subset=["国家代码", "年份", "指标代码"], keep="first")
    panel = panel[~dup_mask].copy()
    log["重复记录剔除数"] = int(dup_mask.sum())

    # 2) 类型统一
    panel["年份"] = panel["年份"].astype(int)
    panel["数值"] = pd.to_numeric(panel["数值"], errors="coerce")

    # 3) 剔除国家级缺失率过高的国家
    wide_tmp = panel.pivot_table(index=["国家代码", "国家", "年份"], columns="指标",
                                 values="数值").reset_index()
    miss_ratio = (wide_tmp.set_index(["国家代码", "国家", "年份"])
                  .isna().groupby(level="国家代码").mean().mean(axis=1))
    bad_countries = miss_ratio[miss_ratio > MISSING_DROP_THRESHOLD].index.tolist()
    log["整国剔除（缺失率>40%）"] = bad_countries
    panel = panel[~panel["国家代码"].isin(bad_countries)]

    # 4) 透视宽表：行=国家×年份，列=指标
    wide = panel.pivot_table(index=["国家代码", "国家", "年份"], columns="指标",
                             values="数值").reset_index()
    wide.columns.name = None

    # 5) 缺失值：先按国家分组对时间线性插值（限连续2年），再按国家中位数填充
    metric_cols = [c for c in wide.columns if c not in ("国家代码", "国家", "年份")]
    before_na = wide[metric_cols].isna().sum().sum()
    wide = wide.sort_values(["国家代码", "年份"])
    for col in metric_cols:
        wide[col] = wide.groupby("国家代码")[col].transform(
            lambda s: s.interpolate(method="linear", limit=2))
        wide[col] = wide.groupby("国家代码")[col].transform(
            lambda s: s.fillna(s.median()))
    wide = wide.sort_values(["国家代码", "年份"]).reset_index(drop=True)
    after_na = wide[metric_cols].isna().sum().sum()
    log["插值/填充前缺失单元数"] = int(before_na)
    log["插值/填充后缺失单元数"] = int(after_na)

    # 6) 派生指标（数据转换）
    wide["人均GDP(美元)"] = pd.to_numeric(wide["人均GDP(美元)"], errors="coerce")
    wide["GDP总量(美元)"] = pd.to_numeric(wide["GDP总量(美元)"], errors="coerce")
    wide["GDP总量(万亿美元)"] = wide["GDP总量(美元)"] / 1e12
    wide["GDP(对数)"] = np.log(wide["GDP总量(美元)"].clip(lower=1))

    # 7) 异常值识别（IQR，仅标记）：GDP增长率、CPI 通胀率
    for col in ["GDP增长率(%)", "CPI通胀率(%)"]:
        q1, q3 = wide[col].quantile(0.25), wide[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        wide[f"{col}_异常"] = (wide[col] < lo) | (wide[col] > hi)
        log[f"{col} IQR异常值数(保留仅标记)"] = int(wide[f"{col}_异常"].sum())

    log["清洗后主表行数"] = len(wide)
    log["清洗前主表长表记录数"] = n0
    log["国家数(清洗后)"] = wide["国家代码"].nunique()
    return wide, log


def clean_ts(ts: pd.DataFrame) -> pd.DataFrame:
    ts = ts.copy()
    ts["年份"] = ts["年份"].astype(int)
    ts["数值"] = pd.to_numeric(ts["数值"], errors="coerce")
    ts = ts.drop_duplicates(subset=["对象", "指标", "年份"], keep="first")
    ts = ts.sort_values(["对象", "指标", "年份"]).reset_index(drop=True)
    # GDP 用万亿美元口径
    ts.loc[ts["指标"] == "GDP总量(美元)", "数值"] = (
        ts.loc[ts["指标"] == "GDP总量(美元)", "数值"] / 1e12)
    ts["指标"] = ts["指标"].replace({"GDP总量(美元)": "GDP总量(万亿美元)"})
    return ts


def main() -> None:
    print("=" * 60)
    print("步骤 2：数据清洗与治理")
    print("=" * 60)
    panel, ts = load_raw()

    wide, log = clean_panel(panel)
    wide.to_csv(CLEAN_DIR / "wdi_wide_clean.csv", index=False, encoding="utf-8-sig")
    wide.to_json(CLEAN_DIR / "wdi_wide_clean.json", orient="records", force_ascii=False, indent=2)

    ts_c = clean_ts(ts)
    ts_c.to_csv(CLEAN_DIR / "wdi_timeseries_clean.csv", index=False, encoding="utf-8-sig")
    ts_c.to_json(CLEAN_DIR / "wdi_timeseries_clean.json", orient="records", force_ascii=False, indent=2)

    # SQLite 导出（复用与验证）
    db_path = CLEAN_DIR / "wdi_clean.db"
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        wide.to_sql("wdi_panel", conn, index=False)
        ts_c.to_sql("wdi_timeseries", conn, index=False)
    print(f"SQLite 导出 → {db_path}")

    with open(CLEAN_DIR / "clean_log.json", "w", encoding="utf-8") as f:
        import json
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print("步骤 2 完成。")


if __name__ == "__main__":
    import json  # noqa: F401
    main()
