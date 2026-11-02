# AI生成
# -*- coding: utf-8 -*-
"""
01_load.py —— 数据获取（合规白名单方式：世界银行开放数据 API 官方公开接口）
数据源：World Bank WDI (World Development Indicators)
接口：https://api.worldbank.org/v2/  （官方公开 API，非爬虫，JSON 直接下载）
获取内容：
  1) 主表：45 个主要经济体 × 2010-2024 年 × 10 个核心宏观指标（宽表面板，约 675 行）
  2) 时间序列：中国 & 世界 GDP / 人均 GDP（1960-2024，用于 SARIMA 预测）
输出：data/raw/ 下的 JSON + CSV 原始文件
"""
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

API = "https://api.worldbank.org/v2"

# 45 个主要经济体（G20 + 东盟 + 新兴市场代表），ISO3 代码
COUNTRIES = [
    "USA", "CHN", "JPN", "DEU", "IND", "GBR", "FRA", "ITA", "BRA", "CAN",
    "RUS", "KOR", "AUS", "MEX", "IDN", "TUR", "SAU", "ZAF", "ARG", "ESP",
    "NLD", "CHE", "POL", "SWE", "BEL", "THA", "IRL", "ISR", "NOR", "SGP",
    "MYS", "PHL", "VNM", "EGY", "NGA", "PAK", "BAN".replace("BAN", "BGD"), "COL", "CHL", "PER",
    "CZE", "PRT", "GRC", "FIN", "NZL",
]

# 10 个核心商业宏观指标（WDI 指标代码）
INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP总量(美元)",          # GDP (current US$)
    "NY.GDP.MKTP.KD.ZG": "GDP增长率(%)",        # GDP growth (annual %)
    "NY.GDP.PCAP.CD": "人均GDP(美元)",           # GDP per capita (current US$)
    "SP.POP.TOTL": "总人口",                     # Population, total
    "FP.CPI.TOTL.ZG": "CPI通胀率(%)",            # Inflation, consumer prices
    "SL.UEM.TOTL.ZS": "失业率(%)",               # Unemployment rate
    "NE.EXP.GNFS.ZS": "出口占GDP(%)",            # Exports of goods and services (% of GDP)
    "NE.IMP.GNFS.ZS": "进口占GDP(%)",            # Imports of goods and services (% of GDP)
    "BX.KLT.DINV.WD.GD.ZS": "外商直接投资占GDP(%)",  # FDI net inflows (% of GDP)
    "IT.NET.USER.ZS": "互联网普及率(%)",          # Individuals using the Internet (%)
}

# 时间序列指标（中国 + 世界，1960-2024，用于 SARIMA 预测）
TS_INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP总量(美元)",
    "NY.GDP.MKTP.KD.ZG": "GDP增长率(%)",
    "NY.GDP.PCAP.CD": "人均GDP(美元)",
}


def fetch_json(url: str, retry: int = 3) -> dict:
    """调用世界银行官方 API 并解析 JSON。"""
    for i in range(retry):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (edu-research)"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            if i == retry - 1:
                raise
            print(f"  重试 {i+1}: {e}")
            time.sleep(2)


def fetch_indicator(indicator: str, countries: list, start: int, end: int) -> pd.DataFrame:
    """拉取单个指标的多国多年数据，返回长表 DataFrame。"""
    ccode = ";".join(countries)
    per = 20000  # 一次取完
    url = (f"{API}/country/{ccode}/indicator/{urllib.parse.quote(indicator)}"
           f"?format=json&per_page={per}&date={start}:{end}")
    data = fetch_json(url)
    rows = []
    for rec in data[1] or []:
        rows.append({
            "国家代码": rec["countryiso3code"],
            "国家": rec["country"]["value"],
            "年份": int(rec["date"]),
            "指标代码": indicator,
            "指标": INDICATORS.get(indicator, indicator),
            "数值": rec["value"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 60)
    print("步骤 1：数据获取 —— 世界银行 WDI 开放数据 API（白名单合规）")
    print("=" * 60)

    # ---------- 主表面板：45 国 × 2010-2024 × 10 指标 ----------
    panel_parts = []
    for code in INDICATORS:
        print(f"拉取指标 {code} ({INDICATORS[code]}) ...")
        df = fetch_indicator(code, COUNTRIES, 2010, 2024)
        print(f"  获得 {len(df)} 条记录")
        panel_parts.append(df)
        time.sleep(0.5)

    panel = pd.concat(panel_parts, ignore_index=True)
    # 原始长表导出（JSON + CSV）
    panel.to_json(RAW_DIR / "wdi_panel_raw.json", orient="records", force_ascii=False, indent=2)
    panel.to_csv(RAW_DIR / "wdi_panel_raw.csv", index=False, encoding="utf-8-sig")
    print(f"主表原始长表：{len(panel)} 条记录 → data/raw/wdi_panel_raw.csv")

    # ---------- 时间序列（中国 & 世界，1960-2024，用于 SARIMA） ----------
    ts_parts = []
    for code in TS_INDICATORS:
        for region, label in [("CHN", "中国"), ("WLD", "世界")]:
            print(f"拉取时间序列 {code} - {label} (1960-2024) ...")
            url = (f"{API}/country/{region}/indicator/{urllib.parse.quote(code)}"
                   f"?format=json&per_page=20000&date=1960:2024")
            data = fetch_json(url)
            for rec in data[1] or []:
                ts_parts.append({
                    "对象": label,
                    "年份": int(rec["date"]),
                    "指标": TS_INDICATORS[code],
                    "数值": rec["value"],
                })
            time.sleep(0.5)

    ts = pd.DataFrame(ts_parts)
    ts.to_json(RAW_DIR / "wdi_timeseries_raw.json", orient="records", force_ascii=False, indent=2)
    ts.to_csv(RAW_DIR / "wdi_timeseries_raw.csv", index=False, encoding="utf-8-sig")
    print(f"时间序列原始表：{len(ts)} 条记录 → data/raw/wdi_timeseries_raw.csv")

    # ---------- 获取日期与来源元信息（合规声明用） ----------
    meta = {
        "数据源名称": "世界银行 WDI（World Development Indicators，世界发展指标）",
        "URL": "https://api.worldbank.org/v2/ （数据门户 https://data.worldbank.org/）",
        "获取方式": "官方开放 API 直接下载 JSON（白名单方式：国际组织开放平台公开数据文件下载）",
        "获取日期": time.strftime("%Y-%m-%d"),
        "授权/License": "世界银行开放数据按 CC BY 4.0 开放使用（Open Data License: CC BY 4.0）",
        "国家数": len(set(panel['国家代码'])),
        "指标数": len(INDICATORS),
        "主表记录数": len(panel),
        "时间序列记录数": len(ts),
    }
    with open(RAW_DIR / "source_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("来源元信息 → data/raw/source_meta.json")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("步骤 1 完成。")


if __name__ == "__main__":
    main()
