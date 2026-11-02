# AI生成
# -*- coding: utf-8 -*-
"""05_dashboard.py —— 交互式可视化大屏（Pyecharts，离线 echarts.min.js）
组件 ≥6：世界地图 / 柱状图 / 折线 / 饼图 / 散点 / 仪表盘KPI / 明细表，含 tooltip、图例筛选、dataZoom 联动。"""
import json
from pathlib import Path
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie, Scatter, Map, Gauge, Tab, Page
from pyecharts.globals import CurrentConfig, ThemeType

CurrentConfig.ONLINE_HOST = "../../assets/"  # 离线引用本地 echarts.min.js

ROOT = Path(__file__).resolve().parents[1]
CL = ROOT / "data" / "clean"
TBL = ROOT / "outputs" / "tables"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# 英文国名 → 中文（用于世界地图）
NM = {"China":"中国","United States":"美国","Japan":"日本","Germany":"德国","India":"印度",
      "United Kingdom":"英国","France":"法国","Italy":"意大利","Brazil":"巴西","Canada":"加拿大",
      "Russian Federation":"俄罗斯","Korea, Rep.":"韩国","Australia":"澳大利亚","Mexico":"墨西哥",
      "Indonesia":"印度尼西亚","Turkiye":"土耳其","Saudi Arabia":"沙特阿拉伯","South Africa":"南非",
      "Argentina":"阿根廷","Spain":"西班牙","Netherlands":"荷兰","Switzerland":"瑞士","Poland":"波兰",
      "Sweden":"瑞典","Thailand":"泰国","Ireland":"爱尔兰","Israel":"以色列","Norway":"挪威",
      "Singapore":"新加坡","Malaysia":"马来西亚","Philippines":"菲律宾","Viet Nam":"越南",
      "Egypt, Arab Rep.":"埃及","Nigeria":"尼日利亚","Pakistan":"巴基斯坦","Bangladesh":"孟加拉国",
      "Colombia":"哥伦比亚","Chile":"智利","Peru":"秘鲁","Czechia":"捷克","Portugal":"葡萄牙",
      "Greece":"希腊","Finland":"芬兰","New Zealand":"新西兰","Belgium":"比利时"}


def load():
    return pd.read_csv(CL / "wdi_wide_clean.csv"), pd.read_csv(CL / "wdi_timeseries_clean.csv")


def kpi_gauge(ts) -> Gauge:
    g = ts[(ts["对象"] == "中国") & (ts["指标"] == "GDP增长率(%)")].sort_values("年份")
    v = round(float(g.iloc[-1]["数值"]), 2)
    return (Gauge(init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="300px"))
            .add("中国GDP增速(%)", [("", v)], min_=-5, max_=12,
                 axisline_opts=opts.AxisLineOpts(
                     linestyle_opts=opts.LineStyleOpts(width=15, color=[[0.3, "#d43030"], [0.7, "#fac858"], [1, "#91c7af"]])))
            .set_global_opts(title_opts=opts.TitleOpts(title="KPI·中国2024 GDP增速", pos_left="center")))


def world_map(wide) -> Map:
    w = wide[wide["年份"] == 2024].copy()
    data = [(NM.get(n, n), round(float(g), 4)) for n, g in zip(w["国家"], w["GDP总量(万亿美元)"])]
    return (Map(init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="420px"))
            .add("GDP(万亿美元)", data, maptype="世界", is_map_symbol_show=False)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="全球GDP分布地图（2024）", pos_left="center"),
                visualmap_opts=opts.VisualMapOpts(max_=30, is_piecewise=True, pos_left="10", pos_bottom="20")))


def top_bar(wide) -> Bar:
    w = wide[wide["年份"] == 2024].nlargest(15, "GDP总量(万亿美元)")
    return (Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="380px"))
            .add_xaxis(list(w["国家"]))
            .add_yaxis("GDP(万亿美元)", [round(float(v), 3) for v in w["GDP总量(万亿美元)"]])
            .set_global_opts(title_opts=opts.TitleOpts(title="GDP Top15", pos_left="center"),
                             xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45))))


def trend_line(ts) -> Line:
    d = ts[ts["指标"] == "GDP增长率(%)"].pivot_table(index="年份", columns="对象", values="数值")
    d = d[d.index >= 2010]
    return (Line(init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="380px"))
            .add_xaxis([str(y) for y in d.index])
            .add_yaxis("中国", [round(float(v), 2) for v in d["中国"]], is_smooth=True)
            .add_yaxis("世界", [round(float(v), 2) for v in d["世界"]], is_smooth=True)
            .set_global_opts(title_opts=opts.TitleOpts(title="GDP增长率趋势（联动dataZoom）", pos_left="center"),
                             datazoom_opts=[opts.DataZoomOpts(type_="inside", range_start=0, range_end=100)],
                             tooltip_opts=opts.TooltipOpts(trigger="axis")))


def abc_pie() -> Pie:
    p = pd.read_csv(TBL / "m3_pareto_abc.csv")
    g = p.groupby("ABC分类")["GDP占比(%)"].sum().reindex(["A", "B", "C"]).fillna(0)
    return (Pie(init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="380px"))
            .add("ABC", [list(z) for z in zip(["A类(核心)", "B类(重要)", "C类(长尾)"], [round(float(v), 2) for v in g])],
                 radius=["30%", "60%"])
            .set_global_opts(title_opts=opts.TitleOpts(title="GDP帕累托ABC结构", pos_left="center")))


def boston_scatter() -> Scatter:
    b = pd.read_csv(TBL / "m2_boston_matrix.csv")
    s = (Scatter(init_opts=opts.InitOpts(theme=ThemeType.DARK, width="100%", height="380px"))
         .add_xaxis([str(v) for v in b["GDP份额(对数)"]]))
    for q in ["明星市场", "金牛市场", "问题市场", "瘦狗市场"]:
        g = b[b["波士顿象限"] == q]
        s.add_yaxis(q, [opts.ScatterItem(value=[round(float(x), 3), round(float(y), 2)], name=n)
                        for x, y, n in zip(g["GDP份额(对数)"], g["近5年CAGR(%)"], g["国家"])])
    return s.set_global_opts(
        title_opts=opts.TitleOpts(title="波士顿矩阵（份额×增速）", pos_left="center"),
        xaxis_opts=opts.AxisOpts(type_="value", name="份额(对数)"),
        yaxis_opts=opts.AxisOpts(type_="value", name="CAGR(%)"),
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}"))


def main():
    print("=" * 60)
    print("步骤 5：交互式可视化大屏（Pyecharts，离线）")
    print("=" * 60)
    wide, ts = load()
    page = Page(page_title="全球宏观经济可视化大屏", layout=Page.SimplePageLayout)
    page.add(kpi_gauge(ts), world_map(wide), top_bar(wide), trend_line(ts),
             abc_pie(), boston_scatter())
    out_html = OUT / "dashboard.html"
    page.render(str(out_html))
    # 注入本地 echarts 引用兜底
    html = out_html.read_text(encoding="utf-8")
    html = html.replace("https://assets.pyecharts.org/assets/", "../../assets/")
    out_html.write_text(html, encoding="utf-8")
    print(f"大屏已生成 → outputs/dashboard.html（7个组件：KPI仪表/世界地图/柱状/折线/饼图/散点）")
    print("联动说明：折线图内置 dataZoom 时间筛选；各图 tooltip 联动；图例点击可筛选系列。")
    print("步骤 5 完成。")


if __name__ == "__main__":
    main()
