import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, PathPatch, Wedge
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.interpolate import CubicSpline

# ---------- 全局设置 ----------
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# 统一配色
BLUE = "#2E75B6"
ORANGE = "#ED7D31"
GRAY = "#BFBFBF"
GREEN = "#70AD47"
RED = "#C00000"
PALETTE = ["#2E75B6", "#ED7D31", "#70AD47", "#C00000", "#7030A0", "#FFC000", "#5B9BD5", "#A5A5A5"]


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("已生成", path)


# 通用渐变柱体绘制（每根柱子垂直渐变）
def gradient_bars(ax, x, height, cmap, width=0.6, labels=None):
    for xi, h in zip(x, height):
        grad = np.linspace(0.25, 1.0, 256).reshape(-1, 1)
        ax.imshow(grad, extent=[xi - width / 2, xi + width / 2, 0, h],
                  aspect="auto", cmap=cmap, zorder=2, origin="lower")
    ax.set_xlim(min(x) - 1, max(x) + 1)


# ============================================================
#  1 渐变柱形图
# ============================================================
def chart01():
    regions = ["华北", "华南", "东北", "西北", "西南", "华东"]
    sales = [2354, 1902, 3524, 2698, 2896, 2563]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(regions))
    cmap = LinearSegmentedColormap.from_list("g1", [BLUE, "#B4D7F7"])
    gradient_bars(ax, x, sales, cmap, labels=regions)
    ax.set_xticks(x); ax.set_xticklabels(regions)
    ax.set_ylim(0, 4200)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("渐变柱形图", fontsize=14, fontweight="bold")
    save(fig, "01_渐变柱形图.png")


# ============================================================
#  2 带均值柱形图
# ============================================================
def chart02():
    regions = ["华北", "华南", "东北", "西北", "西南", "华东"]
    sales = [2354, 1902, 3524, 2698, 2896, 2563]
    mean = np.mean(sales)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(regions))
    bars = ax.bar(x, sales, width=0.6, color=ORANGE, zorder=2)
    ax.axhline(mean, color=RED, ls="--", lw=1.5, zorder=3)
    ax.text(len(regions) - 0.5, mean + 60, f"均值={mean:.0f}", color=RED, ha="right", fontsize=10)
    for xi, s in zip(x, sales):
        ax.text(xi, s + 60, f"{s}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(regions)
    ax.set_ylim(0, 4500)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("带均值柱形图", fontsize=14, fontweight="bold")
    save(fig, "02_带均值柱形图.png")


# ============================================================
#  3 渐变圆角柱形图
# ============================================================
def chart03():
    items = ["口红", "面膜", "隔离", "防晒", "精华", "面霜"]
    sales = [653, 523, 648, 856, 714, 785]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(items))
    cmap = LinearSegmentedColormap.from_list("g3", ["#F8CBAD", ORANGE, "#C55A11"])
    # 用 FancyBboxPatch 画圆角柱，内部叠加渐变
    for xi, h in zip(x, sales):
        w = 0.55
        patch = FancyBboxPatch((xi - w / 2, 0), w, h,
                               boxstyle="round,pad=0.02,rounding_size=0.08",
                               fc="none", ec="none", mutation_scale=1, zorder=2)
        ax.add_patch(patch)
        grad = np.linspace(0, 1, 256).reshape(-1, 1)
        im = ax.imshow(grad, extent=[xi - w / 2 + 0.03, xi + w / 2 - 0.03, 0.03, h - 0.02],
                       aspect="auto", cmap=cmap, zorder=3, origin="lower")
        im.set_clip_path(patch)
    for xi, s in zip(x, sales):
        ax.text(xi, s + 20, f"{s}", ha="center", fontsize=9, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(items)
    ax.set_ylim(0, 1000)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("渐变圆角柱形图", fontsize=14, fontweight="bold")
    save(fig, "03_渐变圆角柱形图.png")


# ============================================================
#  4 标注柱形图
# ============================================================
def chart04():
    items = ["口红", "面膜", "隔离", "防晒", "精华", "面霜", "眼影", "气垫"]
    sales = [9221, 5102, 6571, 5760, 6321, 8612, 2645, 5321]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(items))
    cmap = LinearSegmentedColormap.from_list("g4", ["#2E75B6", "#2E75B6"])
    colors = [BLUE if s >= 6000 else "#9DC3E6" for s in sales]
    ax.bar(x, sales, width=0.6, color=colors, zorder=2)
    for xi, s in zip(x, sales):
        ax.text(xi, s + 150, f"{s:,}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(items)
    ax.set_ylim(0, 11000)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("标注柱形图（高亮销量≥6000）", fontsize=14, fontweight="bold")
    save(fig, "04_标注柱形图.png")


# ============================================================
#  5 层叠柱形图
# ============================================================
def chart05():
    quarters = ["2021Q1", "Q2", "Q3", "Q4", "2022Q1", "Q2"]
    sales = [3121, 4086, 4321, 4601, 4936, 4231]
    profit = [1020, 1421, 1502, 1623, 1781, 1432]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(quarters))
    ax.bar(x, sales, width=0.55, color=BLUE, label="销售额", zorder=2)
    ax.bar(x, profit, width=0.55, bottom=sales, color=ORANGE, label="利润额", zorder=2)
    for xi, s, p in zip(x, sales, profit):
        ax.text(xi, s + 40, f"{s}", ha="center", fontsize=8, color=BLUE)
        ax.text(xi, s + p + 40, f"{p}", ha="center", fontsize=8, color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(quarters)
    ax.set_ylim(0, 7200)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("层叠柱形图", fontsize=14, fontweight="bold")
    save(fig, "05_层叠柱形图.png")


# ============================================================
#  6 蝴蝶图（销量对比）
# ============================================================
def chart06():
    regions = ["华东", "西北", "东北", "华北", "华南"]
    s2022 = [1215, 1321, 1426, 1531, 2238]
    s2021 = [1003, 1265, 1531, 1436, 2066]
    y = np.arange(len(regions))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(y, s2022, height=0.4, color=BLUE, label="2022年销量", zorder=2)
    ax.barh(y, [-v for v in s2021], height=0.4, color=ORANGE, label="2021年销量", zorder=2)
    for yi, v in zip(y, s2022):
        ax.text(v + 30, yi, f"{v}", va="center", fontsize=8, color=BLUE)
    for yi, v in zip(y, s2021):
        ax.text(-v - 30, yi, f"{v}", va="center", ha="right", fontsize=8, color=ORANGE)
    ax.set_yticks(y); ax.set_yticklabels(regions)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-2600, 2600)
    ax.set_xlabel("左：2021  右：2022")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1.12))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("蝴蝶图（销量对比）", fontsize=14, fontweight="bold")
    save(fig, "06_蝴蝶图_销量对比.png")


# ============================================================
#  7 蝴蝶图（占比）
# ============================================================
def chart07():
    regions = ["华东", "西北", "东北", "华北", "华南"]
    p2022 = [0.36, 0.31, 0.18, 0.13, 0.09]
    p2021 = [0.42, 0.26, 0.19, 0.12, 0.05]
    y = np.arange(len(regions))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(y, p2022, height=0.4, color=BLUE, zorder=2)
    ax.barh(y, [-v for v in p2021], height=0.4, color=ORANGE, zorder=2)
    for yi, v in zip(y, p2022):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=9, color=BLUE, fontweight="bold")
    for yi, v in zip(y, p2021):
        ax.text(-v - 0.01, yi, f"{v:.0%}", va="center", ha="right", fontsize=9, color=ORANGE, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(regions)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks(np.arange(-0.5, 0.51, 0.1))
    ax.set_xticklabels([f"{abs(v)*100:.0f}%" for v in np.arange(-0.5, 0.51, 0.1)])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("蝴蝶图（占比对比：左2021 右2022）", fontsize=14, fontweight="bold")
    save(fig, "07_蝴蝶图_占比.png")


# ============================================================
#  8 数值百分比
# ============================================================
def chart08():
    regions = ["华北", "华南", "东北", "西北", "西南", "华东"]
    sales = [4321, 1946, 1536, 1872, 1369, 2109]
    yoy = [-0.136, -0.208, -0.093, -0.159, -0.179, -0.058]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(regions))
    ax.bar(x, sales, width=0.55, color=GRAY, zorder=2)
    for xi, s, y in zip(x, sales, yoy):
        ax.text(xi, s + 80, f"{s:,}", ha="center", fontsize=9)
        ax.text(xi, s - 260, f"{y:+.1%}", ha="center", fontsize=9,
                color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(regions)
    ax.set_ylim(0, 5600)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("数值百分比（销量+同比）", fontsize=14, fontweight="bold")
    save(fig, "08_数值百分比.png")


# ============================================================
#  9 对比柱形图
# ============================================================
def chart09():
    items = ["口红", "面膜", "隔离", "防晒", "精华"]
    s21 = [3568, 4135, 4436, 4106, 4936]
    s22 = [2569, 3241, 2965, 3209, 3541]
    diff = [999, 894, 1471, 897, 1395]
    x = np.arange(len(items))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - w / 2, s21, width=w, color=BLUE, label="2021销量")
    ax.bar(x + w / 2, s22, width=w, color=ORANGE, label="2022销量")
    for xi, d in zip(x, diff):
        ax.text(xi, max(s21[xi], s22[xi]) + 120, f"↓{d}", ha="center", fontsize=9, color=RED)
    ax.set_xticks(x); ax.set_xticklabels(items)
    ax.set_ylim(0, 5800)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("对比柱形图", fontsize=14, fontweight="bold")
    save(fig, "09_对比柱形图.png")


# ============================================================
# 10 甘特图
# ============================================================
def chart10():
    projs = ["制定计划", "方案设计", "资源调配", "第一阶段", "第二阶段", "第三阶段", "项目总结"]
    start = [np.datetime64("2022-03-01"), np.datetime64("2022-03-13"),
             np.datetime64("2022-03-22"), np.datetime64("2022-04-02"),
             np.datetime64("2022-04-16"), np.datetime64("2022-05-11"),
             np.datetime64("2022-05-26")]
    days = [11, 8, 10, 13, 24, 14, 7]
    prog = [0.51, 0.32, 0.21, 0.85, 0.36, 0.68, 0.68]
    y = np.arange(len(projs))[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for yi, st, d, p in zip(y, start, days, prog):
        done = d * p
        ax.barh(yi, done, left=st, height=0.45, color=BLUE, zorder=3)
        ax.barh(yi, d - done, left=st + np.timedelta64(int(done), "D"),
                height=0.45, color="#D9E2F3", zorder=3)
        ax.text(st + np.timedelta64(int(d * p) + 1, "D"), yi,
                f"{p:.0%}", va="center", fontsize=8, color="white", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(projs)
    ax.set_xlim(np.datetime64("2022-03-01"), np.datetime64("2022-06-08"))
    ax.set_title("甘特图（项目进度）", fontsize=14, fontweight="bold")
    save(fig, "10_甘特图.png")


# ============================================================
# 11 平滑折线图
# ============================================================
def chart11():
    labels = ["2021\n5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月",
              "2022\n1月", "2月", "3月"]
    sales = [146, 198, 296, 412, 506, 615, 789, 1021, 3782, 3215, 2936]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, sales, marker="o", color=BLUE, lw=2.2, zorder=3)
    # 平滑曲线
    xs = np.linspace(0, len(labels) - 1, 400)
    cs = CubicSpline(x, sales)
    ax.plot(xs, cs(xs), color=ORANGE, lw=2.4, ls="-", alpha=0.9, zorder=2)
    for xi, s in zip(x, sales):
        ax.text(xi, s + 120, f"{s}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 4600)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("平滑折线图", fontsize=14, fontweight="bold")
    save(fig, "11_平滑折线图.png")


# ============================================================
# 12 菱形走势图
# ============================================================
def chart12():
    months = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月"]
    rate = [0.536, 0.498, 0.527, 0.708, 0.609, 0.496, 0.586, 0.704]
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, rate, marker="D", color=GREEN, lw=2, markersize=7, zorder=3)
    for xi, r in zip(x, rate):
        ax.text(xi, r + 0.02, f"{r:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(months)
    ax.set_ylim(0.3, 0.85)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("菱形走势图（完成率）", fontsize=14, fontweight="bold")
    save(fig, "12_菱形走势图.png")


# ============================================================
# 13 对比折线图
# ============================================================
def chart13():
    months = ["1月", "2月", "3月", "4月", "5月", "6月"]
    y21 = [1686, 1345, 1934, 1658, 1865, 1936]
    y22 = [1385, 1846, 1654, 1936, 2564, 2236]
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, y21, marker="o", color=BLUE, lw=2, label="2021年")
    ax.plot(x, y22, marker="s", color=ORANGE, lw=2, label="2022年")
    for xi, v in zip(x, y21):
        ax.text(xi, v + 60, f"{v}", ha="center", fontsize=8, color=BLUE)
    for xi, v in zip(x, y22):
        ax.text(xi, v + 60, f"{v}", ha="center", fontsize=8, color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(months)
    ax.set_ylim(1000, 3000)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("对比折线图", fontsize=14, fontweight="bold")
    save(fig, "13_对比折线图.png")


# ============================================================
# 14 单值圆环图
# ============================================================
def chart14():
    rate = 0.85
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([rate, 1 - rate], colors=[BLUE, "#E2E2E2"], startangle=90,
           counterclock=False, wedgeprops=dict(width=0.35, edgecolor="white"))
    ax.text(0, 0.08, f"{rate:.0%}", ha="center", fontsize=30, fontweight="bold", color=BLUE)
    ax.text(0, -0.22, "完成率", ha="center", fontsize=12, color="gray")
    ax.set(aspect="equal")
    ax.set_title("单值圆环图", fontsize=14, fontweight="bold")
    save(fig, "14_单值圆环图.png")


# ============================================================
# 15 水球图
# ============================================================
def waterball(rate, wave_amp, fig, ax):
    """在 ax 中绘制水球：圆形水池 + 正弦波水面"""
    r = 1.0
    ax.add_patch(Circle((0, 0), r, fc="#DEEBF7", ec=BLUE, lw=2, zorder=1))
    theta = np.linspace(0, 2 * np.pi, 200)
    # 水面波浪（在 y=rate 附近震荡，正弦波）
    xw = np.linspace(-r, r, 300)
    yw = np.zeros_like(xw)
    n = 0
    for k in range(1, 8):
        yw += wave_amp * np.sin(k * xw * 4 + n) / k
        n += 1
    ybase = 2 * rate - 1
    ytop = ybase + yw
    poly = Polygon(np.column_stack([np.r_[xw, xw[::-1]],
                                    np.r_[ytop, np.full_like(xw, -r)]]),
                   closed=True, fc=BLUE, ec="none", zorder=2)
    # 裁剪成圆形
    circle_path = Circle((0, 0), r).get_path()
    poly.set_clip_path(PathPatch(circle_path, transform=ax.transData))
    ax.add_patch(poly)
    # 高光
    ax.add_patch(Circle((0, 0), r, fc="none", ec=BLUE, lw=2, zorder=4))
    ax.text(0, 0, f"{rate:.0%}", ha="center", va="center",
            fontsize=30, fontweight="bold", color="white", zorder=5)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
    ax.set(aspect="equal")
    ax.axis("off")


def chart15():
    fig, ax = plt.subplots(figsize=(5, 5))
    waterball(0.65, 0.05, fig, ax)
    ax.set_title("水球图（完成率65%）", fontsize=14, fontweight="bold")
    save(fig, "15_水球图.png")


# ============================================================
# 16 波浪水球图
# ============================================================
def chart16():
    fig, ax = plt.subplots(figsize=(5, 5))
    waterball(0.65, 0.09, fig, ax)  # 更大波幅体现波浪
    ax.set_title("波浪水球图（完成率65%）", fontsize=14, fontweight="bold")
    save(fig, "16_波浪水球图.png")


# ============================================================
# 17 玉玦图（缺口圆环）
# ============================================================
def chart17():
    age = [">=50", "[40,50)", "[30,40)", "[20,30)"]
    val = [0.125, 0.208333, 0.291667, 0.375]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.pie(val, labels=age, colors=PALETTE[:4], startangle=90, counterclock=False,
           autopct="%.1f%%", pctdistance=0.78, labeldistance=1.08,
           wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0, 0, "年龄\n占比", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set(aspect="equal")
    ax.set_title("玉玦图（年龄占比）", fontsize=14, fontweight="bold")
    save(fig, "17_玉玦图.png")


# ============================================================
# 18 跑道图（横向层叠：人数+占位=总数）
# ============================================================
def chart18():
    depts = ["销售部", "采购部", "工程部", "财务部", "行政部", "人力部"]
    people = [451, 326, 293, 238, 226, 130]
    total = 832
    pad = [total - p for p in people]
    y = np.arange(len(depts))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(y, people, height=0.5, color=BLUE, zorder=3)
    ax.barh(y, pad, left=people, height=0.5, color="#E2E2E2", zorder=2)
    for yi, p in zip(y, people):
        ax.text(p + 10, yi, f"{p}", va="center", fontsize=10, color="white", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(depts)
    ax.set_xlim(0, total + 20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("跑道图（部门人数）", fontsize=14, fontweight="bold")
    save(fig, "18_跑道图.png")


# ============================================================
# 19 南丁格尔圆饼图
# ============================================================
def nightingale(depts, vals, ring=False):
    n = len(depts)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    widths = np.full(n, 2 * np.pi / n)
    radii = np.array(vals) / max(vals)  # 归一化半径
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))
    ax.bar(angles, radii, width=widths, color=PALETTE[:n],
           edgecolor="white", linewidth=2, alpha=0.95, zorder=3)
    if ring:
        ax.bar(angles, radii, width=widths, color="white",
               edgecolor=PALETTE[:n], linewidth=2, zorder=2, alpha=0.6)
    # 半径标注
    for a, v in zip(angles, vals):
        ax.text(a, 1.18, f"{v:.1%}", ha="center", fontsize=9, color="black")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(depts, fontsize=10)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([])
    return fig


def chart19():
    depts = ["销售部", "采购部", "工程部", "财务部", "行政部", "人力部"]
    vals = [0.292, 0.227, 0.175, 0.136, 0.103, 0.067]
    fig = nightingale(depts, vals, ring=False)
    fig.suptitle("南丁格尔圆饼图", fontsize=14, fontweight="bold")
    save(fig, "19_南丁格尔圆饼图.png")


# ============================================================
# 20 南丁格尔圆环图
# ============================================================
def chart20():
    ages = ["[20,30)", "[30,40)", "[40,50)", ">=50"]
    vals = [0.375, 0.291667, 0.208333, 0.125]
    fig = nightingale(ages, vals, ring=True)
    fig.suptitle("南丁格尔圆环图（年龄占比）", fontsize=14, fontweight="bold")
    save(fig, "20_南丁格尔圆环图.png")


# ============================================================
# 21 南丁格尔（PPT版，人力部5%）
# ============================================================
def chart21():
    depts = ["销售部", "采购部", "工程部", "财务部", "行政部", "人力部"]
    vals = [0.292, 0.227, 0.175, 0.136, 0.103, 0.05]
    fig = nightingale(depts, vals, ring=False)
    fig.suptitle("南丁格尔（PPT版）", fontsize=14, fontweight="bold")
    save(fig, "21_南丁格尔_PPT版.png")


# ============================================================
# 22 仪表盘图
# ============================================================
def chart22():
    value = 76
    vmin, vmax = 50, 150
    frac = (value - vmin) / (vmax - vmin)
    # 弧形刻度带（270°圆弧）
    fig, ax = plt.subplots(figsize=(6, 4.6))
    start_angle = 225
    span = 270
    theta = np.linspace(np.deg2rad(start_angle), np.deg2rad(start_angle + span), 100)
    # 彩色分段
    seg_colors = ["#C00000", "#ED7D31", "#FFC000", "#70AD47"]
    segs = [(0, 0.25, seg_colors[0]), (0.25, 0.5, seg_colors[1]),
            (0.5, 0.75, seg_colors[2]), (0.75, 1.0, seg_colors[3])]
    for a0, a1, c in segs:
        idx = np.arange(int(a0 * 99), int(a1 * 99))
        ax.bar(theta[idx], np.full(len(idx), 0.9),
               width=0.035, bottom=0.0, color=c, alpha=0.85)
    # 指针
    ang = np.deg2rad(start_angle + frac * span)
    ax.plot([0, 0.55 * np.cos(ang)], [0, 0.55 * np.sin(ang)],
            color="black", lw=3, zorder=5)
    ax.add_patch(Circle((0, 0), 0.08, fc="black", zorder=6))
    # 刻度标签
    for t in [50, 75, 100, 125, 150]:
        f = (t - vmin) / (vmax - vmin)
        ta = np.deg2rad(start_angle + f * span)
        ax.text(0.78 * np.cos(ta), 0.78 * np.sin(ta), f"{t}",
                ha="center", va="center", fontsize=10)
    ax.text(0, -0.35, f"{value}", ha="center", va="center",
            fontsize=30, fontweight="bold")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.05, 1.25)
    ax.set(aspect="equal"); ax.axis("off")
    ax.set_title("仪表盘图（指针=76）", fontsize=14, fontweight="bold")
    save(fig, "22_仪表盘图.png")


# ============================================================
# 23 柱形折线图
# ============================================================
def chart23():
    years = ["2017", "2018", "2019", "2020", "2021", "2022"]
    sales = [1603, 2106, 2406, 3265, 3721, 3921]
    yoy = [0.27, 0.3138, 0.1425, 0.3570, 0.1397, 0.0537]
    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x, sales, width=0.5, color=BLUE, zorder=2)
    for xi, s in zip(x, sales):
        ax.text(xi, s + 80, f"{s}", ha="center", fontsize=9)
    ax.set_ylim(0, 4800)
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = ax.twinx()
    ax2.plot(x, [v * 100 for v in yoy], marker="o", color=ORANGE, lw=2.2, zorder=4)
    ax2.set_ylim(0, 45)
    ax2.set_ylabel("同比(%)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.spines[["top", "left"]].set_visible(False)
    ax.set_title("柱形折线图（销量+同比）", fontsize=14, fontweight="bold")
    save(fig, "23_柱形折线图.png")


# ============================================================
# 24 目标柱形图
# ============================================================
def chart24():
    items = ["口红", "面膜", "隔离", "防晒", "精华", "面霜"]
    actual = [653, 523, 648, 856, 714, 785]
    target = [700, 500, 600, 900, 600, 600]
    x = np.arange(len(items))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - w / 2, actual, width=w, color=BLUE, label="实际销量", zorder=2)
    ax.bar(x + w / 2, target, width=w, color="#A9D08E", label="目标销量", zorder=2)
    for xi, a, t in zip(x, actual, target):
        ax.text(xi - w / 2, a + 15, f"{a}", ha="center", fontsize=8)
        ax.text(xi + w / 2, t + 15, f"{t}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(items)
    ax.set_ylim(0, 1050)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("目标柱形图（实际 vs 目标）", fontsize=14, fontweight="bold")
    save(fig, "24_目标柱形图.png")


# ============================================================
# 25 子弹图
# ============================================================
def chart25():
    items = ["口红", "面膜", "隔离", "防晒", "精华", "面霜"]
    actual = [653, 523, 648, 856, 714, 785]
    target = [700, 500, 600, 900, 600, 600]
    ranges = [(0, 200), (200, 400), (400, 600)]  # 及格/良好/优秀档
    y = np.arange(len(items))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for yi in y:
        # 分层背景（深→浅）
        for i, (r0, r1) in enumerate(ranges):
            ax.barh(yi, r1 - r0, left=r0, height=0.6,
                    color=["#5B9BD5", "#8FB7E3", "#C5D9F1"][i], zorder=1)
        # 实际值（细条）
        ax.barh(yi, actual[len(items) - 1 - yi], height=0.28,
                color="#2E75B6", zorder=3)
        # 目标值（竖线标记）
        t = target[len(items) - 1 - yi]
        ax.plot([t, t], [yi - 0.35, yi + 0.35], color="black", lw=2.5, zorder=4)
        ax.text(actual[len(items) - 1 - yi] + 12, yi, f"{actual[len(items)-1-yi]}",
                va="center", fontsize=8, color="#1F4E79")
    ax.set_yticks(y); ax.set_yticklabels(items)
    ax.set_xlim(0, 1000)
    ax.set_title("子弹图（实际vs目标，背景为分级）", fontsize=14, fontweight="bold")
    save(fig, "25_子弹图.png")


# ============================================================
# 26 柱形圆
# ============================================================
def chart26():
    regions = ["华北", "华南", "东北", "西北", "西南", "华东"]
    sales = [2354, 1902, 3524, 2698, 2896, 2563]
    yoy = [0.12, 0.25, 0.16, 0.21, 0.18, 0.25]
    x = np.arange(len(regions))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x, sales, width=0.55, color="#9DC3E6", zorder=2)
    for xi, s, y in zip(x, sales, yoy):
        # 柱顶圆形标记（同比）
        ax.add_patch(Circle((xi, s + 90), 55, fc=BLUE, ec="white", zorder=5))
        ax.text(xi, s + 90, f"{y:+.0%}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
        ax.text(xi, s + 190, f"{s:,}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(regions)
    ax.set_ylim(0, 4600)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("柱形圆（销量+同比圆点）", fontsize=14, fontweight="bold")
    save(fig, "26_柱形圆.png")


# ============================================================
# 27 簇状柱形折线图
# ============================================================
def chart27():
    regions = ["华北", "华南", "东北", "西北", "西南", "华东"]
    y22 = [2354, 1902, 3524, 2698, 2896, 2563]
    y21 = [2021, 1563, 3213, 2531, 2631, 2361]
    yoy = [0.16, 0.22, 0.10, 0.07, 0.10, 0.09]
    x = np.arange(len(regions))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - w / 2, y21, width=w, color="#9DC3E6", label="2021销量", zorder=2)
    ax.bar(x + w / 2, y22, width=w, color=BLUE, label="2022销量", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(regions)
    ax.set_ylim(0, 4600)
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = ax.twinx()
    ax2.plot(x, [v * 100 for v in yoy], marker="o", color=ORANGE, lw=2.2, zorder=4)
    ax2.set_ylim(0, 30)
    ax2.set_ylabel("同比(%)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.spines[["top", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("簇状柱形折线图", fontsize=14, fontweight="bold")
    save(fig, "27_簇状柱形折线图.png")


# ============================================================
# 28 复合柱形图（月度柱 + 季度折线）
# ============================================================
def chart28():
    months = [f"{m}月" for m in range(1, 13)]
    monthly = [2354, 1902, 3524, 2698, 2896, 2563, 3156, 2896, 3621, 2635, 2963, 2789]
    quarterly = [7780, 7780, 7780, 8157, 8157, 8157, 9673, 9673, 9673, 8387, 8387, 8387]
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x, monthly, width=0.55, color="#9DC3E6", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(months, fontsize=8)
    ax.set_ylim(0, 5200)
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = ax.twinx()
    ax2.plot(x, quarterly, marker="o", color=ORANGE, lw=2.4, zorder=4)
    ax2.set_ylim(0, 12000)
    ax2.set_ylabel("季度销量", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.spines[["top", "left"]].set_visible(False)
    ax.set_title("复合柱形图（月度柱+季度折线）", fontsize=14, fontweight="bold")
    save(fig, "28_复合柱形图.png")


# ============================================================
# 29 滑珠图
# ============================================================
def slider(regions, values, labels, colors, title):
    y = np.arange(len(regions))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for yi, v in zip(y, values):
        # 底轨 0-100%
        ax.barh(yi, 1.0, left=0, height=0.45, color="#E7E6E6", zorder=1)
    for yi, (v, lab) in enumerate(zip(values, labels)):
        yi_f = len(regions) - 1 - yi
        ax.scatter(v, yi_f, s=320, color=colors[yi % len(colors)], zorder=4,
                   edgecolor="white", linewidth=1.5)
        ax.text(v, yi_f + 0.22, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_yticks(np.arange(len(regions))[::-1]); ax.set_yticklabels(regions)
    ax.set_xlim(0, 1.15)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f"{t:.0%}" for t in np.arange(0, 1.01, 0.2)])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=14, fontweight="bold")
    return fig


def chart29():
    regions = ["华东", "西北", "东北", "华北", "华南"]
    values = [0.35, 0.51, 0.62, 0.74, 0.86]
    fig = slider(regions, values, values, [BLUE] * 5, "滑珠图（完成率）")
    save(fig, "29_滑珠图.png")


# ============================================================
# 30 对比滑珠图
# ============================================================
def chart30():
    regions = ["华东", "西北", "东北", "华北", "华南"]
    v22 = [0.35, 0.51, 0.62, 0.74, 0.86]
    v21 = [0.45, 0.39, 0.53, 0.69, 0.92]
    y = np.arange(len(regions))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for yi in y:
        ax.barh(yi, 1.0, left=0, height=0.5, color="#E7E6E6", zorder=1)
    for i, yi in enumerate(y):
        ax.scatter(v22[i], yi, s=260, color=ORANGE, zorder=4, edgecolor="white", lw=1.5)
        ax.scatter(v21[i], yi, s=260, color=BLUE, zorder=3, edgecolor="white", lw=1.5)
        ax.text(v22[i], yi + 0.24, f"{v22[i]:.0%}", ha="center", fontsize=8, color=ORANGE)
        ax.text(v21[i], yi - 0.26, f"{v21[i]:.0%}", ha="center", fontsize=8, color=BLUE)
    ax.set_yticks(y); ax.set_yticklabels(regions)
    ax.set_xlim(0, 1.15)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f"{t:.0%}" for t in np.arange(0, 1.01, 0.2)])
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE,
                              markersize=9, label="2022完成率"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
                              markersize=9, label="2021完成率")],
              frameon=False, loc="upper right")
    ax.set_title("对比滑珠图（2021 vs 2022 完成率）", fontsize=14, fontweight="bold")
    save(fig, "30_对比滑珠图.png")


if __name__ == "__main__":
    funcs = [chart01, chart02, chart03, chart04, chart05, chart06, chart07,
             chart08, chart09, chart10, chart11, chart12, chart13, chart14,
             chart15, chart16, chart17, chart18, chart19, chart20, chart21,
             chart22, chart23, chart24, chart25, chart26, chart27, chart28,
             chart29, chart30]
    print("开始生成 30 张图表...")
    for i, f in enumerate(funcs, 1):
        try:
            f()
        except Exception as e:
            print(f"图表{i}失败: {e}")
    print(f"完成，共生成 {len(os.listdir(OUT))} 张图，输出目录: {OUT}")
