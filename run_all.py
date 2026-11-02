# AI生成
# -*- coding: utf-8 -*-
"""run_all.py —— 一键复现入口，依次执行 数据获取→清洗→分析→可视化→大屏"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
STEPS = ["01_load.py", "02_clean.py", "03_analysis.py", "04_visuals.py", "05_dashboard.py"]


def main():
    for s in STEPS:
        print("\n" + "#" * 70)
        print(f"# 执行 {s}")
        print("#" * 70)
        rc = subprocess.call([PY, str(ROOT / "scripts" / s)])
        if rc != 0:
            print(f"[错误] {s} 执行失败（返回码 {rc}），请检查。")
            sys.exit(rc)
    print("\n✅ 全流程完成。")
    print("产物：data/raw、data/clean、outputs/tables、figures、outputs/dashboard.html")


if __name__ == "__main__":
    main()
