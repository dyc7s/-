@echo off
chcp 65001 >nul
echo ========================================
echo   Git 仓库初始化与提交脚本
echo ========================================
echo.

cd /d "%~dp0"

echo [1/5] 初始化 Git 仓库...
git init
if errorlevel 1 (
    echo 错误：未找到 git 命令，请先安装 Git。
    echo 下载地址：https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [2/5] 配置用户信息（如已有全局配置可跳过）...
git config user.name 2>nul || git config user.name "DataAnalysis"
git config user.email 2>nul || git config user.email "analysis@example.com"

echo [3/5] 添加文件到暂存区...
git add -A

echo [4/5] 提交...
git commit -m "商业数据分析期末大作业：世界银行WDI数据分析项目

- 数据获取：世界银行WDI API，45国×15年×10指标=6750条
- 数据清洗：长表转宽表675行，缺失插值，IQR异常标记
- 5种商业模型：K-Means/波士顿矩阵/帕累托ABC/同比环比/ARIMA预测
- 17张可视化图 + 1个Pyecharts交互大屏(7组件联动)
- 论文docx + README + 合规声明 + 一键复现

[数据分析课题组/GLM-5.2]"

echo [5/5] 完成！
echo.
echo ========================================
echo   下一步：推送到 GitHub
echo ========================================
echo.
echo 1. 在 GitHub 上创建一个新仓库（不要勾选 README）
echo    访问：https://github.com/new
echo    仓库名建议：business-data-analysis-final
echo.
echo 2. 复制以下命令运行（替换 YOUR_USERNAME 为你的 GitHub 用户名）：
echo.
echo    git remote add origin https://github.com/YOUR_USERNAME/business-data-analysis-final.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 3. 如果提示输入密码，使用 GitHub Personal Access Token（非登录密码）
echo    生成地址：https://github.com/settings/tokens
echo.
pause
