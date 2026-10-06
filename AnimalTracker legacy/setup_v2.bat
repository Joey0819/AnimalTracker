@echo off
REM AnimalTracker v0.2.0 安装脚本

echo ====================================
echo AnimalTracker v0.2.0 安装程序
echo ====================================
echo.

REM 设置虚拟环境路径
set VENV_PATH=venv
set PYTHON_EXE=%VENV_PATH%\Scripts\python.exe
set PIP_EXE=%VENV_PATH%\Scripts\pip.exe

echo [1/5] 检查虚拟环境...
if exist %PYTHON_EXE% (
    echo OK - 虚拟环境已存在
) else (
    echo 创建虚拟环境...
    python -m venv %VENV_PATH%
)

echo.
echo [2/5] 升级 pip...
%PIP_EXE% install --upgrade pip --quiet

echo.
echo [3/5] 安装核心依赖...
%PIP_EXE% install --quiet ^
    numpy ^
    opencv-python ^
    scipy ^
    pandas ^
    streamlit ^
    ultralytics ^
    torch torchvision

echo.
echo [4/5] 安装额外依赖...
%PIP_EXE% install --quiet ^
    scikit-learn ^
    scikit-image ^
    plotly ^
    pillow

echo.
echo [5/5] 创建 src 模块...
if not exist src\tracking\__init__.py (
    echo. > src\tracking\__init__.py
)

if not exist src\detection\__init__.py (
    echo. > src\detection\__init__.py
)

if not exist src\behavior\__init__.py (
    echo. > src\behavior\__init__.py
)

if not exist src\visualization\__init__.py (
    echo. > src\visualization\__init__.py
)

if not exist src\utils\__init__.py (
    echo. > src\utils\__init__.py
)

echo.
echo ====================================
echo 安装完成！
echo ====================================
echo.
echo 使用方法:
echo   1. 运行应用: streamlit run app.py
echo   2. 运行测试: python tests\quick_test.py
echo.
echo 新功能 (v0.2.0):
echo   - 光流估计
echo   - 匈牙利匹配算法
echo   - 增强版追踪器
echo   - 多种预设配置
echo.
pause
