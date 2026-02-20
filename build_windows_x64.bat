@echo off
setlocal
cd /d "%~dp0"

echo [1/2] Building Windows x64 executable...
python -m PyInstaller --noconfirm --clean --onefile --windowed --icon favicon.ico --name SiyuanXiangqi Chinese_Chess_Python.py
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo [2/2] Done.
echo Output: dist\SiyuanXiangqi.exe
endlocal
