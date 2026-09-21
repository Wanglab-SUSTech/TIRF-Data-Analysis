@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Local Python environment not found. Follow README.md to install it.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0classifier\colocal_classifier.py" %*
exit /b %ERRORLEVEL%
