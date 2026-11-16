@echo off
rem AgriAI - start the app locally (Windows). Double-click this file.
rem First run: creates a Python environment outside OneDrive and installs the packages
rem (a few minutes). Later runs: starts straight away and opens http://localhost:8501.

setlocal
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\AgriAI\venv"

if not exist "%VENV%\Scripts\python.exe" (
    echo Setting up AgriAI for the first time ^(this takes a few minutes^)...
    where python >nul 2>nul || (
        echo Python was not found. Install Python 3.12 or newer from python.org, then run this again.
        pause
        exit /b 1
    )
    python -m venv "%VENV%" || goto :failed
    "%VENV%\Scripts\python.exe" -m pip install --upgrade pip || goto :failed
    "%VENV%\Scripts\python.exe" -m pip install -r requirements.txt || goto :failed
    echo Setup complete.
)

echo Starting AgriAI at http://localhost:8501  ^(close this window to stop it^)
rem Headless mode skips Streamlit's first-run email prompt; open the browser ourselves
rem once the server has had a few seconds to start.
start "" /b cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8501"
"%VENV%\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
goto :eof

:failed
echo.
echo Setup failed. Delete the folder "%VENV%" and run this file again.
pause
exit /b 1
