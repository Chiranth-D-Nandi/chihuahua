@echo off
title FISSION Startup
cd /d "%~dp0"

echo ==========================================
echo   FISSION — Starting all services...
echo ==========================================
echo.

:: Load HF token from fission.env if it exists
if exist fission.env (
    for /f "tokens=1,2 delims==" %%A in (fission.env) do (
        if "%%A"=="FISSION_HF_TOKEN" set FISSION_HF_TOKEN=%%B
    )
    echo [env] Loaded HF token from fission.env
) else (
    echo [env] No fission.env found — enter token manually in the UI
)

:: 1. Start Ollama (in its own window)
echo [1/3] Starting Ollama...
start "Ollama" cmd /k "ollama serve"
timeout /t 2 /nobreak >nul

:: 2. Start MLflow (in its own window)
echo [2/3] Starting MLflow on port 5000...
start "MLflow" cmd /k ".venv\Scripts\mlflow server --host 0.0.0.0 --port 5000"
timeout /t 2 /nobreak >nul

:: 3. Start FISSION API + UI (in its own window)
echo [3/3] Starting FISSION dashboard on port 8000...
start "FISSION Dashboard" cmd /k "set FISSION_HF_TOKEN=%FISSION_HF_TOKEN% && .venv\Scripts\uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul

echo.
echo ==========================================
echo   All services started!
echo.
echo   Dashboard : http://localhost:8000
echo   MLflow    : http://localhost:5000
echo   Ollama    : http://localhost:11434
echo ==========================================
echo.
echo   Close this window anytime.
echo   To stop services, close the other windows.
pause
