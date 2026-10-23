@echo off
title Module 5: CUDA GPU Acceleration Engine
echo =====================================================================
echo    MODULE 5: GPU PROGRAMMING WITH CUDA (HETEROGENEOUS COMPUTING)
echo    NVIDIA GeForce RTX 4050 Acceleration Dashboard
echo =====================================================================
echo.

:: Ensure python is accessible
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH. Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

echo [INFO] Starting FastAPI Uvicorn Server on http://127.0.0.1:8000 ...
echo [INFO] Press Ctrl+C in this terminal to stop the server.
echo.

:: Open default web browser after a 2-second delay in background
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000"

:: Start Uvicorn ASGI server
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload

pause
