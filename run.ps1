# Module 5: GPU Programming with CUDA - PowerShell Launcher

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "   MODULE 5: GPU PROGRAMMING WITH CUDA (HETEROGENEOUS COMPUTING)     " -ForegroundColor Cyan
Write-Host "   NVIDIA GeForce RTX 4050 Acceleration Dashboard                    " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python was not found in PATH. Please ensure Python 3.10+ is installed." -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Starting FastAPI Uvicorn Server on http://127.0.0.1:8000 ..." -ForegroundColor Green
Write-Host "[INFO] Press Ctrl+C in this terminal to stop the server." -ForegroundColor Yellow
Write-Host ""

# Open default web browser after 2 seconds in background
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 2
    Start-Process "http://127.0.0.1:8000"
} | Out-Null

# Start Uvicorn ASGI server
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
