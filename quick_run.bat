@echo off
echo Stock Predictor Quick Run
echo ========================
echo.
echo 1. Morning Tasks (predictions, trades)
echo 2. Evening Tasks (performance, reports)
echo 3. Run Everything Once
echo 4. Start Continuous Scheduler
echo.
set /p choice="Enter choice (1-4): "

if %choice%==1 py scheduler.py --morning
if %choice%==2 py scheduler.py --evening
if %choice%==3 py scheduler.py --once
if %choice%==4 py scheduler.py

pause