@echo off
REM Windows Task Scheduler Setup for Stock Predictor

echo Creating scheduled tasks for Stock Predictor...

REM Morning predictions (8:00 AM)
schtask /create /tn "StockPredictor_Morning" /tr "py C:\Projects\Predictor\scheduler.py --morning" /sc daily /st 08:00 /f

REM Evening performance (4:30 PM)  
schtask /create /tn "StockPredictor_Evening" /tr "py C:\Projects\Predictor\scheduler.py --evening" /sc daily /st 16:30 /f

REM Weekly analysis (Sunday 6:00 PM)
schtask /create /tn "StockPredictor_Weekly" /tr "py C:\Projects\Predictor\scheduler.py --weekly" /sc weekly /d SUN /st 18:00 /f

echo Scheduled tasks created successfully!
echo.
echo Tasks will run automatically when computer is on.
echo To view tasks: Run 'taskschd.msc'
pause