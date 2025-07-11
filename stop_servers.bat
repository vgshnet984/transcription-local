@echo off
echo Stopping Transcription Platform Servers...
echo.

echo Stopping Basic UI (port 8000)...
taskkill /f /im python.exe /fi "WINDOWTITLE eq Basic UI - Port 8000*" >nul 2>&1

echo Stopping Scripflow (port 8001)...
taskkill /f /im python.exe /fi "WINDOWTITLE eq Scripflow - Port 8001*" >nul 2>&1

echo.
echo Servers stopped.
pause 