@echo off
echo Starting Transcription Platform Servers...
echo.

echo Starting Basic UI on port 8000...
start "Basic UI - Port 8000" cmd /k "python start_basic_ui.py"

echo Starting Scripflow on port 8001...
start "Scripflow - Port 8001" cmd /k "python start_scripflow.py"

echo.
echo Servers are starting...
echo Basic UI: http://localhost:8000
echo Scripflow: http://localhost:8001
echo.
echo Press any key to run automated tests...
pause >nul

echo Running enhanced automated tests...
python test_both_uis_enhanced.py

echo.
echo Test completed. Press any key to exit...
pause >nul 