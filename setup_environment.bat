@echo off
echo Setting up Transcription Platform Environment...
echo.

echo Setting UTF-8 encoding for Python...
set PYTHONIOENCODING=utf-8

echo Creating necessary directories...
if not exist "transcripts" mkdir transcripts
if not exist "logs" mkdir logs
if not exist "uploads" mkdir uploads
if not exist "models" mkdir models

echo Setting up Scripflow directories...
if not exist "scripflow-advanced\logs" mkdir scripflow-advanced\logs
if not exist "scripflow-advanced\static" mkdir scripflow-advanced\static

echo Environment setup complete!
echo.
echo UTF-8 encoding enabled: PYTHONIOENCODING=utf-8
echo Directories created/verified: transcripts, logs, uploads, models
echo.
pause 