#!/bin/bash

# Local Transcription Platform Environment Configuration

# Database Configuration
export DATABASE_URL="sqlite:///./transcription.db"

# File Storage
export UPLOAD_DIR="./uploads"
export MODELS_DIR="./models"
export LOGS_DIR="./logs"

# Whisper Configuration
export WHISPER_MODEL="tiny"
export LANGUAGE="en"
export DEVICE="cpu"

# Server Configuration
export HOST="0.0.0.0"
export PORT="8000"
export DEBUG="true"

# Audio Processing
export MAX_FILE_SIZE="100MB"
export SUPPORTED_FORMATS="wav,mp3,m4a,flac"
export SAMPLE_RATE="16000"

# Optional: Speaker Diarization
export ENABLE_SPEAKER_DIARIZATION="false"
export PYANNOTE_MODEL="pyannote/speaker-diarization@2.1"

# Logging
export LOG_LEVEL="INFO"
export LOG_FILE="logs/app.log"

# Security (for local development)
export SECRET_KEY="your-secret-key-change-in-production"
export ALLOWED_HOSTS="localhost,127.0.0.1"

# Performance
export MAX_CONCURRENT_JOBS="5"
export JOB_TIMEOUT="1800"

echo "Environment variables loaded for local transcription platform"