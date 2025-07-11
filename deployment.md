# Local Development Setup Guide

## Quick Start (No Cloud Required)

This guide shows you how to run the transcription platform entirely on your local machine without any cloud services or complex infrastructure.

## Prerequisites

### Minimal Requirements
- Python 3.9+
- 8GB RAM
- 10GB free disk space
- Internet connection (for downloading models)

### Optional (for full features)
- Docker (for PostgreSQL/Redis)
- FFmpeg (for advanced audio processing)
- CUDA-capable GPU (for faster processing)

## Setup Options

### Option 1: Simple Local Setup (Easiest)

This uses SQLite database and in-memory processing - perfect for getting started quickly.

**1. Clone and Setup Environment:**
```bash
git clone <your-repo>
cd transcription-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-local.txt
```

**2. Create Local Environment File:**
```bash
# .env.local
ENVIRONMENT=local
DEBUG=true
LOG_LEVEL=INFO

# Database (SQLite - no setup required)
DATABASE_URL=sqlite:///./transcription.db

# Storage (local filesystem)
STORAGE_BACKEND=local
UPLOAD_PATH=./uploads
MODELS_PATH=./models

# Processing (simple in-memory)
PROCESSING_MODE=simple
MAX_CONCURRENT_JOBS=2

# ML Models
WHISPER_MODEL_SIZE=tiny  # Start small for testing
ENABLE_GPU=false

# API
API_HOST=127.0.0.1
API_PORT=8000

# Security (development keys)
SECRET_KEY=local-dev-secret-key
JWT_SECRET_KEY=local-jwt-secret-key
```

**3. Initialize Database:**
```bash
# Create database tables
python -m src.database.init_db
```

**4. Download Models:**
```bash
# Download required models (this may take a few minutes)
python scripts/download_models.py --model-size tiny
```

**5. Start the Application:**
```bash
# Start the API server
uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

**6. Test the Setup:**
```bash
# Test API is running
curl http://localhost:8000/health

# Upload and transcribe a test file
curl -X POST "http://localhost:8000/api/v1/transcribe" \
     -F "file=@examples/sample_audio/test.wav"
```

### Option 2: Docker Local Setup (Recommended)

This gives you a more production-like environment with PostgreSQL and Redis, but still runs entirely locally.

**1. Setup with Docker Compose:**
```bash
# Copy local environment file
cp .env.local .env

# Start all services
docker-compose -f docker-compose.local.yml up -d

# Wait for services to be ready
docker-compose -f docker-compose.local.yml logs -f
```

**2. Initialize Database:**
```bash
# Run migrations
docker-compose -f docker-compose.local.yml exec api alembic upgrade head
```

**3. Download Models:**
```bash
# Download models inside container
docker-compose -f docker-compose.local.yml exec api python scripts/download_models.py
```

## Local Configuration Files

### requirements-local.txt
```txt
# Core API
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Database (SQLite for local)
sqlalchemy==2.0.23
alembic==1.13.0

# Audio Processing
librosa==0.10.1
pydub==0.25.1
soundfile==0.12.1
numpy==1.24.3
scipy==1.11.4

# Machine Learning
torch==2.1.0
transformers==4.35.2
openai-whisper==20231117

# Speaker Identification (optional for simple setup)
# pyannote-audio==3.1.1  # Uncomment if you want speaker features

# Utilities
python-dotenv==1.0.0
click==8.1.7
tqdm==4.66.1

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
```

### docker-compose.local.yml
```yaml
version: '3.8'

services:
  # PostgreSQL (local)
  postgres:
    image: postgres:13-alpine
    environment:
      POSTGRES_DB: transcription_db
      POSTGRES_USER: transcription_user
      POSTGRES_PASSWORD: local_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U transcription_user -d transcription_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis (local)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # API Service
  api:
    build:
      context: .
      dockerfile: Dockerfile.local
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://transcription_user:local_password@postgres:5432/transcription_db
      - REDIS_URL=redis://redis:6379
      - ENVIRONMENT=local
    volumes:
      - ./src:/app/src
      - ./uploads:/app/uploads
      - ./models:/app/models
      - ./examples:/app/examples
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

  # Worker (optional - for background processing)
  worker:
    build:
      context: .
      dockerfile: Dockerfile.local
    environment:
      - DATABASE_URL=postgresql://transcription_user:local_password@postgres:5432/transcription_db
      - REDIS_URL=redis://redis:6379
      - ENVIRONMENT=local
    volumes:
      - ./src:/app/src
      - ./uploads:/app/uploads
      - ./models:/app/models
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: celery -A src.jobs.celery_app worker --loglevel=info
    profiles: ["worker"]  # Start with: docker-compose --profile worker up

volumes:
  postgres_data:
  redis_data:
```

### Dockerfile.local
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements-local.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-local.txt

# Copy source code
COPY . .

# Create directories
RUN mkdir -p uploads models logs

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Simplified Project Structure (Local)

```
transcription-platform/
├── .env.local                     # Local environment variables
├── requirements-local.txt         # Minimal dependencies
├── docker-compose.local.yml       # Local Docker setup
├── Dockerfile.local              # Simple local Dockerfile
├── transcription.db              # SQLite database (auto-created)
├── uploads/                      # Local file uploads
├── models/                       # Downloaded ML models
├── logs/                        # Application logs
├── src/
│   ├── main.py                   # Simple FastAPI app
│   ├── config.py                 # Local configuration
│   ├── database/
│   │   ├── models.py             # SQLAlchemy models
│   │   └── init_db.py           # Database initialization
│   ├── transcription/
│   │   └── simple_engine.py     # Basic transcription
│   └── api/
│       └── endpoints.py         # API routes
├── scripts/
│   ├── download_models.py       # Model download utility
│   └── test_transcription.py    # Test script
└── examples/
    └── sample_audio/            # Test audio files
```

## Testing Your Local Setup

**1. Test API Health:**
```bash
curl http://localhost:8000/health
```

**2. Upload and Transcribe:**
```bash
# Create a test audio file (or use provided sample)
curl -X POST "http://localhost:8000/api/v1/transcribe" \
     -F "file=@examples/sample_audio/test.wav" \
     -F "enable_speakers=false"
```

**3. Check Transcription Results:**
```bash
# Get job status
curl http://localhost:8000/api/v1/jobs/{job_id}

# Get transcription
curl http://localhost:8000/api/v1/transcriptions/{transcription_id}
```

## Adding Features Gradually

Start simple and add features as needed:

1. **Basic Transcription** ✅ (Start here)
2. **File Management** (Upload, download, delete)
3. **Speaker Diarization** (Add pyannote.audio)
4. **Real-time Processing** (Add WebSocket support)
5. **Advanced Features** (Custom models, webhooks)
6. **Cloud Deployment** (Optional, much later)

## Troubleshooting

**Common Issues:**

1. **Model Download Fails:**
   ```bash
   # Manual model download
   python -c "import whisper; whisper.load_model('tiny')"
   ```

2. **Port Already in Use:**
   ```bash
   # Use different port
   uvicorn src.main:app --port 8001
   ```

3. **Permission Issues:**
   ```bash
   # Fix directory permissions
   mkdir -p uploads models logs
   chmod 755 uploads models logs
   ```

4. **Memory Issues:**
   ```bash
   # Use smaller model
   export WHISPER_MODEL_SIZE=tiny
   ```

## Local Development Workflow

1. **Start with Option 1** (Simple local setup)
2. **Test basic transcription** with small audio files
3. **Add features incrementally** (speakers, real-time, etc.)

