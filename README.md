# Local Transcription Platform

A simple, local-only transcription platform that runs entirely on your machine without cloud dependencies.

## Features

- 🎵 Support for common audio formats (WAV, MP3, M4A, FLAC)
- 🗣️ Speech-to-text transcription using Whisper
- 👥 Basic speaker diarization (optional)
- 💾 Local file storage and SQLite database
- 🚀 Simple FastAPI web interface
- 🔒 No cloud dependencies - everything runs locally
- Speaker diarization can now be enabled or disabled per transcription job by setting `enable_speaker_diarization` in the config payload (API/UI). This overrides the global config for that job.

## Quick Start

### Prerequisites

- Python 3.9+
- FFmpeg (for audio processing)
- 4GB+ RAM (for Whisper models)

### Installation

1. **Clone and setup environment:**
   ```bash
   git clone <your-repo>
   cd transcription-local
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-local.txt
   ```

2. **Set up HuggingFace token (for speaker diarization):**
   ```bash
   # Copy example config and edit with your token
   cp config.json.example config.json
   # Edit config.json and replace YOUR_HUGGINGFACE_TOKEN_HERE with your actual token
   
   # Or use the set_token script
   python set_token.py YOUR_HUGGINGFACE_TOKEN
   ```

3. **Download Whisper models:**
   ```bash
   python scripts/download_models.py
   ```

4. **Initialize database:**
   ```bash
   python -c "from src.database.init_db import init_database; init_database()"
   ```

5. **Run the application:**
   ```bash
   python src/main.py
   ```

6. **Access the web interface:**
   - Open http://localhost:8000
   - Upload audio files and get transcriptions

## Usage

### Web Interface

1. Go to http://localhost:8000
2. Upload an audio file (WAV, MP3, M4A, FLAC)
3. Wait for processing (progress shown in real-time)
4. Download transcription in various formats

### API Endpoints

- `GET /health` - Health check
- `POST /upload` - Upload audio file
- `GET /transcriptions/{id}` - Get transcription
- `GET /files` - List uploaded files

### Command Line

```bash
# Test transcription
python scripts/test_transcription.py examples/sample_audio/test.wav
```

## Project Structure

```
transcription-local/
├── src/                    # Main application code
│   ├── main.py            # FastAPI application
│   ├── config.py          # Configuration
│   ├── database/          # SQLite database models
│   ├── transcription/     # Whisper transcription engine
│   ├── audio/             # Audio processing
│   └── api/               # API routes
├── scripts/               # Setup and utility scripts
├── examples/              # Sample audio files
├── uploads/               # Uploaded audio files
├── models/                # Downloaded ML models
└── logs/                  # Application logs
```

## Configuration

Copy `.env.local.example` to `.env.local` and adjust settings:

```bash
# Database
DATABASE_URL=sqlite:///./transcription.db

# File storage
UPLOAD_DIR=./uploads
MODELS_DIR=./models

# Whisper settings
WHISPER_MODEL=tiny
LANGUAGE=en

# Server settings
HOST=0.0.0.0
PORT=8000
```

## Docker (Optional)

```bash
# Build and run with Docker
docker-compose -f docker-compose.local.yml up --build
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found:**
   - Install FFmpeg: https://ffmpeg.org/download.html
   - Add to PATH

2. **Out of memory:**
   - Use smaller Whisper model (tiny/base)
   - Process shorter audio files

3. **Model download fails:**
   - Check internet connection
   - Run `python scripts/download_models.py` manually

### Logs

Check `logs/app.log` for detailed error information.

## Development

### Adding New Features

1. Audio format support: Edit `src/audio/processor.py`
2. Transcription engines: Edit `src/transcription/engine.py`
3. API endpoints: Edit `src/api/routes.py`

### Testing

```bash
# Run tests
python -m pytest tests/

# Test specific component
python scripts/test_transcription.py
```

## License

MIT License - see LICENSE file for details.