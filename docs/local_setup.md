# Local Setup Instructions

This guide will help you set up the transcription platform on your local machine.

## Prerequisites

- Python 3.9 or higher
- FFmpeg installed and available in PATH
- At least 4GB RAM (8GB recommended)
- 2GB free disk space for models

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd transcription-local
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements-local.txt
```

### 4. Install FFmpeg

**Windows:**
- Download from https://ffmpeg.org/download.html
- Add to PATH environment variable

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

### 5. Initialize Database

```bash
python scripts/manage.py init-db
```

### 6. Download Models (Optional)

```bash
python scripts/download_models.py
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=sqlite:///./transcription.db

# Server
HOST=127.0.0.1
PORT=8000
DEBUG=true

# Audio Processing
MAX_FILE_SIZE_MB=100
SAMPLE_RATE=16000

# Whisper Model
WHISPER_MODEL=base

# Speaker Diarization (Optional)
HF_TOKEN=your_huggingface_token_here
ENABLE_SPEAKER_DIARIZATION=false

# Logging
LOG_LEVEL=INFO
```

### HuggingFace Token (Optional)

For speaker diarization, you need a HuggingFace token:

1. Go to https://huggingface.co/settings/tokens
2. Create a new token
3. Add it to your `.env` file as `HF_TOKEN`

## Running the Application

### Start the Server

```bash
python src/main.py
```

The web interface will be available at: http://127.0.0.1:8000

### Using the CLI

```bash
# Test transcription
python scripts/transcribe_cli.py path/to/audio.wav

# Run comprehensive tests
python scripts/test_transcription.py path/to/audio.wav --output results.json
```

## Directory Structure

```
transcription-local/
├── src/                    # Source code
│   ├── api/               # API endpoints
│   ├── audio/             # Audio processing
│   ├── database/          # Database models
│   ├── transcription/     # Transcription engine
│   └── speakers/          # Speaker identification
├── templates/             # Web interface templates
├── tests/                 # Test suite
├── scripts/               # Utility scripts
├── uploads/               # Uploaded audio files
├── models/                # Downloaded models
├── output/                # Transcription outputs
└── logs/                  # Application logs
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# API tests
pytest tests/test_api.py -v

# Transcription tests
pytest tests/test_transcription.py -v

# Audio processing tests
pytest tests/test_audio.py -v

# Database tests
pytest tests/test_database.py -v
```

### Performance Testing

```bash
python scripts/test_transcription.py examples/sample_audio/*.wav --output performance_results.json
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Ensure FFmpeg is installed and in PATH
   - Test with: `ffmpeg -version`

2. **CUDA/GPU issues**
   - The platform works with CPU only
   - For GPU acceleration, install PyTorch with CUDA support

3. **Memory issues**
   - Use smaller Whisper models (tiny, base)
   - Process smaller audio files
   - Close other applications

4. **Database errors**
   - Delete `transcription.db` and reinitialize
   - Check file permissions

5. **Model download issues**
   - Check internet connection
   - Clear model cache: `rm -rf models/`

### Performance Optimization

1. **Use SSD storage** for faster file I/O
2. **Increase RAM** for larger audio files
3. **Use GPU** if available (requires CUDA setup)
4. **Optimize audio quality** (16kHz, mono recommended)

## Development

### Adding New Features

1. Create feature branch
2. Add tests in `tests/`
3. Update documentation
4. Run full test suite
5. Submit pull request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings
- Keep functions small and focused

### Logging

The application uses structured logging with loguru:

```python
from loguru import logger

logger.info("Processing started")
logger.error("Processing failed", error=str(e))
```

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with:
   - Error message
   - System information
   - Steps to reproduce
   - Log files (if applicable) 