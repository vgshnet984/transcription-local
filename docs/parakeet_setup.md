# NVIDIA Parakeet Integration Guide

## Overview

NVIDIA Parakeet is a fast, efficient speech recognition model optimized for English transcription. It's been integrated into your transcription platform as an additional engine option.

## Model Details

- **Model**: `nvidia/parakeet-tdt-0.6b-v2`
- **Size**: 0.6B parameters (much smaller than Whisper large-v3)
- **Language**: English only
- **Performance**: 10-15x real-time transcription
- **Memory**: ~600MB vs 3GB for Whisper large-v3

## Installation

### Prerequisites

The Parakeet integration uses the existing `transformers` library, so no additional installation is required if you already have the platform set up.

### Verify Installation

```bash
# Test Parakeet integration
python scripts/test_parakeet.py
```

## Usage

### Command Line Interface

```bash
# Basic usage
python scripts/transcribe_cli_ultra_fast.py \
    --input "audio.wav" \
    --engine parakeet \
    --language en

# With CUDA acceleration
python scripts/transcribe_cli_ultra_fast.py \
    --input "audio.wav" \
    --engine parakeet \
    --device cuda \
    --language en
```

### Web Interface

1. Open the web interface at `http://localhost:8000`
2. Select "NVIDIA Parakeet (Fast English)" from the Transcription Engine dropdown
3. Upload your audio file
4. Click "Transcribe"

### API Usage

```python
from src.transcription.engine import TranscriptionEngine

# Initialize Parakeet engine
engine = TranscriptionEngine(
    engine="parakeet",
    device="cuda",  # or "cpu"
    enable_speaker_diarization=False  # Parakeet is English-only
)

# Transcribe audio
result = engine.transcribe("audio.wav", language="en")
print(result["text"])
```

## Performance Comparison

| Engine | Speed | Memory | Accuracy | Languages |
|--------|-------|--------|----------|-----------|
| Whisper (large-v3) | 1.5x | 3GB | High | 99+ |
| Faster-Whisper | 6x | 1.5GB | High | 99+ |
| **Parakeet** | **10-15x** | **600MB** | **Good** | **English** |
| WhisperX | 2x | 3GB | Very High | 99+ |

## Use Cases

### Best For:
- **English-only content** (podcasts, lectures, interviews)
- **Speed-critical applications** (real-time transcription)
- **Memory-constrained environments** (limited GPU memory)
- **Batch processing** (multiple files)

### Not Recommended For:
- **Multilingual content** (use Whisper or Faster-Whisper)
- **Heavy accents** (use WhisperX with large-v3)
- **Speaker diarization** (Parakeet doesn't support this)

## Configuration

### Engine Settings

Parakeet uses these default settings:
- **Model**: Fixed (no size options like Whisper)
- **Language**: English only
- **Device**: CUDA (if available) or CPU
- **Speaker Diarization**: Disabled (not supported)

### Performance Tuning

```python
# For maximum speed
engine = TranscriptionEngine(
    engine="parakeet",
    device="cuda",
    enable_speaker_diarization=False
)

# For CPU-only environments
engine = TranscriptionEngine(
    engine="parakeet",
    device="cpu",
    enable_speaker_diarization=False
)
```

## Troubleshooting

### Common Issues

1. **"NVIDIA Parakeet not available"**
   - Ensure `transformers` is installed: `pip install transformers`
   - Check internet connection (model downloads on first use)

2. **CUDA out of memory**
   - Parakeet uses much less memory than Whisper
   - Try reducing batch size if processing multiple files

3. **Language detection issues**
   - Parakeet is English-only
   - Set language parameter to "en" explicitly

4. **Slow performance**
   - Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
   - Check GPU memory usage

### Testing

Run the comprehensive test suite:

```bash
# Test Parakeet integration
python scripts/test_parakeet.py

# Test all engines
python scripts/test_models.py

# Test configuration
python scripts/test_configuration.py
```

## Integration with Existing Workflows

### Batch Processing

```bash
# Process multiple files with Parakeet
python scripts/transcribe_cli_ultra_fast.py \
    --input "audio_folder/" \
    --engine parakeet \
    --batch \
    --batch-size 8 \
    --max-workers 4
```

### Streaming

```bash
# Real-time streaming with Parakeet
python scripts/transcribe_cli_ultra_fast.py \
    --input "audio.wav" \
    --engine parakeet \
    --stream \
    --chunk-size 10
```

### API Integration

```python
# Use in your own applications
from src.transcription.engine import TranscriptionEngine

def transcribe_with_parakeet(audio_path):
    engine = TranscriptionEngine(engine="parakeet")
    return engine.transcribe(audio_path, language="en")
```

## Future Enhancements

- **Model quantization** for even smaller memory footprint
- **Batch inference** optimization
- **Custom vocabulary** support
- **Domain-specific** fine-tuning

## Support

For issues with Parakeet integration:
1. Check the troubleshooting section above
2. Run the test script: `python scripts/test_parakeet.py`
3. Review logs for detailed error messages
4. Ensure you're using English audio content 