# CLI Transcription Tool Usage Examples

## Basic Usage

### Single File Transcription
```bash
# Basic transcription with default settings (whisper, base model, CPU)
python scripts/transcribe_cli.py --input audio.wav

# Specify language
python scripts/transcribe_cli.py --input audio.wav --language ta

# Use WhisperX with large model on GPU
python scripts/transcribe_cli.py --input audio.wav --engine whisperx --model large-v3 --device cuda

# Enable speaker diarization
python scripts/transcribe_cli.py --input audio.wav --speaker-diarization

# Show romanized text for Indian languages
python scripts/transcribe_cli.py --input audio.wav --language ta --romanized
```

### Batch Processing
```bash
# Process all audio files in a directory
python scripts/transcribe_cli.py --input /path/to/audio/folder

# Process with specific settings
python scripts/transcribe_cli.py --input /path/to/audio/folder --engine whisperx --model large --device cuda --language en
```

## Configuration Options

### Engine Options
- `--engine whisper` (default) - Standard Whisper
- `--engine whisperx` - Enhanced WhisperX (requires alignment models)

### Model Options
- `--model tiny` - Fastest, least accurate
- `--model base` (default) - Good balance
- `--model small` - Better accuracy
- `--model medium` - High accuracy
- `--model large` - Very high accuracy
- `--model large-v3` - Best accuracy (requires more memory)

### Device Options
- `--device cpu` (default) - CPU processing
- `--device cuda` - GPU processing (requires CUDA)

### Language Options
- `--language en` (default) - English
- `--language ta` - Tamil
- `--language sa` - Sanskrit
- `--language hi` - Hindi
- Any other language code supported by Whisper

### VAD Options
- `--vad simple` (default) - Simple voice activity detection
- `--vad webrtc` - WebRTC VAD (requires webrtcvad)
- `--vad silero` - Silero VAD

### Features
- `--speaker-diarization` - Enable speaker diarization (requires HF_TOKEN)
- `--romanized` - Show romanized text instead of native script

### Output Options
- `--no-db` - Skip saving to database
- `--verbose` - Verbose output

## Memory Optimization Examples

### For Low Memory Systems
```bash
# Use CPU with small model
python scripts/transcribe_cli.py --input audio.wav --device cpu --model tiny

# Process one file at a time
python scripts/transcribe_cli.py --input audio1.wav
python scripts/transcribe_cli.py --input audio2.wav
```

### For High Performance Systems
```bash
# Use GPU with large model
python scripts/transcribe_cli.py --input audio.wav --device cuda --model large-v3 --engine whisperx
```

## Real-world Examples

### Tamil Audio Processing
```bash
# Process Tamil audio with native script
python scripts/transcribe_cli.py --input tamil_audio.wav --language ta --engine whisperx --model large

# Process Tamil audio with romanized text
python scripts/transcribe_cli.py --input tamil_audio.wav --language ta --romanized
```

### Sanskrit Audio Processing
```bash
# Process Sanskrit audio with Devanagari script
python scripts/transcribe_cli.py --input sanskrit_audio.wav --language sa --engine whisperx --model large

# Process Sanskrit audio with romanized text
python scripts/transcribe_cli.py --input sanskrit_audio.wav --language sa --romanized
```

### Batch Processing Multiple Languages
```bash
# Process English files
python scripts/transcribe_cli.py --input english_folder --language en --engine whisperx --model large

# Process Tamil files
python scripts/transcribe_cli.py --input tamil_folder --language ta --engine whisperx --model large

# Process Sanskrit files
python scripts/transcribe_cli.py --input sanskrit_folder --language sa --engine whisperx --model large
```

## Tips for Batch Processing

1. **Memory Management**: Use smaller models for batch processing to avoid memory issues
2. **GPU Usage**: Use `--device cuda` for faster processing if you have a GPU
3. **Speaker Diarization**: Only enable for longer audio files with multiple speakers
4. **Database**: Use `--no-db` if you don't need to save results to database
5. **Interruption**: Use Ctrl+C to stop processing at any time

## Error Handling

The CLI tool will:
- Continue processing other files if one fails
- Show detailed error messages
- Provide a summary of successful vs failed transcriptions
- Save successful results to database (unless `--no-db` is used) 