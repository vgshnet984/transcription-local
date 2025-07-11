# Quick Start Reference

## 🚀 Start Commands

| UI | Command | URL | Purpose |
|----|---------|-----|---------|
| **Basic UI** | `python start_basic_ui.py` | http://localhost:8000 | Simple transcription |
| **Scripflow UI** | `python start_scripflow.py` | http://localhost:8001 | Advanced features |

## 🔧 Setup Commands

```bash
# Install dependencies
pip install -r requirements_local.txt

# Setup CUDA/cuDNN for GPU acceleration (if you have NVIDIA GPU)
python scripts/download_cudnn.py

# Set HuggingFace token (for speaker diarization)
python set_token.py YOUR_HF_TOKEN_HERE

# Test diarization setup
python test_diarization_simple.py

# Verify CUDA setup
python scripts/verify_cuda_setup.py
```

## 🎯 Quick Usage

### Basic UI (Simple)
1. `python start_basic_ui.py`
2. Open http://localhost:8000
3. Upload audio file
4. Select options (language, engine, model)
5. Enable "Speaker Diarization" if needed
6. Click "Start Transcription"
7. Download results

### Scripflow UI (Advanced)
1. `python start_scripflow.py`
2. Open http://localhost:8001
3. Upload audio file
4. Configure advanced settings
5. Monitor real-time progress
6. Export in multiple formats

## 🛠️ CLI Commands

```bash
# Quick transcription
python scripts/transcribe_cli.py uploads/audio.wav

# Fast transcription
python scripts/transcribe_cli_fast.py uploads/audio.wav

# Ultra-fast transcription
python scripts/transcribe_cli_ultra_fast.py uploads/audio.wav
```

## 🔍 Testing

```bash
# Test diarization
python test_diarization_simple.py

# Test Basic UI
python test_basic_ui.py

# Test both UIs
python test_both_uis_enhanced.py
```

## 📁 Key Directories

- `uploads/` - Audio files
- `transcripts/` - Output files
- `models/` - Downloaded models
- `logs/` - Log files

## ⚡ Performance Tips

- **Fast**: Use "faster-whisper" + "base" model
- **Accurate**: Use "whisperx" + "large" model
- **Diarization**: Enable for multi-speaker audio
- **GPU**: Automatically used if available

## 🛑 Stop Servers

- Press `Ctrl+C` in terminal
- Or run `stop_servers.bat`

---

**Note**: Both UIs share the same backend, so you can switch between them seamlessly. 