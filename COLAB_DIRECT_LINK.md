# 🚀 Direct Google Colab Integration

## Quick Start Links

### Option 1: Use the Pre-made Notebook
**Direct Colab Link:** https://colab.research.google.com/github/vgshnet984/transcription-local/blob/main/transcription_colab.ipynb

### Option 2: Manual Setup
1. Go to: https://colab.research.google.com/
2. Create a new notebook
3. Follow the setup steps below

## Setup Commands for Colab

### Step 1: Clone and Setup
```python
# Clone the repository
!git clone https://github.com/vgshnet984/transcription-local.git
%cd transcription-local

# Run the setup script
!python colab_setup.py
```

### Step 2: Setup HuggingFace Token (Optional)
```python
# For speaker diarization features
!python setup_hf_token_colab.py
```

### Step 3: Upload Audio and Start
```python
# Upload audio file
from google.colab import files
uploaded = files.upload()

# Move to uploads directory
import shutil
for filename in uploaded.keys():
    shutil.move(filename, f"uploads/{filename}")
    print(f"Moved {filename} to uploads/")

# Start the server
!python src/main.py
```

## Important Notes

1. **Enable GPU**: Runtime → Change runtime type → GPU
2. **Token Setup**: Required only for speaker diarization
3. **File Size**: Keep audio files under 100MB
4. **Session Time**: Colab sessions timeout after 12 hours

## Features Available

- ✅ GPU-accelerated transcription
- ✅ Speaker diarization (with token)
- ✅ Web interface accessible from anywhere
- ✅ Multiple output formats (JSON, SRT, TXT)
- ✅ Real-time progress updates

## Troubleshooting

- **CUDA not available**: Enable GPU in Colab runtime settings
- **Out of memory**: Use smaller audio files
- **Token issues**: Re-run the HuggingFace token setup

## Support

For issues, check the main README.md or create an issue on GitHub. 