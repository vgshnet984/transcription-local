# Transcription Platform - Google Colab Setup

This repository is optimized for running on Google Colab with GPU acceleration.

## Quick Setup in Google Colab

### 1. Clone the Repository
```python
!git clone https://github.com/vgshnet984/transcription-local.git
%cd transcription-local
```

### 2. Run the Setup Script
```python
!python colab_setup.py
```

### 3. Upload Audio File
```python
from google.colab import files
uploaded = files.upload()  # Upload your audio file
```

### 4. Move Audio to Uploads Directory
```python
import shutil
import os

# Move uploaded file to uploads directory
for filename in uploaded.keys():
    shutil.move(filename, f"uploads/{filename}")
    print(f"Moved {filename} to uploads/")
```

### 5. Start the Transcription Server
```python
!python src/main.py
```

### 6. Access the Web Interface
The server will provide a public URL that you can access from any browser.

## Features Available in Colab

- ✅ GPU-accelerated transcription
- ✅ Speaker diarization (with HuggingFace token)
- ✅ Real-time progress updates
- ✅ Multiple output formats (JSON, SRT, TXT)
- ✅ Web interface accessible from anywhere

## Important Notes

1. **GPU Required**: Make sure to enable GPU in Colab (Runtime → Change runtime type → GPU)
2. **HuggingFace Token**: Required for speaker diarization features
3. **File Size**: Colab has upload limits, so keep audio files under 100MB
4. **Session Time**: Colab sessions timeout after 12 hours

## Troubleshooting

### Common Issues

1. **CUDA not available**: Enable GPU in Colab runtime settings
2. **Out of memory**: Use smaller Whisper models or shorter audio files
3. **Model download fails**: Check internet connection and try again

### Performance Tips

- Use "medium" or "small" Whisper models for faster processing
- Process audio in chunks for very long files
- Enable GPU acceleration for best performance

## API Usage in Colab

```python
import requests

# Upload file
with open('uploads/your_audio.wav', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/upload', files=files)
    job_id = response.json()['job_id']

# Check status
status = requests.get(f'http://localhost:8000/transcriptions/{job_id}').json()
print(status)

# Download result
result = requests.get(f'http://localhost:8000/transcriptions/{job_id}/download').content
with open('transcript.txt', 'wb') as f:
    f.write(result)
```

## Support

For issues specific to Colab setup, check the main README.md or create an issue on GitHub. 