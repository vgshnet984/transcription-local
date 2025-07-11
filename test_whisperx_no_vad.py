import os
import whisperx
import torch

# Set environment variables
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

print("Testing WhisperX without VAD...")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    print("Loading WhisperX model with VAD disabled...")
    # Try to load without VAD
    model = whisperx.load_model('large-v3', device='cuda', vad=False)
    print("Model loaded successfully!")
    
    print("Loading audio file...")
    audio = whisperx.load_audio('examples/sample_audio/tes1.mp3')
    
    print("Transcribing...")
    result = model.transcribe(audio)
    
    print("Transcription completed!")
    print(f"Number of segments: {len(result['segments'])}")
    if 'language' in result:
        print(f"Detected language: {result['language']}")
    
    print("First 3 segments:")
    for i, segment in enumerate(result['segments'][:3]):
        print(f"Segment {i+1}: {segment['text']} (start: {segment['start']:.2f}s, end: {segment['end']:.2f}s)")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 