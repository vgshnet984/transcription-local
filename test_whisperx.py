import os
import torch
import traceback

try:
    os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
    print("CUDA available:", torch.cuda.is_available())
    import whisperx
    print("WhisperX imported.")
    print("Loading WhisperX large-v3 on CUDA...")
    model = whisperx.load_model('large-v3', device='cuda')
    print("Model loaded.")
    # Load only first 30s for quick test
    print("Loading first 30s of audio...")
    audio = whisperx.load_audio('examples/sample_audio/tes1.mp3')
    import torchaudio
    waveform, sr = torchaudio.load('examples/sample_audio/tes1.mp3')
    if waveform.shape[1] > sr * 30:
        waveform = waveform[:, :sr*30]
    temp_path = 'temp_30s.wav'
    torchaudio.save(temp_path, waveform, sr)
    audio = whisperx.load_audio(temp_path)
    print("Transcribing...")
    result = model.transcribe(audio)
    print("Transcription completed!")
    print(f"Detected language: {result.get('language', 'N/A')}")
    print("First 5 segments:")
    for i, segment in enumerate(result['segments'][:5]):
        print(f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}")
    os.remove(temp_path)
except Exception as e:
    print("ERROR during WhisperX test:")
    traceback.print_exc() 